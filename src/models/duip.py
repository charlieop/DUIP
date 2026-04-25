"""DUIPModel — frozen Qwen3.5-2B + LSTM + soft-prompt projector.

Training-time forward
---------------------
Given a session history and a set of candidate items (positive + negatives),
the model produces a score per candidate via:

1. ``LSTMEncoder``  -> dynamic-intent hidden state h_t              (Eq. 2)
2. ``SoftPromptProjector`` -> K pseudo-token embeddings P_soft       (Eq. 3)
3. Build the input-embedding stream:
   ``[pre_text_embeds]  ++  P_soft  ++  [post_text_embeds]``
   where pre_text and post_text are the two halves of the hard prompt
   produced by splitting on the literal string ``<SOFT_PROMPT>``.
4. Tile the prompt-embedding stream once per candidate item, append the
   candidate's title tokens, and run the frozen LLM **once** over the
   flattened ``[B*C, T, H]`` tensor (Flash-Attention-2 if available, with
   automatic SDPA / eager fallback). Optionally chunked along ``B*C`` to
   bound peak memory.
5. Score = mean log-prob of the candidate's title tokens under the LLM.
   This is a length-normalized version of Eq. 7.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lstm_encoder import LSTMEncoder
from .soft_prompt import SoftPromptProjector


SOFT_PROMPT_MARKER = "<SOFT_PROMPT>"

_logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Helpers for loading the Qwen LM portion of the (multimodal) checkpoint.
# ----------------------------------------------------------------------------

def _attn_fallback_chain(requested: Optional[str]) -> List[Optional[str]]:
    """Return the order in which to try ``attn_implementation`` values.

    A ``None`` entry means "do not pass attn_implementation, let HF pick".
    """
    if not requested or requested == "auto":
        return [None]
    if requested == "flash_attention_2":
        return ["flash_attention_2", "sdpa", "eager"]
    if requested == "sdpa":
        return ["sdpa", "eager"]
    if requested == "eager":
        return ["eager"]
    return [requested, "sdpa", "eager"]


def _is_attn_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    keywords = (
        "flash", "attn_implementation", "attention implementation",
        "fa2", "sdpa", "flashattention",
    )
    return any(k in msg for k in keywords)


def _try_load(loader_cls, model_name: str, dtype: torch.dtype,
              impl_chain: List[Optional[str]]) -> Tuple[nn.Module, Optional[str]]:
    """Try the given loader class with each attn impl in ``impl_chain``.

    Returns ``(model, used_impl)``. Re-raises the *last* non-attn error if
    every impl fails for a non-attn reason; raises the *last* attn error
    only when the chain is exhausted by attn errors alone.
    """
    last_attn_err: Optional[BaseException] = None
    for impl in impl_chain:
        kwargs = dict(
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if impl is not None:
            kwargs["attn_implementation"] = impl
        try:
            model = loader_cls.from_pretrained(model_name, **kwargs)
            return model, impl
        except (ImportError, ValueError, RuntimeError) as e:
            if _is_attn_error(e):
                _logger.warning(
                    "attn_implementation=%r failed (%s); falling back.",
                    impl, e.__class__.__name__,
                )
                last_attn_err = e
                continue
            raise
    assert last_attn_err is not None
    raise last_attn_err


def _load_qwen_lm(
    model_name: str,
    dtype: torch.dtype,
    attn_impl_request: Optional[str] = "flash_attention_2",
) -> Tuple[nn.Module, Optional[str]]:
    """Load Qwen3.5-2B and return *just* the text/causal LM component.

    Works for both pure causal-LM checkpoints and the multimodal
    ImageTextToText variant (in which we discard the vision encoder).

    Returns the LM module **and** the ``attn_implementation`` actually used
    (so the caller can log it).
    """
    impl_chain = _attn_fallback_chain(attn_impl_request)

    # 1) Try as a CausalLM first.
    try:
        return _try_load(AutoModelForCausalLM, model_name, dtype, impl_chain)
    except (ValueError, KeyError, OSError) as e:
        _logger.info(
            "AutoModelForCausalLM could not load %s (%s); trying multimodal.",
            model_name, e.__class__.__name__,
        )

    # 2) Multimodal fallback: image-text-to-text or generic AutoModel.
    full: nn.Module
    used_impl: Optional[str]
    try:
        from transformers import AutoModelForImageTextToText  # type: ignore

        full, used_impl = _try_load(
            AutoModelForImageTextToText, model_name, dtype, impl_chain,
        )
    except Exception:
        from transformers import AutoModel

        full, used_impl = _try_load(AutoModel, model_name, dtype, impl_chain)

    # Common attribute names across HF VL models.
    for attr in ("language_model", "text_model", "model"):
        sub = getattr(full, attr, None)
        if sub is not None and hasattr(sub, "get_input_embeddings"):
            if hasattr(sub, "lm_head") or hasattr(full, "lm_head"):
                if not hasattr(sub, "lm_head") and hasattr(full, "lm_head"):
                    sub.lm_head = full.lm_head
                return sub, used_impl
            return sub, used_impl

    raise RuntimeError(
        f"Could not locate a language-model sub-module inside {model_name}; "
        "please open an issue with the loaded model class name: "
        f"{type(full).__name__}"
    )


@dataclass
class DUIPOutputs:
    scores: torch.Tensor          # [B, num_candidates] candidate log-prob (mean over title tokens)


class DUIPModel(nn.Module):
    def __init__(
        self,
        *,
        num_items: int,
        item_titles: List[str],
        llm_name: str = "Qwen/Qwen3.5-2B",
        llm_dtype: str = "bfloat16",
        item_embed_dim: int = 128,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.1,
        num_soft_tokens: int = 8,
        max_title_tokens: int = 24,
        hard_prompt_template: str = (
            "The user has recently interacted with the following games "
            "(most recent last): {history}.\n{soft_prompt}\nBased on this "
            "interaction history, the next game the user is most likely "
            "to want is:"
        ),
        warm_start_item_embeddings: bool = True,
        freeze_llm: bool = True,
        gradient_checkpointing: bool = True,
        attn_implementation: Optional[str] = "flash_attention_2",
        cand_chunk_size: Optional[int] = None,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()

        if num_items != len(item_titles):
            raise ValueError(
                f"num_items={num_items} but got {len(item_titles)} item titles."
            )

        self.num_items = int(num_items)
        self.item_titles = list(item_titles)
        self.llm_name = llm_name
        self.max_title_tokens = int(max_title_tokens)
        self.hard_prompt_template = hard_prompt_template
        self.device_ = torch.device(device)
        self.cand_chunk_size = int(cand_chunk_size) if cand_chunk_size else None

        if SOFT_PROMPT_MARKER not in hard_prompt_template:
            raise ValueError(
                f"hard_prompt_template must contain the literal "
                f"'{SOFT_PROMPT_MARKER}' marker"
            )

        # ---- Load tokenizer + frozen LLM ---------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_name, trust_remote_code=True, use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # We always left-pad prompts so all examples end at the same column.
        self.tokenizer.padding_side = "left"

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[llm_dtype]
        self.llm, self.attn_impl_used = _load_qwen_lm(
            llm_name, torch_dtype, attn_impl_request=attn_implementation,
        )
        _logger.info(
            "Loaded %s with attn_implementation=%r",
            llm_name, self.attn_impl_used,
        )
        self.llm.to(self.device_)

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad_(False)
            self.llm.eval()
        if gradient_checkpointing:
            try:
                self.llm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.llm.gradient_checkpointing_enable()
            if hasattr(self.llm, "config"):
                self.llm.config.use_cache = False
            # When the LLM is frozen but we still want gradients to flow
            # back into our soft-prompt embeddings, this hook keeps the
            # embedding-layer output's requires_grad=True. Safe to call
            # even when we bypass the embedding via inputs_embeds.
            if freeze_llm and hasattr(self.llm, "enable_input_require_grads"):
                self.llm.enable_input_require_grads()
        else:
            if hasattr(self.llm, "config"):
                self.llm.config.use_cache = False

        self.llm_hidden_dim = int(self.llm.get_input_embeddings().embedding_dim)

        # ---- LSTM + soft-prompt projector --------------------------------
        self.encoder = LSTMEncoder(
            num_items=self.num_items,
            item_embed_dim=item_embed_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
        )
        self.projector = SoftPromptProjector(
            in_dim=lstm_hidden_dim,
            llm_hidden_dim=self.llm_hidden_dim,
            num_soft_tokens=num_soft_tokens,
        )
        self.num_soft_tokens = num_soft_tokens

        if warm_start_item_embeddings:
            self._warm_start_item_embeddings()

        # ---- Pre-compute candidate-title token ids -----------------------
        self._cand_input_ids: List[List[int]] = self._tokenize_titles_for_scoring(
            self.item_titles, max_len=self.max_title_tokens
        )
        # Pre-pack into a [num_items, T_c_max] long tensor + companion mask
        # so that runtime candidate lookup is a single fancy-index op.
        T_c_max = max(len(ids) for ids in self._cand_input_ids)
        pad_id = int(self.tokenizer.pad_token_id)
        cand_ids_padded = torch.full(
            (self.num_items, T_c_max), fill_value=pad_id, dtype=torch.long,
        )
        cand_mask_padded = torch.zeros(
            (self.num_items, T_c_max), dtype=torch.long,
        )
        for i, ids in enumerate(self._cand_input_ids):
            L = len(ids)
            cand_ids_padded[i, :L] = torch.tensor(ids, dtype=torch.long)
            cand_mask_padded[i, :L] = 1
        # Keep on device so per-step lookups are GPU-resident.
        self._cand_ids_padded = cand_ids_padded.to(self.device_)
        self._cand_mask_padded = cand_mask_padded.to(self.device_)
        self._cand_T_max = int(T_c_max)

        # ---- Pre-compute prompt prefix/suffix templates ------------------
        pre, post = hard_prompt_template.split(SOFT_PROMPT_MARKER, 1)
        self._pre_template = pre        # contains {history}
        self._post_template = post      # the "Based on this..." part

        # The post-template is constant -> tokenize once and cache.
        post_ids_list = self.tokenizer(
            self._post_template, add_special_tokens=False,
        )["input_ids"]
        if len(post_ids_list) == 0:
            # Defensive: ensure we always have at least one post-token so
            # downstream slicing T_p-1:T_p+T_c-1 stays well-defined.
            post_ids_list = [self.tokenizer.eos_token_id]
        self._post_ids = torch.tensor(
            post_ids_list, dtype=torch.long, device=self.device_,
        )

        # Move encoder + projector to device.
        self.encoder.to(self.device_)
        self.projector.to(self.device_)

    # ------------------------------------------------------------------
    # Warm-start
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _warm_start_item_embeddings(self) -> None:
        """Init item embedding table from mean Qwen title embeddings, then
        randomly project to ``item_embed_dim``."""
        emb_layer = self.llm.get_input_embeddings()
        embed_weight = emb_layer.weight  # [V, llm_hidden_dim]

        title_vecs = torch.zeros(
            (self.num_items, self.llm_hidden_dim),
            dtype=torch.float32,
        )
        batch = 256
        for start in range(0, self.num_items, batch):
            end = min(start + batch, self.num_items)
            chunk = self.item_titles[start:end]
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_title_tokens,
                return_tensors="pt",
            )
            ids = enc["input_ids"].to(embed_weight.device)
            mask = enc["attention_mask"].to(embed_weight.device).float()
            tok_embs = embed_weight[ids].float()  # [b, T, H]
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            mean = (tok_embs * mask.unsqueeze(-1)).sum(dim=1) / denom
            title_vecs[start:end] = mean.detach().cpu()

        # Random Gaussian projection to item_embed_dim (JL preservation).
        gen = torch.Generator().manual_seed(0)
        proj = torch.randn(
            self.llm_hidden_dim,
            self.encoder.item_embed_dim,
            generator=gen,
        ) / (self.llm_hidden_dim ** 0.5)
        warm = title_vecs @ proj
        warm = warm / warm.std().clamp(min=1e-6)
        self.encoder.warm_start_from(warm)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _format_history(self, history_ids: torch.Tensor, history_mask: torch.Tensor,
                        idx_in_batch: int) -> str:
        ids = history_ids[idx_in_batch][history_mask[idx_in_batch]].tolist()
        titles = [self.item_titles[i] for i in ids]
        if not titles:
            return "(no recent interactions)"
        return "; ".join(titles)

    def _build_prompt_embeds(
        self,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
        soft_prompts: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct per-example prompt embeddings in a *single* batched
        tokenizer + embedding-lookup call.

        Returns ``(prompt_embeds, prompt_attention)`` of shape
        ``[B, T_p, H]`` and ``[B, T_p]`` respectively. The pre-text portion
        is left-padded by the tokenizer (``padding_side='left'``) so that
        the soft + post sections – which are always present – align across
        the batch and the prompt always ends at column ``T_p - 1``.
        """
        B = history_ids.shape[0]
        embed_layer = self.llm.get_input_embeddings()

        # Build the per-example pre-text strings (cheap CPU work).
        pre_texts = [
            self._pre_template.format(
                history=self._format_history(history_ids, history_mask, b)
            )
            for b in range(B)
        ]

        # Single batched tokenizer call (left-padded; fast tokenizer).
        pre_enc = self.tokenizer(
            pre_texts,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
            truncation=False,
        )
        pre_ids = pre_enc["input_ids"].to(self.device_, non_blocking=True)
        pre_attn = pre_enc["attention_mask"].to(
            self.device_, non_blocking=True,
        ).long()

        pre_emb = embed_layer(pre_ids).to(target_dtype)            # [B, T_pre, H]

        # Cached, constant post-template tokens -> embed once, expand.
        post_ids = self._post_ids
        T_post = int(post_ids.shape[0])
        post_emb = embed_layer(post_ids).to(target_dtype)          # [T_post, H]
        post_emb_b = post_emb.unsqueeze(0).expand(B, -1, -1)       # [B, T_post, H]

        soft_emb = soft_prompts.to(target_dtype)                   # [B, K, H]
        K = soft_emb.shape[1]

        prompt_emb = torch.cat([pre_emb, soft_emb, post_emb_b], dim=1)

        soft_attn = torch.ones(
            (B, K), dtype=torch.long, device=self.device_,
        )
        post_attn_b = torch.ones(
            (B, T_post), dtype=torch.long, device=self.device_,
        )
        prompt_attn = torch.cat([pre_attn, soft_attn, post_attn_b], dim=1)
        return prompt_emb, prompt_attn

    def _score_chunk(
        self,
        emb: torch.Tensor,           # [N, T, H]
        attn: torch.Tensor,          # [N, T]
        cand_ids: torch.Tensor,      # [N, T_c]
        cand_mask: torch.Tensor,     # [N, T_c]
        T_p: int,
        T_c: int,
    ) -> torch.Tensor:
        """Run one LLM forward over ``N`` (sub-batch) sequences and return
        the per-row mean log-prob over the candidate-token positions.
        """
        # Position ids that respect left-padding (cumsum over the mask).
        position_ids = attn.long().cumsum(dim=-1) - 1
        position_ids = position_ids.masked_fill(attn == 0, 1)

        out = self.llm(
            inputs_embeds=emb,
            attention_mask=attn,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        # Logit at index T_p-1+t predicts the candidate token at position t.
        shift_logits = out.logits[:, T_p - 1 : T_p + T_c - 1, :]   # [N, T_c, V]
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        tok_log_probs = log_probs.gather(
            2, cand_ids.unsqueeze(-1)
        ).squeeze(-1)                                              # [N, T_c]
        tok_mask = cand_mask.float()
        denom = tok_mask.sum(dim=1).clamp(min=1.0)
        return (tok_log_probs * tok_mask).sum(dim=1) / denom        # [N]

    def forward(
        self,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
        candidates: torch.Tensor,
    ) -> DUIPOutputs:
        """Compute mean log-prob scores for each candidate item.

        Shapes:
            history_ids: [B, L]
            history_mask: [B, L] (bool)
            candidates: [B, C]   (item indices; column 0 = positive)
        Returns:
            scores: [B, C] -- higher is more likely.

        Implementation note
        -------------------
        Unlike the per-candidate Python loop in earlier revisions, this
        version constructs a single ``[B*C, T_p+T_c, H]`` tensor and runs
        the frozen LLM **once** (optionally chunked along the ``B*C`` axis
        via ``self.cand_chunk_size`` to bound peak activation memory).
        This lets Flash-Attention-2 / SDPA actually saturate the GPU.
        """
        history_ids = history_ids.to(self.device_, non_blocking=True)
        history_mask = history_mask.to(self.device_, non_blocking=True)
        candidates = candidates.to(self.device_, non_blocking=True)

        B, C = candidates.shape

        # 1) LSTM hidden state h_t.
        h_t = self.encoder(history_ids, history_mask)               # [B, H_lstm]
        # 2) Soft prompt.
        soft = self.projector(h_t)                                  # [B, K, H_llm]

        embed_layer = self.llm.get_input_embeddings()
        target_dtype = embed_layer.weight.dtype

        # 3) Prompt embeddings (single batched tokenizer call).
        prompt_emb, prompt_attn = self._build_prompt_embeds(
            history_ids, history_mask, soft, target_dtype,
        )                                                            # [B, T_p, H], [B, T_p]
        T_p = int(prompt_emb.shape[1])
        H = int(prompt_emb.shape[2])

        # 4) Candidate ids/embeddings via fancy-index into the cached table.
        cand_ids = self._cand_ids_padded[candidates]                 # [B, C, T_c]
        cand_mask = self._cand_mask_padded[candidates]               # [B, C, T_c]
        T_c = int(cand_ids.shape[-1])

        cand_emb = embed_layer(cand_ids).to(target_dtype)            # [B, C, T_c, H]

        # 5) Tile prompt over candidates and concatenate, then flatten.
        prompt_emb_tiled = prompt_emb.unsqueeze(1).expand(-1, C, -1, -1)
        prompt_attn_tiled = prompt_attn.unsqueeze(1).expand(-1, C, -1)

        full_emb = torch.cat([prompt_emb_tiled, cand_emb], dim=2)    # [B, C, T, H]
        full_attn = torch.cat([prompt_attn_tiled, cand_mask], dim=2) # [B, C, T]

        BC = B * C
        T = T_p + T_c
        full_emb = full_emb.reshape(BC, T, H).contiguous()
        full_attn = full_attn.reshape(BC, T).contiguous()
        cand_ids_flat = cand_ids.reshape(BC, T_c)
        cand_mask_flat = cand_mask.reshape(BC, T_c)

        # 6) One (or a few chunked) LLM forward(s).
        chunk = self.cand_chunk_size or BC
        if chunk >= BC:
            scores_flat = self._score_chunk(
                full_emb, full_attn, cand_ids_flat, cand_mask_flat,
                T_p=T_p, T_c=T_c,
            )
        else:
            score_parts = []
            for start in range(0, BC, chunk):
                end = min(start + chunk, BC)
                score_parts.append(self._score_chunk(
                    full_emb[start:end],
                    full_attn[start:end],
                    cand_ids_flat[start:end],
                    cand_mask_flat[start:end],
                    T_p=T_p, T_c=T_c,
                ))
            scores_flat = torch.cat(score_parts, dim=0)             # [BC]

        return DUIPOutputs(scores=scores_flat.view(B, C))

    # ------------------------------------------------------------------
    # Tokenization helper
    # ------------------------------------------------------------------

    def _tokenize_titles_for_scoring(
        self, titles: List[str], max_len: int
    ) -> List[List[int]]:
        """Tokenize each title (no special tokens) for scoring after the prompt.

        We prepend a single space so the title behaves as a continuation
        of the prompt for BPE/byte-level tokenizers, then optionally append
        the EOS token so the model is forced to score 'end of name' too.
        """
        out: List[List[int]] = []
        eos_id = self.tokenizer.eos_token_id
        for t in titles:
            txt = " " + t.strip()
            ids = self.tokenizer(
                txt, add_special_tokens=False, truncation=True, max_length=max_len,
            )["input_ids"]
            if eos_id is not None:
                if len(ids) >= max_len:
                    ids = ids[: max_len - 1]
                ids = ids + [eos_id]
            out.append(ids)
        return out

    # ------------------------------------------------------------------
    # Trainable / persisted parameter groups
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        return list(self.encoder.parameters()) + list(self.projector.parameters())

    def trainable_state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "projector": self.projector.state_dict(),
        }

    def load_trainable_state_dict(self, sd: dict) -> None:
        self.encoder.load_state_dict(sd["encoder"])
        self.projector.load_state_dict(sd["projector"])
