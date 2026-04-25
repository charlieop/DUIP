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
   candidate's title tokens, and run the frozen LLM in BF16 with
   gradient checkpointing.
5. Score = mean log-prob of the candidate's title tokens under the LLM.
   This is a length-normalized version of Eq. 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lstm_encoder import LSTMEncoder
from .soft_prompt import SoftPromptProjector


SOFT_PROMPT_MARKER = "<SOFT_PROMPT>"


# ----------------------------------------------------------------------------
# Helpers for loading the Qwen LM portion of the (multimodal) checkpoint.
# ----------------------------------------------------------------------------

def _load_qwen_lm(model_name: str, dtype: torch.dtype) -> nn.Module:
    """Load Qwen3.5-2B and return *just* the text/causal LM component.

    Works for both pure causal-LM checkpoints and the multimodal
    ImageTextToText variant (in which we discard the vision encoder).
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        return model
    except (ValueError, KeyError, OSError):
        pass

    # Fall back to the multimodal class and slice off the vision tower.
    try:
        from transformers import AutoModelForImageTextToText  # type: ignore

        full = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except Exception:
        from transformers import AutoModel

        full = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    # Common attribute names across HF VL models.
    for attr in ("language_model", "text_model", "model"):
        sub = getattr(full, attr, None)
        if sub is not None and hasattr(sub, "get_input_embeddings"):
            # If the sub-module already has an `lm_head`, treat it as the LM.
            # Otherwise try to wrap it back up; in practice Qwen exposes
            # `language_model` with everything we need.
            if hasattr(sub, "lm_head") or hasattr(full, "lm_head"):
                # Re-attach the lm_head from the parent if necessary so the
                # returned module behaves like a CausalLM.
                if not hasattr(sub, "lm_head") and hasattr(full, "lm_head"):
                    sub.lm_head = full.lm_head
                return sub
            return sub

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

        if SOFT_PROMPT_MARKER not in hard_prompt_template:
            raise ValueError(
                f"hard_prompt_template must contain the literal "
                f"'{SOFT_PROMPT_MARKER}' marker"
            )

        # ---- Load tokenizer + frozen LLM ---------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[llm_dtype]
        self.llm = _load_qwen_lm(llm_name, torch_dtype)
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
                # Older transformers signatures.
                self.llm.gradient_checkpointing_enable()
            if hasattr(self.llm, "config"):
                self.llm.config.use_cache = False
            # When the LLM is frozen but we still want gradients to flow
            # back into our soft-prompt embeddings, we need this hook so
            # that gradient checkpointing keeps the embedding-layer
            # output's requires_grad=True. Safe to call even when we
            # bypass the embedding via inputs_embeds.
            if freeze_llm and hasattr(self.llm, "enable_input_require_grads"):
                self.llm.enable_input_require_grads()

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

        # ---- Pre-compute prompt prefix/suffix tokens ---------------------
        pre, post = hard_prompt_template.split(SOFT_PROMPT_MARKER, 1)
        self._pre_template = pre        # contains {history}
        self._post_template = post      # the "Based on this..." part

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
            # mean over real tokens
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
        # normalize to unit-ish scale
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct per-example prompt embeddings.

        Returns ``(prompt_embeds, prompt_attention)`` of shape
        ``[B, T_max, H]`` and ``[B, T_max]`` respectively, left-padded so
        the prompts all end at the same column (this means we can append
        candidate tokens uniformly).
        """
        B = history_ids.shape[0]
        embed_layer = self.llm.get_input_embeddings()

        per_example: List[torch.Tensor] = []
        for b in range(B):
            history_str = self._format_history(history_ids, history_mask, b)
            pre_text = self._pre_template.format(history=history_str)
            post_text = self._post_template

            pre_ids = self.tokenizer(
                pre_text, add_special_tokens=True, return_tensors="pt"
            )["input_ids"][0].to(self.device_)
            post_ids = self.tokenizer(
                post_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0].to(self.device_)

            pre_emb = embed_layer(pre_ids).to(soft_prompts.dtype)
            post_emb = embed_layer(post_ids).to(soft_prompts.dtype)
            soft_b = soft_prompts[b].to(soft_prompts.dtype)
            per_example.append(torch.cat([pre_emb, soft_b, post_emb], dim=0))

        # Right-align (left-pad) so end positions coincide.
        T_max = max(t.shape[0] for t in per_example)
        H = per_example[0].shape[1]
        out = torch.zeros((B, T_max, H), dtype=per_example[0].dtype,
                          device=self.device_)
        attn = torch.zeros((B, T_max), dtype=torch.long, device=self.device_)
        for b, t in enumerate(per_example):
            L = t.shape[0]
            out[b, T_max - L :] = t
            attn[b, T_max - L :] = 1
        return out, attn

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
        """
        history_ids = history_ids.to(self.device_)
        history_mask = history_mask.to(self.device_)
        candidates = candidates.to(self.device_)

        # 1) LSTM hidden state h_t.
        h_t = self.encoder(history_ids, history_mask)             # [B, H_lstm]
        # 2) Soft prompt.
        soft = self.projector(h_t)                                # [B, K, H_llm]
        # 3) Build per-example prompt embeddings.
        prompt_emb, prompt_attn = self._build_prompt_embeds(
            history_ids, history_mask, soft
        )                                                          # [B, T_p, H]

        B, C = candidates.shape
        scores = torch.zeros((B, C), device=self.device_, dtype=torch.float32)

        embed_layer = self.llm.get_input_embeddings()

        # 4) For each candidate column, run a single batched forward.
        # Memory: [B, T_p + T_c, H] activations, with grad-checkpointing
        # enabled this is the largest tensor.
        for c in range(C):
            cand_idx = candidates[:, c]                            # [B]

            # Look up candidate token ids (already truncated/padded list).
            cand_id_lists = [self._cand_input_ids[int(i)] for i in cand_idx.tolist()]
            T_c = max(len(x) for x in cand_id_lists)
            cand_ids = torch.full(
                (B, T_c), fill_value=self.tokenizer.pad_token_id,
                dtype=torch.long, device=self.device_,
            )
            cand_attn = torch.zeros((B, T_c), dtype=torch.long, device=self.device_)
            for b, ids in enumerate(cand_id_lists):
                L = len(ids)
                cand_ids[b, :L] = torch.tensor(ids, device=self.device_)
                cand_attn[b, :L] = 1

            cand_emb = embed_layer(cand_ids).to(prompt_emb.dtype)

            full_emb = torch.cat([prompt_emb, cand_emb], dim=1)     # [B, T_p+T_c, H]
            full_attn = torch.cat([prompt_attn, cand_attn], dim=1)  # [B, T_p+T_c]

            out = self.llm(
                inputs_embeds=full_emb,
                attention_mask=full_attn,
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits  # [B, T_p+T_c, V]
            T_p = prompt_emb.shape[1]

            # log-prob of cand_ids[:, t] at logits[:, T_p + t - 1, :]
            # for t in 0..T_c-1.
            shift_logits = logits[:, T_p - 1 : T_p + T_c - 1, :]    # [B, T_c, V]
            log_probs = F.log_softmax(shift_logits.float(), dim=-1)
            tok_log_probs = log_probs.gather(
                2, cand_ids.unsqueeze(-1)
            ).squeeze(-1)                                           # [B, T_c]
            tok_mask = cand_attn.float()
            mean_log_prob = (tok_log_probs * tok_mask).sum(dim=1) / \
                            tok_mask.sum(dim=1).clamp(min=1.0)      # [B]

            scores[:, c] = mean_log_prob

        return DUIPOutputs(scores=scores)

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
