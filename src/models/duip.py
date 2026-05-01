





















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
from .transformer_encoder import TransformerEncoder


SOFT_PROMPT_MARKER = "<SOFT_PROMPT>"
PROMPT_MODES = ("soft_hard", "soft_only", "hard_only")
ENCODER_TYPES = ("lstm", "transformer")

_logger = logging.getLogger(__name__)






def _attn_fallback_chain(requested: Optional[str]) -> List[Optional[str]]:




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








    impl_chain = _attn_fallback_chain(attn_impl_request)

    
    try:
        return _try_load(AutoModelForCausalLM, model_name, dtype, impl_chain)
    except (ValueError, KeyError, OSError) as e:
        _logger.info(
            "AutoModelForCausalLM could not load %s (%s); trying multimodal.",
            model_name, e.__class__.__name__,
        )

    
    full: nn.Module
    used_impl: Optional[str]
    try:
        from transformers import AutoModelForImageTextToText  

        full, used_impl = _try_load(
            AutoModelForImageTextToText, model_name, dtype, impl_chain,
        )
    except Exception:
        from transformers import AutoModel

        full, used_impl = _try_load(AutoModel, model_name, dtype, impl_chain)

    
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
    scores: torch.Tensor          


class DUIPModel(nn.Module):
    def __init__(
        self,
        *,
        num_items: int,
        item_titles: List[str],
        llm_name: str = "Qwen/Qwen3.5-2B",
        llm_dtype: str = "bfloat16",
        item_embed_dim: int = 128,
        encoder_type: str = "lstm",
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.1,
        transformer_hidden_dim: Optional[int] = None,
        transformer_num_layers: Optional[int] = None,
        transformer_num_heads: int = 4,
        transformer_ff_dim: Optional[int] = None,
        transformer_dropout: Optional[float] = None,
        transformer_max_seq_len: int = 512,
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
        prompt_mode: str = "soft_hard",
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
        self.prompt_mode = self._validate_prompt_mode(prompt_mode)
        self.encoder_type = self._validate_encoder_type(encoder_type)

        if SOFT_PROMPT_MARKER not in hard_prompt_template:
            raise ValueError(
                f"hard_prompt_template must contain the literal "
                f"'{SOFT_PROMPT_MARKER}' marker"
            )

        
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_name, trust_remote_code=True, use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
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
            
            
            
            
            if freeze_llm and hasattr(self.llm, "enable_input_require_grads"):
                self.llm.enable_input_require_grads()
        else:
            if hasattr(self.llm, "config"):
                self.llm.config.use_cache = False

        self.llm_hidden_dim = int(self.llm.get_input_embeddings().embedding_dim)

        
        self.encoder, encoder_hidden_dim = self._build_encoder(
            encoder_type=self.encoder_type,
            num_items=self.num_items,
            item_embed_dim=item_embed_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            lstm_dropout=lstm_dropout,
            transformer_hidden_dim=transformer_hidden_dim,
            transformer_num_layers=transformer_num_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_ff_dim=transformer_ff_dim,
            transformer_dropout=transformer_dropout,
            transformer_max_seq_len=transformer_max_seq_len,
        )
        self.projector = SoftPromptProjector(
            in_dim=encoder_hidden_dim,
            llm_hidden_dim=self.llm_hidden_dim,
            num_soft_tokens=num_soft_tokens,
        )
        self.num_soft_tokens = num_soft_tokens

        if warm_start_item_embeddings:
            self._warm_start_item_embeddings()

        
        self._cand_input_ids: List[List[int]] = self._tokenize_titles_for_scoring(
            self.item_titles, max_len=self.max_title_tokens
        )
        
        
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
        
        self._cand_ids_padded = cand_ids_padded.to(self.device_)
        self._cand_mask_padded = cand_mask_padded.to(self.device_)
        self._cand_T_max = int(T_c_max)

        
        pre, post = hard_prompt_template.split(SOFT_PROMPT_MARKER, 1)
        self._pre_template = pre        
        self._post_template = post      

        
        post_ids_list = self.tokenizer(
            self._post_template, add_special_tokens=False,
        )["input_ids"]
        if len(post_ids_list) == 0:
            
            
            post_ids_list = [self.tokenizer.eos_token_id]
        self._post_ids = torch.tensor(
            post_ids_list, dtype=torch.long, device=self.device_,
        )

        
        self.encoder.to(self.device_)
        self.projector.to(self.device_)

    
    
    

    @torch.no_grad()
    def _warm_start_item_embeddings(self) -> None:


        emb_layer = self.llm.get_input_embeddings()
        embed_weight = emb_layer.weight  

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
            tok_embs = embed_weight[ids].float()  
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            mean = (tok_embs * mask.unsqueeze(-1)).sum(dim=1) / denom
            title_vecs[start:end] = mean.detach().cpu()

        
        gen = torch.Generator().manual_seed(0)
        proj = torch.randn(
            self.llm_hidden_dim,
            self.encoder.item_embed_dim,
            generator=gen,
        ) / (self.llm_hidden_dim ** 0.5)
        warm = title_vecs @ proj
        warm = warm / warm.std().clamp(min=1e-6)
        self.encoder.warm_start_from(warm)

    
    
    

    def _format_history(self, history_ids: torch.Tensor, history_mask: torch.Tensor,
                        idx_in_batch: int) -> str:
        ids = history_ids[idx_in_batch][history_mask[idx_in_batch]].tolist()
        titles = [self.item_titles[i] for i in ids]
        if not titles:
            return "(no recent interactions)"
        return "; ".join(titles)

    @staticmethod
    def _validate_prompt_mode(prompt_mode: str) -> str:
        if prompt_mode not in PROMPT_MODES:
            valid = ", ".join(PROMPT_MODES)
            raise ValueError(f"prompt_mode must be one of: {valid}")
        return prompt_mode

    @staticmethod
    def _validate_encoder_type(encoder_type: str) -> str:
        encoder_type = encoder_type.lower()
        if encoder_type not in ENCODER_TYPES:
            valid = ", ".join(ENCODER_TYPES)
            raise ValueError(f"encoder_type must be one of: {valid}")
        return encoder_type

    @staticmethod
    def _build_encoder(
        *,
        encoder_type: str,
        num_items: int,
        item_embed_dim: int,
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        lstm_dropout: float,
        transformer_hidden_dim: Optional[int],
        transformer_num_layers: Optional[int],
        transformer_num_heads: int,
        transformer_ff_dim: Optional[int],
        transformer_dropout: Optional[float],
        transformer_max_seq_len: int,
    ) -> Tuple[nn.Module, int]:
        if encoder_type == "lstm":
            encoder = LSTMEncoder(
                num_items=num_items,
                item_embed_dim=item_embed_dim,
                hidden_dim=lstm_hidden_dim,
                num_layers=lstm_num_layers,
                dropout=lstm_dropout,
            )
            return encoder, int(lstm_hidden_dim)

        hidden_dim = int(
            transformer_hidden_dim
            if transformer_hidden_dim is not None
            else lstm_hidden_dim
        )
        num_layers = int(
            transformer_num_layers
            if transformer_num_layers is not None
            else lstm_num_layers
        )
        dropout = float(
            transformer_dropout
            if transformer_dropout is not None
            else lstm_dropout
        )
        encoder = TransformerEncoder(
            num_items=num_items,
            item_embed_dim=item_embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=transformer_num_heads,
            ff_dim=transformer_ff_dim,
            dropout=dropout,
            max_seq_len=transformer_max_seq_len,
        )
        return encoder, hidden_dim

    def set_prompt_mode(self, prompt_mode: str) -> None:

        self.prompt_mode = self._validate_prompt_mode(prompt_mode)

    def _build_prompt_embeds(
        self,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
        soft_prompts: Optional[torch.Tensor],
        target_dtype: torch.dtype,
        prompt_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:









        prompt_mode = self._validate_prompt_mode(prompt_mode)
        B = history_ids.shape[0]
        embed_layer = self.llm.get_input_embeddings()

        if prompt_mode == "soft_only":
            if soft_prompts is None:
                raise ValueError("soft_only prompt mode requires soft prompts.")
            prompt_emb = soft_prompts.to(target_dtype)
            prompt_attn = torch.ones(
                (B, prompt_emb.shape[1]), dtype=torch.long, device=self.device_,
            )
            return prompt_emb, prompt_attn

        
        pre_texts = [
            self._pre_template.format(
                history=self._format_history(history_ids, history_mask, b)
            )
            for b in range(B)
        ]

        
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

        pre_emb = embed_layer(pre_ids).to(target_dtype)            

        
        post_ids = self._post_ids
        T_post = int(post_ids.shape[0])
        post_emb = embed_layer(post_ids).to(target_dtype)          
        post_emb_b = post_emb.unsqueeze(0).expand(B, -1, -1)       

        if prompt_mode == "hard_only":
            prompt_emb = torch.cat([pre_emb, post_emb_b], dim=1)
            post_attn_b = torch.ones(
                (B, T_post), dtype=torch.long, device=self.device_,
            )
            prompt_attn = torch.cat([pre_attn, post_attn_b], dim=1)
            return prompt_emb, prompt_attn

        if soft_prompts is None:
            raise ValueError("soft_hard prompt mode requires soft prompts.")
        soft_emb = soft_prompts.to(target_dtype)                   
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
        emb: torch.Tensor,           
        attn: torch.Tensor,          
        cand_ids: torch.Tensor,      
        cand_mask: torch.Tensor,     
        T_p: int,
        T_c: int,
    ) -> torch.Tensor:



        
        position_ids = attn.long().cumsum(dim=-1) - 1
        position_ids = position_ids.masked_fill(attn == 0, 1)

        out = self.llm(
            inputs_embeds=emb,
            attention_mask=attn,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        
        shift_logits = out.logits[:, T_p - 1 : T_p + T_c - 1, :]   
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        tok_log_probs = log_probs.gather(
            2, cand_ids.unsqueeze(-1)
        ).squeeze(-1)                                              
        tok_mask = cand_mask.float()
        denom = tok_mask.sum(dim=1).clamp(min=1.0)
        return (tok_log_probs * tok_mask).sum(dim=1) / denom        

    def forward(
        self,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
        candidates: torch.Tensor,
    ) -> DUIPOutputs:

















        history_ids = history_ids.to(self.device_, non_blocking=True)
        history_mask = history_mask.to(self.device_, non_blocking=True)
        candidates = candidates.to(self.device_, non_blocking=True)

        B, C = candidates.shape

        embed_layer = self.llm.get_input_embeddings()
        target_dtype = embed_layer.weight.dtype

        
        soft: Optional[torch.Tensor] = None
        if self.prompt_mode != "hard_only":
            h_t = self.encoder(history_ids, history_mask)            
            soft = self.projector(h_t)                               

        
        prompt_emb, prompt_attn = self._build_prompt_embeds(
            history_ids, history_mask, soft, target_dtype, self.prompt_mode,
        )                                                            
        T_p = int(prompt_emb.shape[1])
        H = int(prompt_emb.shape[2])

        
        cand_ids = self._cand_ids_padded[candidates]                 
        cand_mask = self._cand_mask_padded[candidates]               
        T_c = int(cand_ids.shape[-1])

        cand_emb = embed_layer(cand_ids).to(target_dtype)            

        
        prompt_emb_tiled = prompt_emb.unsqueeze(1).expand(-1, C, -1, -1)
        prompt_attn_tiled = prompt_attn.unsqueeze(1).expand(-1, C, -1)

        full_emb = torch.cat([prompt_emb_tiled, cand_emb], dim=2)    
        full_attn = torch.cat([prompt_attn_tiled, cand_mask], dim=2) 

        BC = B * C
        T = T_p + T_c
        full_emb = full_emb.reshape(BC, T, H).contiguous()
        full_attn = full_attn.reshape(BC, T).contiguous()
        cand_ids_flat = cand_ids.reshape(BC, T_c)
        cand_mask_flat = cand_mask.reshape(BC, T_c)

        
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
            scores_flat = torch.cat(score_parts, dim=0)             

        return DUIPOutputs(scores=scores_flat.view(B, C))

    
    
    

    def _tokenize_titles_for_scoring(
        self, titles: List[str], max_len: int
    ) -> List[List[int]]:






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
