"""Evaluation: HR@K and NDCG@K with the 1-positive + N-negative protocol."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data.dataset import SessionDataset, collate
from .data.preprocess import load_items_table
from .logging_utils import RunLogger, gpu_mem_stats
from .models.duip import DUIPModel
from .utils import (
    ensure_dir,
    hit_at_k,
    load_config,
    ndcg_at_k,
    set_seed,
)


def _build_eval_loader(
    ds: SessionDataset,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


@torch.inference_mode()
def evaluate_split(
    model: DUIPModel,
    jsonl_path: str | Path,
    *,
    num_items: int,
    num_negatives: int,
    ks: List[int],
    max_history_len: int,
    seed: int,
    micro_batch_size: int = 1,
    max_sessions: Optional[int] = None,
    desc: str = "eval",
    num_workers: int = 0,
    cand_chunk_size: Optional[int] = None,
    rlog: Optional[RunLogger] = None,
    log_prefix: str = "eval",
    log_step: Optional[int] = None,
) -> Dict[str, float]:
    """Score all sessions in ``jsonl_path`` and return HR/NDCG metrics.

    ``cand_chunk_size`` (when provided) temporarily overrides
    ``model.cand_chunk_size`` for the duration of the call. Because eval
    is forward-only and skips activation storage, it usually tolerates a
    much larger chunk than training.

    If ``rlog`` is provided, per-batch progress (sessions/sec, running
    HR@1, GPU memory) is also forwarded to the structured logger / W&B
    run.
    """
    ds = SessionDataset(
        jsonl_path,
        num_items=num_items,
        mode="eval",
        num_negatives=num_negatives,
        max_history_len=max_history_len,
        seed=seed,
    )
    if max_sessions is not None:
        ds.sessions = ds.sessions[:max_sessions]

    loader = _build_eval_loader(
        ds, batch_size=micro_batch_size, num_workers=num_workers,
    )

    # Eval-time chunk override (restored in `finally`).
    prev_chunk = getattr(model, "cand_chunk_size", None)
    if cand_chunk_size is not None:
        model.cand_chunk_size = int(cand_chunk_size)

    # Free any cached blocks left over from the training step before we
    # start a long forward-only loop. Helps after a backward-OOM retry too.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    all_scores: List[torch.Tensor] = []
    n_seen = 0
    t_start = time.time()

    try:
        pbar = tqdm(loader, desc=desc, dynamic_ncols=True)
        for batch in pbar:
            out = model(
                batch["history_ids"], batch["history_mask"], batch["candidates"]
            )
            s = out.scores.detach().cpu()
            all_scores.append(s)
            n_seen += s.shape[0]

            elapsed = max(time.time() - t_start, 1e-9)
            sps = n_seen / elapsed

            # Running HR@1 over what we've seen (cheap; reuses tensors).
            running = torch.cat(all_scores, dim=0)
            running_hr1 = hit_at_k(running, 1)

            postfix = {
                "n": n_seen,
                "samp/s": f"{sps:.1f}",
                "HR@1": f"{running_hr1:.4f}",
            }
            if torch.cuda.is_available():
                postfix["mem"] = f"{torch.cuda.memory_allocated() / 1024 ** 3:.1f}G"
            pbar.set_postfix(postfix)

        scores = torch.cat(all_scores, dim=0)
        elapsed = time.time() - t_start

        metrics: Dict[str, float] = {}
        for k in ks:
            metrics[f"HR@{k}"] = hit_at_k(scores, k)
            metrics[f"NDCG@{k}"] = ndcg_at_k(scores, k)
        metrics["num_sessions"] = float(len(ds))
        metrics["elapsed_sec"] = round(elapsed, 2)
        metrics["sessions_per_sec"] = round(len(ds) / max(elapsed, 1e-9), 2)

        if rlog is not None:
            log_metrics = dict(metrics)
            log_metrics.update(gpu_mem_stats(reset_peak=False))
            rlog.log_eval(log_metrics, prefix=log_prefix, step=log_step)

        return metrics
    finally:
        # Restore the model's chunk size for any subsequent training step.
        if cand_chunk_size is not None:
            model.cand_chunk_size = prev_chunk


def run_evaluation(config_path: str, checkpoint: Optional[str] = None) -> Dict[str, float]:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    rlog = RunLogger(cfg, run_kind="eval")
    try:
        # Honour the same runtime tweaks here so eval-only invocations also
        # benefit from TF32 / cudnn.benchmark when available.
        rt = cfg.get("runtime", {}) or {}
        if rt.get("tf32", True) and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if rt.get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True

        titles, _ = load_items_table(cfg["paths"]["processed_dir"])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = DUIPModel(
            num_items=len(titles),
            item_titles=titles,
            llm_name=cfg["model"]["llm_name"],
            llm_dtype=cfg["model"]["llm_dtype"],
            item_embed_dim=cfg["model"]["item_embed_dim"],
            lstm_hidden_dim=cfg["model"]["lstm_hidden_dim"],
            lstm_num_layers=cfg["model"]["lstm_num_layers"],
            lstm_dropout=cfg["model"]["lstm_dropout"],
            num_soft_tokens=cfg["model"]["num_soft_tokens"],
            max_title_tokens=cfg["model"]["max_title_tokens"],
            hard_prompt_template=cfg["model"]["hard_prompt_template"],
            warm_start_item_embeddings=cfg["model"]["warm_start_item_embeddings"],
            freeze_llm=cfg["model"]["freeze_llm"],
            gradient_checkpointing=False,  # not needed at eval
            attn_implementation=cfg["model"].get(
                "attn_implementation", "flash_attention_2"
            ),
            cand_chunk_size=cfg["model"].get("cand_chunk_size"),
            device=device,
        )

        rlog.log_hardware({
            "attn_implementation_used": getattr(model, "attn_impl_used", None),
            "num_items": len(titles),
            "checkpoint": checkpoint,
        })

        if checkpoint is not None:
            sd = torch.load(checkpoint, map_location="cpu")
            model.load_trainable_state_dict(sd)
            rlog.log_text("Loaded trainable weights from %s", checkpoint)
        else:
            rlog.log_warning(
                "No checkpoint provided; evaluating the *untrained* model "
                "(baseline only)."
            )

        test_path = Path(cfg["paths"]["processed_dir"]) / "test.jsonl"
        metrics = evaluate_split(
            model,
            test_path,
            num_items=len(titles),
            num_negatives=cfg["eval"]["num_negatives"],
            ks=cfg["eval"]["ks"],
            max_history_len=cfg["data"]["max_session_len"],
            seed=cfg["eval"]["negative_seed"],
            micro_batch_size=int(
                cfg["eval"].get(
                    "micro_batch_size",
                    cfg["train"]["micro_batch_size"],
                )
            ),
            max_sessions=cfg["eval"].get("max_test_sessions"),
            desc="test",
            num_workers=int(cfg["eval"].get("num_workers", 0)),
            cand_chunk_size=cfg["eval"].get("cand_chunk_size"),
            rlog=rlog,
            log_prefix="test",
        )

        results_path = Path(cfg["paths"]["results_path"])
        ensure_dir(results_path.parent)
        import json
        with open(results_path, "w") as f:
            json.dump({"test": metrics}, f, indent=2)
        rlog.log_text("Test metrics: %s", metrics)
        rlog.log_text("Wrote metrics to %s", results_path)
        rlog.log_summary({
            "results_path": str(results_path),
            **{f"test_{k}": v for k, v in metrics.items()},
        })
        return metrics
    finally:
        rlog.finish()
