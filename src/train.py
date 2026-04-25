"""DUIP training loop.

The LLM is kept frozen; only the LSTM (with its item embedding table) and
the soft-prompt projector are trained. Loss is InfoNCE / cross-entropy
over the (1 positive + N negative) candidate scores produced by
``DUIPModel``.

This module also drives the *detailed* logging stack: every optimizer
step we record loss / EMA / LR / grad-norm / throughput (samples & tokens
per second) / GPU memory / in-batch HR@K + NDCG@K. The metrics are
streamed to the console, a per-run JSONL file under
``cfg.paths.log_dir/<run_name>/``, and to Weights & Biases (when
configured / installed).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data.dataset import SessionDataset, collate
from .data.preprocess import load_items_table
from .evaluate import evaluate_split
from .logging_utils import RunLogger, gpu_mem_stats
from .models.duip import DUIPModel
from .utils import ensure_dir, hr_ndcg_from_scores, load_config, set_seed


def _cosine_schedule(optimizer, num_warmup: int, num_training_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup:
            return float(step) / float(max(1, num_warmup))
        progress = (step - num_warmup) / float(max(1, num_training_steps - num_warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def _apply_runtime_flags(cfg: Dict, logger) -> None:
    """Best-effort matmul / cudnn tweaks. All driven by cfg['runtime']."""
    rt = cfg.get("runtime", {}) or {}
    if rt.get("tf32", True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if rt.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True
    prec = rt.get("matmul_precision", "high")
    if prec:
        try:
            torch.set_float32_matmul_precision(prec)
        except Exception as e:  # pragma: no cover
            logger.log_warning("set_float32_matmul_precision(%r) failed: %s", prec, e)


def _build_train_loader(cfg: Dict, train_ds: SessionDataset) -> DataLoader:
    train_cfg = cfg["train"]
    num_workers = int(train_cfg.get("num_workers", 0))
    kwargs = dict(
        batch_size=train_cfg["micro_batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))
    return DataLoader(train_ds, **kwargs)


def run_training(config_path: str) -> Dict[str, float]:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    # Init the structured run logger first so everything downstream can
    # use it (console + per-run JSONL + W&B).
    rlog = RunLogger(cfg, run_kind="train")
    _apply_runtime_flags(cfg, rlog)

    titles, _ = load_items_table(cfg["paths"]["processed_dir"])
    num_items = len(titles)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        rlog.log_warning("CUDA not available; this will be very slow.")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    rlog.log_text("Initializing DUIP model with LLM=%s ...", cfg["model"]["llm_name"])
    model = DUIPModel(
        num_items=num_items,
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
        gradient_checkpointing=cfg["model"]["gradient_checkpointing"],
        attn_implementation=cfg["model"].get("attn_implementation", "flash_attention_2"),
        cand_chunk_size=cfg["model"].get("cand_chunk_size"),
        device=device,
    )
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_frozen = sum(p.numel() for p in model.llm.parameters())
    rlog.log_text("Trainable params (LSTM + projector + item-emb): %s",
                  f"{n_trainable:,}")
    rlog.log_text("Frozen LLM params: %s", f"{n_frozen:,}")

    # Now that the model is built, log hardware + the actually-used attn
    # implementation + parameter counts. Goes to console / JSONL / W&B.
    rlog.log_hardware({
        "attn_implementation_used": getattr(model, "attn_impl_used", None),
        "trainable_params": n_trainable,
        "frozen_llm_params": n_frozen,
        "num_items": num_items,
    })

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds = SessionDataset(
        Path(cfg["paths"]["processed_dir"]) / "train.jsonl",
        num_items=num_items,
        mode="train",
        num_negatives=cfg["train"]["num_negatives"],
        max_history_len=cfg["data"]["max_session_len"],
        seed=cfg["seed"],
    )
    max_train_sessions = cfg["train"].get("max_train_sessions")
    if max_train_sessions is not None:
        train_ds.sessions = train_ds.sessions[: int(max_train_sessions)]
        rlog.log_text("Capped train set to %d sessions for smoke test.",
                      len(train_ds.sessions))

    val_ds_path = Path(cfg["paths"]["processed_dir"]) / "val.jsonl"
    max_val_sessions = cfg["eval"].get("max_val_sessions")

    train_loader = _build_train_loader(cfg, train_ds)
    rlog.log_text("DataLoader: batch=%d workers=%d pin_memory=%s",
                  cfg["train"]["micro_batch_size"],
                  int(cfg["train"].get("num_workers", 0)),
                  torch.cuda.is_available())

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------
    optim = AdamW(
        model.trainable_parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    grad_accum = cfg["train"]["grad_accum_steps"]
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * cfg["train"]["num_epochs"]
    scheduler = _cosine_schedule(optim, cfg["train"]["warmup_steps"], total_steps)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    ckpt_dir = ensure_dir(cfg["paths"]["checkpoint_dir"])
    best_path = ckpt_dir / "best.pt"
    best_metric = -float("inf")
    bad_epochs = 0
    history: Dict[str, list] = {"train_loss": [], "val_metrics": []}
    primary_metric = "NDCG@5"
    log_ks = cfg["eval"].get("ks", [1, 5])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    global_step = 0
    last_step_t = time.time()
    last_grad_norm = float("nan")

    try:
        for epoch in range(cfg["train"]["num_epochs"]):
            train_ds.set_epoch(epoch)
            model.train()
            if cfg["model"]["freeze_llm"]:
                model.llm.eval()  # keep dropout off in the frozen LLM

            running = 0.0
            n_loss = 0
            optim.zero_grad(set_to_none=True)
            t_epoch = time.time()

            pbar = tqdm(
                train_loader,
                desc=f"epoch {epoch + 1}/{cfg['train']['num_epochs']}",
                dynamic_ncols=True,
            )
            for step, batch in enumerate(pbar):
                B_step = batch["history_ids"].shape[0]
                C_step = batch["candidates"].shape[1]

                out = model(
                    batch["history_ids"], batch["history_mask"], batch["candidates"]
                )
                B = out.scores.shape[0]
                target = torch.zeros(B, dtype=torch.long, device=out.scores.device)
                loss = F.cross_entropy(out.scores, target)

                (loss / grad_accum).backward()
                running += float(loss.item())
                n_loss += 1

                # Cheap in-batch ranking metrics for logging only.
                with torch.no_grad():
                    in_batch = hr_ndcg_from_scores(out.scores.detach(), ks=log_ks)
                loss_ema = rlog.update_loss_ema(float(loss.item()))

                # ---- optimizer step -----------------------------------
                if (step + 1) % grad_accum == 0:
                    grad_norm_t = torch.nn.utils.clip_grad_norm_(
                        model.trainable_parameters(),
                        cfg["train"]["max_grad_norm"],
                    )
                    last_grad_norm = float(grad_norm_t)
                    optim.step()
                    scheduler.step()
                    optim.zero_grad(set_to_none=True)
                    global_step += 1

                # ---- per-step logging ---------------------------------
                now = time.time()
                step_dt = now - last_step_t
                last_step_t = now
                # token count for throughput (rough but informative).
                approx_tokens = B_step * C_step * (
                    int(batch["history_ids"].shape[1]) + int(model.max_title_tokens)
                )
                tput = rlog.update_throughput(B_step, approx_tokens, step_dt)

                # tqdm postfix: short and useful at every iteration.
                postfix = {
                    "loss": f"{loss.item():.4f}",
                    "ema": f"{loss_ema:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "samp/s": f"{tput['samples_per_sec']:.1f}",
                }
                if torch.cuda.is_available():
                    postfix["mem"] = f"{torch.cuda.memory_allocated() / 1024 ** 3:.1f}G"
                pbar.set_postfix(postfix)

                # Detailed log every cfg.logging.log_every_steps optimizer steps.
                if (step + 1) % grad_accum == 0 and rlog.should_log(global_step):
                    mem = gpu_mem_stats(reset_peak=True)
                    metrics = {
                        "loss": float(loss.item()),
                        "loss_ema": float(loss_ema),
                        "lr": float(scheduler.get_last_lr()[0]),
                        "grad_norm": last_grad_norm,
                        "samples_per_sec": float(tput["samples_per_sec"]),
                        "tokens_per_sec": float(tput["tokens_per_sec"]),
                        "epoch": epoch + 1,
                        "micro_batch": B_step,
                        "candidates": C_step,
                    }
                    metrics.update(mem)
                    metrics.update({f"in_batch_{k}": v for k, v in in_batch.items()})
                    rlog.log_step(metrics, step=global_step, prefix="train")

            epoch_loss = running / max(n_loss, 1)
            history["train_loss"].append(epoch_loss)
            epoch_dt = time.time() - t_epoch
            rlog.log_epoch(
                {"loss": epoch_loss, "elapsed_sec": round(epoch_dt, 2)},
                epoch=epoch + 1,
                prefix="train",
            )

            # ---- validation ------------------------------------------
            if (epoch + 1) % cfg["train"]["eval_every_epochs"] == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                metrics = evaluate_split(
                    model,
                    val_ds_path,
                    num_items=num_items,
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
                    max_sessions=max_val_sessions,
                    desc=f"val epoch {epoch + 1}",
                    num_workers=int(cfg["eval"].get("num_workers", 0)),
                    cand_chunk_size=cfg["eval"].get("cand_chunk_size"),
                    rlog=rlog,
                    log_prefix="val",
                    log_step=global_step,
                )
                history["val_metrics"].append({"epoch": epoch + 1, **metrics})

                cur = metrics.get(primary_metric, 0.0)
                if cur > best_metric:
                    best_metric = cur
                    torch.save(model.trainable_state_dict(), best_path)
                    rlog.log_text(
                        "New best %s=%.4f, saved %s",
                        primary_metric, cur, best_path,
                    )
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= cfg["train"]["early_stop_patience"]:
                        rlog.log_text(
                            "Early stop: %d epochs without improvement.", bad_epochs,
                        )
                        break

        # ------------------------------------------------------------------
        # Persist training history alongside results.
        # ------------------------------------------------------------------
        import json
        out_dir = ensure_dir(Path(cfg["paths"]["results_path"]).parent)
        with open(out_dir / "training_history.json", "w") as f:
            json.dump(
                {"best_val_" + primary_metric: best_metric, "history": history},
                f, indent=2,
            )
        rlog.log_summary({
            "best_val_" + primary_metric: best_metric,
            "global_steps": global_step,
            "checkpoint": str(best_path),
        })
        rlog.log_text("Training complete. Best val %s = %.4f",
                      primary_metric, best_metric)
        return {"best_val_" + primary_metric: best_metric}
    finally:
        rlog.finish()
