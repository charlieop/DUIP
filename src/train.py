"""DUIP training loop.

The LLM is kept frozen; only the LSTM (with its item embedding table) and
the soft-prompt projector are trained. Loss is InfoNCE / cross-entropy
over the (1 positive + N negative) candidate scores produced by
``DUIPModel``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
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
from .models.duip import DUIPModel
from .utils import ensure_dir, get_logger, load_config, set_seed


def _cosine_schedule(optimizer, num_warmup: int, num_training_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup:
            return float(step) / float(max(1, num_warmup))
        progress = (step - num_warmup) / float(max(1, num_training_steps - num_warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def run_training(config_path: str) -> Dict[str, float]:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    logger = get_logger("train", cfg["paths"]["log_dir"])

    titles, _ = load_items_table(cfg["paths"]["processed_dir"])
    num_items = len(titles)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA not available; this will be very slow.")

    # --------------------------------------------------------------------
    # Model
    # --------------------------------------------------------------------
    logger.info("Initializing DUIP model with LLM=%s ...", cfg["model"]["llm_name"])
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
        device=device,
    )
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    logger.info("Trainable params (LSTM + projector + item-emb): %s", f"{n_trainable:,}")

    # --------------------------------------------------------------------
    # Data
    # --------------------------------------------------------------------
    train_ds = SessionDataset(
        Path(cfg["paths"]["processed_dir"]) / "train.jsonl",
        num_items=num_items,
        mode="train",
        num_negatives=cfg["train"]["num_negatives"],
        max_history_len=cfg["data"]["max_session_len"],
        seed=cfg["seed"],
    )
    val_ds_path = Path(cfg["paths"]["processed_dir"]) / "val.jsonl"

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["micro_batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate,
        pin_memory=True,
    )

    # --------------------------------------------------------------------
    # Optimizer / scheduler
    # --------------------------------------------------------------------
    optim = AdamW(
        model.trainable_parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    grad_accum = cfg["train"]["grad_accum_steps"]
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * cfg["train"]["num_epochs"]
    scheduler = _cosine_schedule(optim, cfg["train"]["warmup_steps"], total_steps)

    # --------------------------------------------------------------------
    # Loop
    # --------------------------------------------------------------------
    ckpt_dir = ensure_dir(cfg["paths"]["checkpoint_dir"])
    best_path = ckpt_dir / "best.pt"
    best_metric = -float("inf")
    bad_epochs = 0
    history: Dict[str, list] = {"train_loss": [], "val_metrics": []}
    primary_metric = "NDCG@5"

    global_step = 0
    for epoch in range(cfg["train"]["num_epochs"]):
        train_ds.set_epoch(epoch)
        model.train()
        # Keep LLM in eval mode (dropout off) since it's frozen.
        if cfg["model"]["freeze_llm"]:
            model.llm.eval()

        running = 0.0
        n_loss = 0
        optim.zero_grad(set_to_none=True)
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg['train']['num_epochs']}")
        for step, batch in enumerate(pbar):
            out = model(
                batch["history_ids"], batch["history_mask"], batch["candidates"]
            )
            # InfoNCE: target is column 0 (the positive).
            B = out.scores.shape[0]
            target = torch.zeros(B, dtype=torch.long, device=out.scores.device)
            loss = F.cross_entropy(out.scores, target)

            (loss / grad_accum).backward()
            running += float(loss.item())
            n_loss += 1

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.trainable_parameters(), cfg["train"]["max_grad_norm"]
                )
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                pbar.set_postfix(
                    loss=f"{running / max(n_loss, 1):.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            if (step + 1) % cfg["train"]["log_every"] == 0:
                logger.info(
                    "epoch %d step %d loss=%.4f lr=%.2e elapsed=%.1fs",
                    epoch + 1, step + 1,
                    running / max(n_loss, 1),
                    scheduler.get_last_lr()[0],
                    time.time() - t0,
                )

        epoch_loss = running / max(n_loss, 1)
        history["train_loss"].append(epoch_loss)
        logger.info("epoch %d done, avg loss=%.4f", epoch + 1, epoch_loss)

        # ---- validation ------------------------------------------------
        if (epoch + 1) % cfg["train"]["eval_every_epochs"] == 0:
            metrics = evaluate_split(
                model,
                val_ds_path,
                num_items=num_items,
                num_negatives=cfg["eval"]["num_negatives"],
                ks=cfg["eval"]["ks"],
                max_history_len=cfg["data"]["max_session_len"],
                seed=cfg["eval"]["negative_seed"],
                micro_batch_size=cfg["train"]["micro_batch_size"],
                desc=f"val epoch {epoch + 1}",
            )
            history["val_metrics"].append({"epoch": epoch + 1, **metrics})
            logger.info("val metrics @ epoch %d: %s", epoch + 1, metrics)

            cur = metrics.get(primary_metric, 0.0)
            if cur > best_metric:
                best_metric = cur
                torch.save(model.trainable_state_dict(), best_path)
                logger.info("New best %s=%.4f, saved %s", primary_metric, cur, best_path)
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg["train"]["early_stop_patience"]:
                    logger.info("Early stop: %d epochs without improvement.", bad_epochs)
                    break

    # Persist training history alongside results.
    import json
    out_dir = ensure_dir(Path(cfg["paths"]["results_path"]).parent)
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(
            {"best_val_" + primary_metric: best_metric, "history": history}, f, indent=2
        )
    logger.info("Training complete. Best val %s = %.4f", primary_metric, best_metric)
    return {"best_val_" + primary_metric: best_metric}
