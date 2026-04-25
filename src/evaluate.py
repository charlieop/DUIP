"""Evaluation: HR@K and NDCG@K with the 1-positive + N-negative protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data.dataset import SessionDataset, collate
from .data.preprocess import load_items_table
from .models.duip import DUIPModel
from .utils import (
    ensure_dir,
    get_logger,
    hit_at_k,
    load_config,
    ndcg_at_k,
    set_seed,
)


@torch.no_grad()
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
) -> Dict[str, float]:
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

    loader = DataLoader(
        ds,
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    model.eval()
    all_scores: List[torch.Tensor] = []
    for batch in tqdm(loader, desc=desc):
        out = model(
            batch["history_ids"], batch["history_mask"], batch["candidates"]
        )
        all_scores.append(out.scores.detach().cpu())
    scores = torch.cat(all_scores, dim=0)

    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"HR@{k}"] = hit_at_k(scores, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(scores, k)
    metrics["num_sessions"] = float(len(ds))
    return metrics


def run_evaluation(config_path: str, checkpoint: Optional[str] = None) -> Dict[str, float]:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    logger = get_logger("evaluate", cfg["paths"]["log_dir"])

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
        device=device,
    )

    if checkpoint is not None:
        sd = torch.load(checkpoint, map_location="cpu")
        model.load_trainable_state_dict(sd)
        logger.info("Loaded trainable weights from %s", checkpoint)

    test_path = Path(cfg["paths"]["processed_dir"]) / "test.jsonl"
    metrics = evaluate_split(
        model,
        test_path,
        num_items=len(titles),
        num_negatives=cfg["eval"]["num_negatives"],
        ks=cfg["eval"]["ks"],
        max_history_len=cfg["data"]["max_session_len"],
        seed=cfg["eval"]["negative_seed"],
        micro_batch_size=cfg["train"]["micro_batch_size"],
        max_sessions=cfg["eval"].get("max_test_sessions"),
        desc="test",
    )

    results_path = Path(cfg["paths"]["results_path"])
    ensure_dir(results_path.parent)
    import json
    with open(results_path, "w") as f:
        json.dump({"test": metrics}, f, indent=2)
    logger.info("Test metrics: %s", metrics)
    logger.info("Wrote metrics to %s", results_path)
    return metrics
