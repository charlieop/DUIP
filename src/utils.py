"""Shared utilities: config loading, seeding, logging, ranking metrics."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str, log_dir: str | os.PathLike | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def read_jsonl(path: str | os.PathLike) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ----------------------------------------------------------------------------
# Ranking metrics
#
# `scores` has shape [B, C] where index 0 of the candidate dimension is the
# positive item (this is the convention used throughout this codebase).
# ----------------------------------------------------------------------------

def hit_at_k(scores: torch.Tensor, k: int) -> float:
    """Hit Rate @ K. Positive is at index 0 along the last dim."""
    ranks = _ranks_of_positive(scores)
    return float((ranks < k).float().mean().item())


def ndcg_at_k(scores: torch.Tensor, k: int) -> float:
    """NDCG @ K with a single relevant item at index 0."""
    ranks = _ranks_of_positive(scores)
    in_top = ranks < k
    gains = torch.zeros_like(ranks, dtype=torch.float32)
    valid = in_top
    gains[valid] = 1.0 / torch.log2(ranks[valid].float() + 2.0)
    return float(gains.mean().item())


def hr_ndcg_from_scores(
    scores: torch.Tensor, ks: Iterable[int] = (1, 5)
) -> Dict[str, float]:
    """Vectorized HR@K + NDCG@K for all ``k`` in ``ks`` from a [B, C] tensor.

    Cheaper than calling :func:`hit_at_k` / :func:`ndcg_at_k` once per K
    because it computes the rank-of-positive only once. Used during
    training to log in-batch ranking quality every step.
    """
    ranks = _ranks_of_positive(scores)
    out: Dict[str, float] = {}
    for k in ks:
        in_top = ranks < int(k)
        out[f"HR@{k}"] = float(in_top.float().mean().item())
        gains = torch.zeros_like(ranks, dtype=torch.float32)
        gains[in_top] = 1.0 / torch.log2(ranks[in_top].float() + 2.0)
        out[f"NDCG@{k}"] = float(gains.mean().item())
    return out


def _ranks_of_positive(scores: torch.Tensor) -> torch.Tensor:
    """Return the 0-based rank of the candidate at index 0 of each row.

    A rank of 0 means the positive item has the *highest* score (best).
    Ties are broken pessimistically (counted as worse), to avoid spurious
    metric inflation.
    """
    pos_scores = scores[:, 0:1]
    higher = (scores > pos_scores).sum(dim=1)
    return higher
