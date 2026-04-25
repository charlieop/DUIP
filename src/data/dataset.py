"""SessionDataset: yields (history, target, candidate) triples for DUIP.

The candidate list always has the *positive* item at index 0, followed by
``num_negatives`` randomly sampled negatives. This ordering is exploited
by ``src.utils.{hit_at_k, ndcg_at_k}``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from ..utils import read_jsonl


@dataclass
class SessionExample:
    history: List[int]
    target: int
    candidates: List[int]


class SessionDataset(Dataset):
    """Sessions -> (history, target, candidates) examples.

    ``mode='train'``: pick a *random* split point inside each session each
        epoch and resample fresh negatives. Set ``epoch`` via ``set_epoch``
        to vary the RNG without resetting the loader.
    ``mode='eval'``: target is always the *last* item of the session;
        history is the prefix; negatives are sampled deterministically
        with a fixed seed so the metric is reproducible.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        num_items: int,
        *,
        mode: str = "train",
        num_negatives: int = 99,
        max_history_len: int = 20,
        seed: int = 0,
    ) -> None:
        assert mode in ("train", "eval")
        self.sessions = read_jsonl(jsonl_path)
        self.num_items = int(num_items)
        self.mode = mode
        self.num_negatives = int(num_negatives)
        self.max_history_len = int(max_history_len)
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.sessions)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __getitem__(self, idx: int) -> SessionExample:
        sess = self.sessions[idx]
        items: List[int] = list(sess["items"])
        n = len(items)
        if n < 2:
            # Should not happen because preprocess filters this, but be safe.
            raise IndexError("Session too short")

        if self.mode == "train":
            rng = random.Random((self.seed * 1_000_003 + self._epoch) * 97 + idx)
            # split point t in [1, n-1] -> history = items[:t], target = items[t]
            t = rng.randint(1, n - 1)
            history = items[:t][-self.max_history_len :]
            target = items[t]
        else:
            rng = random.Random(self.seed * 7919 + idx)
            history = items[:-1][-self.max_history_len :]
            target = items[-1]

        seen = set(history) | {target}
        negatives: List[int] = []
        while len(negatives) < self.num_negatives:
            cand = rng.randrange(self.num_items)
            if cand in seen:
                continue
            seen.add(cand)
            negatives.append(cand)

        candidates = [target] + negatives
        return SessionExample(history=history, target=target, candidates=candidates)


def collate(batch: List[SessionExample]) -> Dict[str, torch.Tensor]:
    """Right-pad histories; stack candidates."""
    max_len = max(len(ex.history) for ex in batch)
    B = len(batch)
    history_ids = torch.zeros((B, max_len), dtype=torch.long)
    history_mask = torch.zeros((B, max_len), dtype=torch.bool)
    for i, ex in enumerate(batch):
        L = len(ex.history)
        history_ids[i, :L] = torch.tensor(ex.history, dtype=torch.long)
        history_mask[i, :L] = True

    candidates = torch.tensor(
        [ex.candidates for ex in batch], dtype=torch.long
    )  # [B, 1+N]
    targets = torch.tensor([ex.target for ex in batch], dtype=torch.long)
    return {
        "history_ids": history_ids,
        "history_mask": history_mask,
        "candidates": candidates,
        "targets": targets,
    }
