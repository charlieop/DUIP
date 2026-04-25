"""Sessionize the Amazon Reviews 2023 Games subset and produce splits.

Sessions are formed per user, grouped by the calendar day of the
``timestamp`` field (paper §4.1). Items are identified by ``parent_asin``
(so product variants collapse together). The final per-user session list
is split chronologically 80/10/10 across *sessions* (not users), matching
the protocol described in the paper.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from ..utils import ensure_dir, get_logger, write_jsonl


def _to_day(ts: int) -> str:
    # Amazon Reviews 2023 timestamps are in milliseconds since epoch.
    return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")


def build_sessions(
    raw_dir: str,
    processed_dir: str,
    *,
    min_session_len: int = 2,
    max_session_len: int = 20,
    min_user_sessions: int = 2,
    min_item_freq: int = 5,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, int]:
    """Run the full preprocessing pipeline.

    Writes ``items.json``, ``train.jsonl``, ``val.jsonl``, ``test.jsonl``
    under ``processed_dir`` and returns dataset statistics.
    """
    logger = get_logger("data.preprocess")
    raw = Path(raw_dir)
    out = ensure_dir(processed_dir)

    logger.info("Reading reviews / meta parquet files from %s ...", raw)
    reviews = pd.read_parquet(raw / "reviews.parquet")
    meta = pd.read_parquet(raw / "meta.parquet")

    # ---- 1. clean & merge titles -------------------------------------------
    reviews = reviews.dropna(subset=["user_id", "parent_asin", "timestamp"])
    reviews["timestamp"] = reviews["timestamp"].astype("int64")

    meta = meta.dropna(subset=["parent_asin"])
    meta["title"] = meta["title"].fillna("").astype(str).str.strip()
    meta = meta[meta["title"].str.len() > 0]
    meta = meta.drop_duplicates(subset=["parent_asin"])

    title_map: Dict[str, str] = dict(zip(meta["parent_asin"], meta["title"]))
    reviews = reviews[reviews["parent_asin"].isin(title_map)]
    logger.info("After title-join: %d reviews, %d unique items.",
                len(reviews), reviews["parent_asin"].nunique())

    # ---- 2. iterative k-core on item / user (single pass each) -------------
    item_counts = reviews["parent_asin"].value_counts()
    keep_items = set(item_counts[item_counts >= min_item_freq].index)
    reviews = reviews[reviews["parent_asin"].isin(keep_items)]

    # ---- 3. sessionize per user per day ------------------------------------
    logger.info("Sessionizing %d reviews ...", len(reviews))
    reviews = reviews.sort_values(["user_id", "timestamp"], kind="mergesort")
    reviews["day"] = reviews["timestamp"].map(_to_day)

    sessions: List[Dict] = []
    for (uid, day), grp in reviews.groupby(["user_id", "day"], sort=False):
        items = grp["parent_asin"].tolist()
        # de-duplicate consecutive duplicates within a day while preserving order
        deduped: List[str] = []
        for it in items:
            if not deduped or deduped[-1] != it:
                deduped.append(it)
        if len(deduped) < min_session_len:
            continue
        if len(deduped) > max_session_len:
            deduped = deduped[-max_session_len:]
        sessions.append(
            {
                "user_id": uid,
                "day": day,
                "session_ts": int(grp["timestamp"].min()),
                "items": deduped,
            }
        )

    logger.info("Built %d raw sessions.", len(sessions))

    # ---- 4. drop users with too few sessions -------------------------------
    user_session_counts: Counter = Counter(s["user_id"] for s in sessions)
    sessions = [s for s in sessions if user_session_counts[s["user_id"]] >= min_user_sessions]
    logger.info("After min_user_sessions=%d filter: %d sessions.",
                min_user_sessions, len(sessions))

    # ---- 5. build item id table from items that survived -------------------
    surviving_items = sorted({i for s in sessions for i in s["items"]})
    item_id: Dict[str, int] = {asin: idx for idx, asin in enumerate(surviving_items)}
    items_table = [
        {"item_idx": idx, "parent_asin": asin, "title": title_map[asin]}
        for asin, idx in item_id.items()
    ]
    with open(out / "items.json", "w") as f:
        import json
        json.dump(items_table, f)

    # ---- 6. chronological 80/10/10 split over sessions ---------------------
    sessions.sort(key=lambda s: s["session_ts"])
    n = len(sessions)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = sessions[:n_train]
    val = sessions[n_train : n_train + n_val]
    test = sessions[n_train + n_val :]

    def to_records(split):
        for s in split:
            yield {
                "user_id": s["user_id"],
                "day": s["day"],
                "items": [item_id[i] for i in s["items"]],
            }

    write_jsonl(to_records(train), out / "train.jsonl")
    write_jsonl(to_records(val), out / "val.jsonl")
    write_jsonl(to_records(test), out / "test.jsonl")

    stats = {
        "n_items": len(item_id),
        "n_sessions_total": n,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "avg_session_len": round(
            sum(len(s["items"]) for s in sessions) / max(n, 1), 3
        ),
    }
    with open(out / "stats.json", "w") as f:
        import json
        json.dump(stats, f, indent=2)
    logger.info("Stats: %s", stats)
    return stats


def load_items_table(processed_dir: str) -> Tuple[List[str], List[str]]:
    """Return (titles_by_index, parent_asins_by_index)."""
    import json
    with open(Path(processed_dir) / "items.json", "r") as f:
        items = json.load(f)
    items.sort(key=lambda r: r["item_idx"])
    titles = [r["title"] for r in items]
    asins = [r["parent_asin"] for r in items]
    return titles, asins
