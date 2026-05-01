from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from ..utils import ensure_dir, get_logger


REVIEW_COLS = ("user_id", "parent_asin", "asin", "timestamp", "rating")
META_COLS = ("parent_asin", "title", "main_category", "categories")

REVIEW_SCHEMA = pa.schema(
    [
        ("user_id", pa.string()),
        ("parent_asin", pa.string()),
        ("asin", pa.string()),
        ("timestamp", pa.int64()),
        ("rating", pa.float32()),
    ]
)
META_SCHEMA = pa.schema(
    [
        ("parent_asin", pa.string()),
        ("title", pa.string()),
        ("main_category", pa.string()),
        ("categories", pa.list_(pa.string())),
    ]
)


def _category_from_config(config_name: str) -> str:

    for prefix in ("raw_review_", "raw_meta_"):
        if config_name.startswith(prefix):
            return config_name[len(prefix) :]
    return config_name


def _stream_jsonl_to_parquet(
    src_path: Path,
    dst_path: Path,
    keep_cols: Tuple[str, ...],
    schema: pa.Schema,
    *,
    batch_size: int = 100_000,
    logger=None,
) -> int:

    rows_written = 0
    buffer: Dict[str, list] = {c: [] for c in keep_cols}

    def _flush(writer: pq.ParquetWriter) -> None:
        nonlocal rows_written
        if not buffer[keep_cols[0]]:
            return
        table = pa.table(buffer, schema=schema)
        writer.write_table(table)
        rows_written += len(buffer[keep_cols[0]])
        for c in keep_cols:
            buffer[c].clear()

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(str(dst_path), schema=schema, compression="snappy") as writer:
        with open(src_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for c in keep_cols:
                    buffer[c].append(obj.get(c))
                if len(buffer[keep_cols[0]]) >= batch_size:
                    _flush(writer)
                    if logger is not None and rows_written % (batch_size * 10) == 0:
                        logger.info("  ... %d rows -> %s", rows_written, dst_path.name)
        _flush(writer)
    return rows_written


def download_games(
    raw_dir: str,
    hf_dataset: str = "McAuley-Lab/Amazon-Reviews-2023",
    review_config: str = "raw_review_Video_Games",
    meta_config: str = "raw_meta_Video_Games",
) -> Tuple[Path, Path]:

    logger = get_logger("data.download")
    ensure_dir(raw_dir)

    cache_dir = str(Path(raw_dir).resolve())
    review_cat = _category_from_config(review_config)
    meta_cat = _category_from_config(meta_config)

    review_remote = f"raw/review_categories/{review_cat}.jsonl"
    meta_remote = f"raw/meta_categories/meta_{meta_cat}.jsonl"

    review_path = Path(raw_dir) / "reviews.parquet"
    meta_path = Path(raw_dir) / "meta.parquet"

    if not review_path.exists():
        logger.info("Downloading reviews JSONL %s ...", review_remote)
        review_jsonl = hf_hub_download(
            repo_id=hf_dataset,
            filename=review_remote,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        logger.info("Streaming -> %s ...", review_path)
        n = _stream_jsonl_to_parquet(
            Path(review_jsonl),
            review_path,
            REVIEW_COLS,
            REVIEW_SCHEMA,
            logger=logger,
        )
        logger.info("Wrote %d review rows to %s.", n, review_path)
    else:
        logger.info("Reusing cached %s.", review_path)

    if not meta_path.exists():
        logger.info("Downloading meta JSONL %s ...", meta_remote)
        meta_jsonl = hf_hub_download(
            repo_id=hf_dataset,
            filename=meta_remote,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        logger.info("Streaming -> %s ...", meta_path)
        n = _stream_jsonl_to_parquet(
            Path(meta_jsonl),
            meta_path,
            META_COLS,
            META_SCHEMA,
            logger=logger,
        )
        logger.info("Wrote %d meta rows to %s.", n, meta_path)
    else:
        logger.info("Reusing cached %s.", meta_path)

    return review_path, meta_path


def field_summary(raw_dir: str) -> Dict[str, list]:

    import pandas as pd

    rev = pd.read_parquet(Path(raw_dir) / "reviews.parquet")
    met = pd.read_parquet(Path(raw_dir) / "meta.parquet")
    return {"reviews": list(rev.columns), "meta": list(met.columns)}
