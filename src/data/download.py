"""Download the Amazon Reviews 2023 (Video Games) raw review + meta splits.

We rely on the official Hugging Face mirror
``McAuley-Lab/Amazon-Reviews-2023`` published with the paper
"Bridging Language and Items for Retrieval and Recommendation" (McAuley et al.,
2024).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from datasets import load_dataset

from ..utils import ensure_dir, get_logger


def download_games(
    raw_dir: str,
    hf_dataset: str = "McAuley-Lab/Amazon-Reviews-2023",
    review_config: str = "raw_review_Video_Games",
    meta_config: str = "raw_meta_Video_Games",
) -> Tuple[Path, Path]:
    """Download (or load from cache) the review and metadata Parquet files.

    Returns the (review_dir, meta_dir) paths the dataset cache landed in.
    """
    logger = get_logger("data.download")
    ensure_dir(raw_dir)

    cache_dir = str(Path(raw_dir).resolve())

    logger.info("Loading reviews split %s ...", review_config)
    reviews = load_dataset(
        hf_dataset,
        review_config,
        split="full",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    logger.info("Loaded %d review rows.", len(reviews))

    logger.info("Loading meta split %s ...", meta_config)
    meta = load_dataset(
        hf_dataset,
        meta_config,
        split="full",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    logger.info("Loaded %d meta rows.", len(meta))

    # Persist a flattened, slimmed-down view so preprocess.py doesn't need
    # the full HF cache.
    review_path = Path(raw_dir) / "reviews.parquet"
    meta_path = Path(raw_dir) / "meta.parquet"

    keep_review = [c for c in ["user_id", "parent_asin", "asin", "timestamp", "rating"]
                   if c in reviews.column_names]
    keep_meta = [c for c in ["parent_asin", "title", "main_category", "categories"]
                 if c in meta.column_names]

    reviews.select_columns(keep_review).to_parquet(str(review_path))
    meta.select_columns(keep_meta).to_parquet(str(meta_path))

    logger.info("Wrote %s and %s.", review_path, meta_path)
    return review_path, meta_path


def field_summary(raw_dir: str) -> Dict[str, list]:
    """Convenience: list columns of the cached parquet files."""
    import pandas as pd

    rev = pd.read_parquet(Path(raw_dir) / "reviews.parquet")
    met = pd.read_parquet(Path(raw_dir) / "meta.parquet")
    return {"reviews": list(rev.columns), "meta": list(met.columns)}
