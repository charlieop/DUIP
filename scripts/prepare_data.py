"""Download + sessionize the Amazon Reviews 2023 Games subset."""

from __future__ import annotations

import argparse

from src.data.download import download_games
from src.data.preprocess import build_sessions
from src.utils import ensure_dir, get_logger, load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse parquet files already present under raw_dir",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("prepare_data", cfg["paths"]["log_dir"])

    ensure_dir(cfg["paths"]["raw_dir"])
    ensure_dir(cfg["paths"]["processed_dir"])

    if not args.skip_download:
        logger.info("Downloading raw Amazon Reviews 2023 (Games) ...")
        download_games(
            raw_dir=cfg["paths"]["raw_dir"],
            hf_dataset=cfg["data"]["hf_dataset"],
            review_config=cfg["data"]["review_config"],
            meta_config=cfg["data"]["meta_config"],
        )

    logger.info("Sessionizing ...")
    stats = build_sessions(
        raw_dir=cfg["paths"]["raw_dir"],
        processed_dir=cfg["paths"]["processed_dir"],
        min_session_len=cfg["data"]["min_session_len"],
        max_session_len=cfg["data"]["max_session_len"],
        min_user_sessions=cfg["data"]["min_user_sessions"],
        min_item_freq=cfg["data"]["min_item_freq"],
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
    )
    logger.info("Done. %s", stats)


if __name__ == "__main__":
    main()
