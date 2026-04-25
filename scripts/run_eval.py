"""Evaluate a trained DUIP checkpoint on the test split."""

from __future__ import annotations

import argparse

from src.evaluate import run_evaluation


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a .pt file produced by training (LSTM + projector "
             "weights). If omitted, evaluates the untrained model "
             "(useful as a sanity baseline).",
    )
    args = p.parse_args()
    run_evaluation(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
