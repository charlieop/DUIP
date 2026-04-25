"""Train DUIP on Amazon Reviews 2023 Games (frozen Qwen3.5-2B)."""

from __future__ import annotations

import argparse

from src.train import run_training


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    args = p.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
