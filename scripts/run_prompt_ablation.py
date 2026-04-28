"""Evaluate prompt ablations for a trained DUIP checkpoint."""

from __future__ import annotations

import argparse

from src.evaluate import DEFAULT_PROMPT_ABLATION_MODES, run_prompt_ablation


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a .pt file produced by training. If omitted, evaluates "
             "the untrained model in each prompt mode.",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=list(DEFAULT_PROMPT_ABLATION_MODES),
        help="Prompt modes to evaluate. Defaults to all ablation modes.",
    )
    p.add_argument(
        "--results-path",
        default=None,
        help="Optional output JSON path. Defaults to a sibling of "
             "paths.results_path with '_prompt_ablation' appended.",
    )
    args = p.parse_args()
    run_prompt_ablation(
        args.config,
        args.checkpoint,
        modes=args.modes,
        results_path=args.results_path,
    )


if __name__ == "__main__":
    main()
