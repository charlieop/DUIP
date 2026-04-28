"""Modal entrypoint for the DUIP pipeline (Qwen3.5-0.8B + LSTM, frozen LLM).

Runs the same `prepare_data`, `train`, and `eval` flows as the local
scripts, but on Modal's managed **H100** GPUs with persistent volumes for
the dataset, the Hugging Face cache, model checkpoints, results, and
per-run logs.

Quickstart (from the repo root):

    pip install -r requirements.txt           # installs `modal` + `python-dotenv`
    cp .env.example .env                      # then fill in the values
    modal run modal_app.py::prepare_data      # one-time: data -> volume
    modal run modal_app.py::train             # train on H100
    modal run modal_app.py::evaluate          # eval best.pt on H100
    modal run modal_app.py::run_all           # all three in sequence

Pass a non-default config with e.g. ``modal run modal_app.py::train \
--config configs/games.yaml``.

The first invocation will build the container image (CUDA 12.4 + Python
3.11 + Torch + Transformers + Flash-Attention-2). That image is cached
across subsequent runs, so only the first launch pays the build cost.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Load .env *before* importing modal so MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
# (and WANDB_API_KEY / HF_TOKEN, which we forward as a Modal Secret) are
# visible to the Modal client at authentication time.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - python-dotenv is a hard dep in requirements.txt
    pass

import modal


# --------------------------------------------------------------------------
# Container image. Notes:
#
# * We use `debian_slim` rather than the `nvidia/cuda:*-devel` base because
#   we don't compile any CUDA extensions in-image — PyTorch's wheel ships
#   its own CUDA runtime, which is everything Transformers / SDPA need on
#   an H100. Image builds in ~1-2 minutes instead of 10-15.
#
# * Torch is installed from the cu124 index. The default PyPI index now
#   ships a torch built against CUDA 13 which mismatches Modal's GPU
#   drivers and breaks downstream extensions; pinning the index avoids it.
#
# * Flash-Attention-2 is intentionally *not* installed. The DUIP code's
#   `_attn_fallback_chain` will fall back to PyTorch SDPA, and on H100
#   PyTorch's SDPA already dispatches to Flash-Attention-2 / -3 kernels
#   internally — so the runtime performance is effectively the same with
#   none of the version-matrix headaches. To force the standalone
#   `flash-attn` wheel back in, see the commented block below.
#
# * The local `src/`, `scripts/` and `configs/` directories are mounted at
#   runtime so iterating on code doesn't trigger a full image rebuild.
# --------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.4,<2.7",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.46",
        "datasets>=2.20",
        "accelerate>=0.34",
        "huggingface-hub>=0.24",
        "hf-transfer>=0.1.6",
        "safetensors>=0.4",
        "sentencepiece>=0.2",
        "protobuf>=4.25",
        "pyyaml>=6.0",
        "tqdm>=4.66",
        "numpy>=1.26",
        "pandas>=2.0",
        "pyarrow>=15.0",
        "wandb>=0.17",
    )
    # ---- Qwen3.5 fast-path dependencies ------------------------------------
    # Qwen3.5 is a hybrid model with **Gated DeltaNet** (linear-attention)
    # layers. Without these two packages, transformers falls back to a
    # pure-PyTorch `torch_chunk_gated_delta_rule` that materializes huge
    # intermediates and OOMs even an 80 GB H100 within a single forward.
    #
    #   * flash-linear-attention (FLA): pure Triton kernels — no nvcc
    #     needed, installs cleanly on `debian_slim`.
    #   * causal-conv1d: depthwise 1D conv at the front of each DeltaNet
    #     block. PyPI ships only an sdist whose setup.py needs nvcc just
    #     to detect the CUDA version (and the 1.6.1 sdist additionally
    #     has no GitHub-release wheels at all), so on a no-CUDA-toolkit
    #     base image like `debian_slim` we MUST install it from a
    #     precompiled wheel URL.
    #
    #     We pin v1.5.0.post8 (NOT the newer v1.6.x), specifically the
    #     `cu12torch2.6cxx11abiFALSE-cp311` variant, because:
    #       1. v1.6.0's `cxx11abiFALSE` wheel was empirically built with
    #          `_GLIBCXX_USE_CXX11_ABI=1` (NEW ABI -- references
    #          `std::__cxx11::basic_string` torchCheckFail symbols),
    #          which mismatches the OLD-ABI torch 2.6.0+cu124 from the
    #          official PyTorch CDN and yields an undefined-symbol
    #          ImportError at first `import causal_conv1d`. Confirmed
    #          via `nm` on the .so files inside the built image.
    #       2. v1.5.0.post8's `cxx11abiFALSE` wheel for the same
    #          (cu12, torch 2.6, cp311) combo has 33k+ downloads and
    #          its references genuinely match the old-ABI torch.
    #     If you ever bump torch past 2.6, re-pick the matching
    #     `+cu12torch{X.Y}cxx11abiFALSE-cp311` wheel from
    #     https://github.com/Dao-AILab/causal-conv1d/releases.
    #
    # We re-pin `torch>=2.4,<2.7` here (with the cu124 extra index) so
    # pip's resolver can't satisfy causal-conv1d's unbounded `torch`
    # dependency by upgrading to a default-index `+cu130` build that
    # would mismatch Modal's drivers and break the wheel ABI.
    .pip_install(
        "torch>=2.4,<2.7",
        "flash-linear-attention",
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/"
        "v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.6cxx11abiFALSE"
        "-cp311-cp311-linux_x86_64.whl",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # ---- Optional: standalone flash-attn -----------------------------------
    # If you really want the standalone `flash-attn` package (rather than
    # SDPA's built-in FA dispatch), pick a precompiled wheel that matches
    # the torch version above and uncomment. Building from source needs
    # the cuda:devel base image and ~10 min of compile time.
    #
    # .pip_install(
    #     "https://github.com/Dao-AILab/flash-attention/releases/download/"
    #     "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE"
    #     "-cp311-cp311-linux_x86_64.whl"
    # )
    .env({
        # Persist the HF cache on a volume (see VOLUMES below).
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
        # `hf-transfer` saturates the NIC for big downloads (Qwen weights).
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # Make `from src.X import ...` and `from scripts.Y import ...` work
        # regardless of cwd inside the container.
        "PYTHONPATH": "/root",
    })
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("scripts", remote_path="/root/scripts")
)


# --------------------------------------------------------------------------
# Persistent volumes — one per artifact kind so they can be wiped /
# inspected independently via `modal volume ...`.
# --------------------------------------------------------------------------

data_volume = modal.Volume.from_name("duip-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("duip-checkpoints", create_if_missing=True)
results_volume = modal.Volume.from_name("duip-results", create_if_missing=True)
logs_volume = modal.Volume.from_name("duip-logs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("duip-hf-cache", create_if_missing=True)

# Mount points line up with the relative paths in `configs/games.yaml`,
# so the same config works locally and on Modal as long as cwd is /root.
VOLUMES = {
    "/root/data": data_volume,
    "/root/checkpoints": checkpoints_volume,
    "/root/results": results_volume,
    "/root/logs": logs_volume,
    "/root/.cache/huggingface": hf_cache_volume,
}


# --------------------------------------------------------------------------
# Secrets — wandb + HF token, lifted out of the local `.env`. Modal tokens
# are intentionally NOT forwarded into the container; they're only used by
# the local CLI to authenticate against Modal itself.
# --------------------------------------------------------------------------

_secret_kv = {}
for _key in ("WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    _val = os.environ.get(_key)
    if _val:
        _secret_kv[_key] = _val
SECRETS = [modal.Secret.from_dict(_secret_kv)] if _secret_kv else []


# --------------------------------------------------------------------------
# App + shared function kwargs.
# --------------------------------------------------------------------------

app = modal.App("duip-qwen3p5-0p8b")

DEFAULT_CONFIG = "configs/games.yaml"
DEFAULT_CHECKPOINT = "checkpoints/games/best.pt"
_PROCESSED_REQUIRED_FILES = (
    "items.json",
    "train.jsonl",
    "val.jsonl",
    "test.jsonl",
    "stats.json",
)


def _cached_processed_stats(processed_dir: str) -> dict | None:
    """Return cached preprocessing stats when all split artifacts exist."""
    out = Path(processed_dir)
    for name in _PROCESSED_REQUIRED_FILES:
        path = out / name
        if not path.is_file() or path.stat().st_size == 0:
            return None

    try:
        with open(out / "stats.json", "r") as f:
            stats = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    return stats if isinstance(stats, dict) else None

# CPU-only step (data download + sessionization). Bumping memory because
# the raw Amazon Reviews 2023 (Games) JSONL is multi-GB.
_CPU_KWARGS = dict(
    image=image,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=60 * 60 * 4,  # 4 h
    cpu=4,
    memory=16 * 1024,  # 16 GiB
)

# GPU steps: a single H100. Override with e.g. gpu="H100:2" if you ever
# want multi-GPU.
_GPU_KWARGS = dict(
    image=image,
    gpu="H100",
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=60 * 60 * 8,  # 8 h
)


def _enter_workdir() -> None:
    """Cd into /root so the relative paths in the YAML configs resolve to
    the mounted volumes (e.g. ``checkpoints/games/best.pt`` ->
    ``/root/checkpoints/games/best.pt``), and make sure /root is on
    ``sys.path`` so ``import src...`` / ``import scripts...`` works."""
    import sys

    os.chdir("/root")
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")


# --------------------------------------------------------------------------
# Remote functions.
# --------------------------------------------------------------------------

@app.function(**_CPU_KWARGS)
def prepare_data_remote(config: str = DEFAULT_CONFIG, force: bool = False) -> dict:
    """Download + sessionize the Amazon Reviews 2023 (Games) subset onto
    the ``duip-data`` volume."""
    _enter_workdir()

    from src.data.download import download_games
    from src.data.preprocess import build_sessions
    from src.utils import ensure_dir, get_logger, load_config

    cfg = load_config(config)
    logger = get_logger("prepare_data", cfg["paths"]["log_dir"])

    ensure_dir(cfg["paths"]["raw_dir"])
    ensure_dir(cfg["paths"]["processed_dir"])

    if force:
        logger.info("Force rebuild requested; ignoring existing processed data.")
    else:
        cached_stats = _cached_processed_stats(cfg["paths"]["processed_dir"])
        if cached_stats is not None:
            logger.info(
                "Reusing processed data in %s. Pass force=True to rebuild.",
                cfg["paths"]["processed_dir"],
            )
            logs_volume.commit()
            return cached_stats

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

    data_volume.commit()
    logs_volume.commit()
    return stats


@app.function(**_GPU_KWARGS)
def train_remote(config: str = DEFAULT_CONFIG) -> dict:
    """Train DUIP on a single H100; persists checkpoints + logs + Qwen
    weights cache to their respective volumes."""
    _enter_workdir()

    from src.train import run_training

    out = run_training(config)

    checkpoints_volume.commit()
    logs_volume.commit()
    results_volume.commit()
    hf_cache_volume.commit()
    return out


@app.function(**_GPU_KWARGS)
def evaluate_remote(
    config: str = DEFAULT_CONFIG,
    checkpoint: str | None = DEFAULT_CHECKPOINT,
) -> dict:
    """Evaluate a trained DUIP checkpoint on the test split (H100)."""
    _enter_workdir()

    from src.evaluate import run_evaluation

    metrics = run_evaluation(config, checkpoint)

    results_volume.commit()
    logs_volume.commit()
    hf_cache_volume.commit()
    return metrics


# --------------------------------------------------------------------------
# Local entrypoints — invoked via `modal run modal_app.py::<name>`.
# --------------------------------------------------------------------------

@app.local_entrypoint()
def prepare_data(config: str = DEFAULT_CONFIG, force: bool = False) -> None:
    """Run the data download + sessionization on Modal."""
    out = prepare_data_remote.remote(config, force=force)
    print("[prepare_data] ->", out)


@app.local_entrypoint()
def train(config: str = DEFAULT_CONFIG) -> None:
    """Train DUIP on an H100. Writes the best checkpoint to the
    ``duip-checkpoints`` volume at ``checkpoints/games/best.pt``."""
    out = train_remote.remote(config)
    print("[train] ->", out)


@app.local_entrypoint()
def evaluate(
    config: str = DEFAULT_CONFIG,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> None:
    """Evaluate the best checkpoint on the test split. ``checkpoint`` is a
    path *relative to /root* (i.e. inside the volumes)."""
    metrics = evaluate_remote.remote(config, checkpoint)
    print("[evaluate] ->", metrics)


@app.local_entrypoint()
def run_all(config: str = DEFAULT_CONFIG, force_prepare: bool = False) -> None:
    """End-to-end: prepare_data -> train -> evaluate.

    The eval-stage checkpoint is derived from ``cfg.paths.checkpoint_dir``
    so this works for both ``configs/games.yaml`` (-> ``checkpoints/games/
    best.pt``) and ``configs/games_h100.yaml`` (-> ``checkpoints/
    games_h100/best.pt``) without manual override.
    """
    import yaml as _yaml
    from pathlib import PurePosixPath as _PPath

    with open(config, "r") as _f:
        _cfg = _yaml.safe_load(_f)
    ckpt = str(_PPath(_cfg["paths"]["checkpoint_dir"]) / "best.pt")

    print("[1/3] Preparing data ...")
    prepare_data_remote.remote(config, force=force_prepare)
    print("[2/3] Training ...")
    train_remote.remote(config)
    print("[3/3] Evaluating (%s) ..." % ckpt)
    metrics = evaluate_remote.remote(config, ckpt)
    print("[run_all] test metrics:", metrics)
