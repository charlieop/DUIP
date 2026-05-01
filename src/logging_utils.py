from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .utils import ensure_dir, get_logger


try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


def _flatten(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:

    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten(v, key, sep))
        else:
            out[key] = v
    return out


def collect_hardware_info(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info["gpu_index"] = idx
        info["gpu_name"] = props.name
        info["gpu_total_mem_gb"] = round(props.total_memory / (1024**3), 2)
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        )
    if extra:
        info.update(extra)
    return info


@dataclass
class _Throughput:

    window: int = 50
    samples: deque = field(default_factory=deque)
    tokens: deque = field(default_factory=deque)
    times: deque = field(default_factory=deque)

    def update(self, n_samples: int, n_tokens: int, dt: float) -> Dict[str, float]:
        self.samples.append(n_samples)
        self.tokens.append(n_tokens)
        self.times.append(max(dt, 1e-9))
        while len(self.times) > self.window:
            self.samples.popleft()
            self.tokens.popleft()
            self.times.popleft()
        total_t = sum(self.times)
        return {
            "samples_per_sec": sum(self.samples) / total_t,
            "tokens_per_sec": sum(self.tokens) / total_t,
        }


class RunLogger:

    def __init__(
        self,
        cfg: Dict[str, Any],
        run_kind: str = "train",
        *,
        run_name: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.run_kind = run_kind

        log_cfg = cfg.get("logging") or {}
        wandb_cfg = log_cfg.get("wandb") or {}

        self.log_every_steps = int(log_cfg.get("log_every_steps", 10))
        self.jsonl_enabled = bool(log_cfg.get("jsonl", True))

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        explicit = run_name or wandb_cfg.get("run_name")
        self.run_name = explicit or f"{run_kind}-{ts}"

        log_dir = Path(cfg.get("paths", {}).get("log_dir", "logs"))
        self.run_dir = ensure_dir(log_dir / self.run_name)

        self.logger = get_logger(f"{run_kind}.{self.run_name}", str(self.run_dir))

        self._jsonl_fp = None
        if self.jsonl_enabled:
            self._jsonl_fp = open(self.run_dir / "metrics.jsonl", "w", encoding="utf-8")

        self.wandb_enabled = (
            bool(wandb_cfg.get("enabled", True))
            and wandb_cfg.get("mode", "online") != "disabled"
            and _WANDB_AVAILABLE
        )
        self.wandb_run = None
        if bool(wandb_cfg.get("enabled", True)) and not _WANDB_AVAILABLE:
            self.logger.warning(
                "logging.wandb.enabled=true but `wandb` is not installed; "
                "skipping wandb. `pip install wandb` to enable."
            )
        if self.wandb_enabled:
            try:
                init_kwargs = dict(
                    project=wandb_cfg.get("project", "duip"),
                    entity=wandb_cfg.get("entity"),
                    name=self.run_name,
                    mode=wandb_cfg.get("mode", "online"),
                    job_type=run_kind,
                    config=_flatten(cfg),
                    dir=str(self.run_dir),
                )

                try:
                    self.wandb_run = wandb.init(
                        reinit="finish_previous",
                        **init_kwargs,
                    )
                except (TypeError, ValueError):
                    self.wandb_run = wandb.init(reinit=True, **init_kwargs)
            except Exception as e:
                self.logger.warning(
                    "wandb.init failed (%s); continuing without wandb.", e
                )
                self.wandb_enabled = False
                self.wandb_run = None

        self._throughput = _Throughput()
        self._loss_ema: Optional[float] = None
        self._ema_alpha = 0.1
        self._step = 0
        self._last_log_step = -1
        self._t_start = time.time()

    def log_hardware(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info = collect_hardware_info(extra=extra)
        for k, v in info.items():
            self.logger.info("env | %s=%s", k, v)
        if self.wandb_enabled and self.wandb_run is not None:
            with suppress(Exception):
                self.wandb_run.config.update({"env": info}, allow_val_change=True)
        self._write_jsonl({"event": "env", **info})
        return info

    def log_text(self, msg: str, *args: Any) -> None:
        self.logger.info(msg, *args)

    def log_warning(self, msg: str, *args: Any) -> None:
        self.logger.warning(msg, *args)

    def update_throughput(
        self, n_samples: int, n_tokens: int, dt: float
    ) -> Dict[str, float]:
        return self._throughput.update(n_samples, n_tokens, dt)

    def update_loss_ema(self, loss: float) -> float:
        if self._loss_ema is None:
            self._loss_ema = float(loss)
        else:
            self._loss_ema = (
                1.0 - self._ema_alpha
            ) * self._loss_ema + self._ema_alpha * float(loss)
        return self._loss_ema

    def should_log(self, step: int) -> bool:
        if step <= 0:
            return False
        if step - self._last_log_step >= self.log_every_steps:
            return True
        return False

    def log_step(
        self,
        metrics: Dict[str, Any],
        step: int,
        *,
        prefix: str = "train",
    ) -> None:

        self._step = step
        self._last_log_step = step
        flat: Dict[str, Any] = {f"{prefix}/{k}": v for k, v in metrics.items()}
        flat[f"{prefix}/step"] = step

        compact = " ".join(
            f"{k.split('/', 1)[-1]}={_fmt(v)}"
            for k, v in flat.items()
            if not k.endswith("/step")
        )
        self.logger.info("step=%d | %s", step, compact)

        if self.wandb_enabled and self.wandb_run is not None:
            with suppress(Exception):
                self.wandb_run.log(flat, step=step)

        self._write_jsonl({"event": "step", "step": step, "prefix": prefix, **metrics})

    def log_epoch(
        self, metrics: Dict[str, Any], epoch: int, *, prefix: str = "train"
    ) -> None:
        flat = {f"{prefix}/epoch_{k}": v for k, v in metrics.items()}
        flat[f"{prefix}/epoch"] = epoch
        self.logger.info(
            "epoch=%d | %s",
            epoch,
            " ".join(
                f"{k.split('/', 1)[-1]}={_fmt(v)}"
                for k, v in flat.items()
                if not k.endswith("/epoch")
            ),
        )
        if self.wandb_enabled and self.wandb_run is not None:
            with suppress(Exception):
                self.wandb_run.log(flat, step=self._step)
        self._write_jsonl(
            {"event": "epoch", "epoch": epoch, "prefix": prefix, **metrics}
        )

    def log_eval(
        self,
        metrics: Dict[str, Any],
        *,
        prefix: str = "val",
        step: Optional[int] = None,
    ) -> None:
        flat = {f"{prefix}/{k}": v for k, v in metrics.items()}
        s = step if step is not None else self._step
        self.logger.info(
            "[%s] %s", prefix, " ".join(f"{k}={_fmt(v)}" for k, v in metrics.items())
        )
        if self.wandb_enabled and self.wandb_run is not None:
            with suppress(Exception):
                self.wandb_run.log(flat, step=s)
        self._write_jsonl({"event": "eval", "prefix": prefix, "step": s, **metrics})

    def log_summary(self, summary: Dict[str, Any]) -> None:
        for k, v in summary.items():
            self.logger.info("summary | %s=%s", k, v)
            if self.wandb_enabled and self.wandb_run is not None:
                with suppress(Exception):
                    self.wandb_run.summary[k] = v
        self._write_jsonl({"event": "summary", **summary})

    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        if self._jsonl_fp is None:
            return
        record = {"ts": time.time(), **record}
        try:
            self._jsonl_fp.write(json.dumps(record, default=_json_default) + "\n")
            self._jsonl_fp.flush()
        except Exception as e:
            self.logger.warning("Failed to write JSONL row: %s", e)

    def finish(self) -> None:
        elapsed = time.time() - self._t_start
        self.log_summary({"elapsed_sec": round(elapsed, 2)})
        if self._jsonl_fp is not None:
            try:
                self._jsonl_fp.close()
            except Exception:
                pass
            self._jsonl_fp = None
        if self.wandb_enabled and self.wandb_run is not None:
            with suppress(Exception):
                self.wandb_run.finish()
            self.wandb_run = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish()


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if abs(v) >= 1e4 or (0 < abs(v) < 1e-3):
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


def _json_default(o: Any) -> Any:
    if isinstance(o, (torch.Tensor,)):
        return o.tolist()
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    return str(o)


def gpu_mem_stats(reset_peak: bool = False) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    return {
        "gpu_mem_alloc_gb": round(alloc, 3),
        "gpu_mem_reserved_gb": round(reserved, 3),
        "gpu_mem_peak_gb": round(peak, 3),
    }
