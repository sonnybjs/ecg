from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import ECGDATA_DIR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def next_experiment_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in root.glob("exp_*") if p.is_dir())
    next_idx = 1
    if existing:
        next_idx = max(int(p.name.split("_")[-1]) for p in existing) + 1
    exp_dir = root / f"exp_{next_idx:03d}"
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir


class TeeLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        print(message, flush=True)
        with self.log_path.open("a") as handle:
            handle.write(message + "\n")


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def load_matching_state_dict(model: nn.Module, state_dict: dict, logger: TeeLogger, prefix: str) -> None:
    model_sd = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    for key, value in state_dict.items():
        if key in model_sd and tuple(model_sd[key].shape) == tuple(value.shape):
            model_sd[key] = value
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)
    model.load_state_dict(model_sd)
    logger.log(f"{prefix} loaded {len(loaded_keys)} tensors; skipped {len(skipped_keys)} mismatched tensors")


def available_days() -> list[str]:
    return sorted(p.name for p in ECGDATA_DIR.iterdir() if p.is_dir())


def resolve_days(requested_days: list[str]) -> list[str]:
    normalized = [day.strip() for day in requested_days if day.strip()]
    if not normalized or any(day.upper() == "ALL" for day in normalized):
        return available_days()
    return normalized

