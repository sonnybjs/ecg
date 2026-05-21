from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DENOISER_FS, EXTERNAL_MODELS_DIR, TARGET_WINDOW_SAMPLES
from ..utils.runtime import TeeLogger


class GRUDenoiserWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        repo = EXTERNAL_MODELS_DIR / "ECG_Denoiser"
        sys.path.insert(0, str(repo))
        from gru_denoiser import GRU  # type: ignore

        self.model = GRU(n_features=1, hid_dim=64, n_layers=1, dropout=0, bidirectional=True, gpu_id=None)
        weights = repo / "best_gru_denoiser_360Hz"
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))

    @staticmethod
    def minmax_per_sample(x: torch.Tensor) -> torch.Tensor:
        x_min = x.amin(dim=1, keepdim=True)
        x_max = x.amax(dim=1, keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-6)

    def forward(self, raw_window: torch.Tensor) -> torch.Tensor:
        x = self.minmax_per_sample(raw_window)
        x = x.unsqueeze(1)
        x = F.interpolate(x, size=TARGET_WINDOW_SAMPLES, mode="linear", align_corners=False)
        x = x.transpose(1, 2)
        y = self.model(x).transpose(1, 2)
        y = F.interpolate(y, size=TARGET_WINDOW_SAMPLES, mode="linear", align_corners=False)
        y = y.squeeze(1)
        return self.minmax_per_sample(y)


def maybe_load_denoiser_weights(model: nn.Module, init_path: str | None, logger: TeeLogger) -> None:
    if not init_path:
        return
    state = torch.load(init_path, map_location="cpu")
    model.load_state_dict(state["model_state"] if isinstance(state, dict) and "model_state" in state else state)
    logger.log(f"[init] loaded denoiser weights from {init_path}")


def compute_classifier_input(
    raw_batch: torch.Tensor,
    classifier_input: str,
    denoiser: nn.Module | None,
) -> torch.Tensor:
    if classifier_input == "raw":
        return raw_batch
    if classifier_input == "frozen_denoiser":
        assert denoiser is not None
        return denoiser(raw_batch)
    raise ValueError(f"Unsupported classifier_input: {classifier_input}")

