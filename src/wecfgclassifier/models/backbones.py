from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

from ..config import EXTERNAL_MODELS_DIR
from ..utils.runtime import TeeLogger, load_matching_state_dict


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RhythmWindowClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock1D(1, 16, stride=2),
            ConvBlock1D(16, 32, stride=2),
            ConvBlock1D(32, 64, stride=2),
            ConvBlock1D(64, 96, stride=2),
            ConvBlock1D(96, 128, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 96),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(96, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.head[:-1](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head[-1](self.encode(x))


class DualHeadRhythmWindowClassifier(nn.Module):
    def __init__(self, num_diagnosis_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock1D(1, 16, stride=2),
            ConvBlock1D(16, 32, stride=2),
            ConvBlock1D(32, 64, stride=2),
            ConvBlock1D(64, 96, stride=2),
            ConvBlock1D(96, 128, stride=2),
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.diagnosis_classifier = nn.Sequential(
            nn.Linear(128, 96),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(96, num_diagnosis_classes),
        )
        self.noise_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.unsqueeze(1)
        x = self.pool(self.features(x))
        return {
            "diagnosis_logits": self.diagnosis_classifier(x),
            "noise_logit": self.noise_classifier(x).squeeze(-1),
        }


class CNNBiLSTMAttentionClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.attention = nn.Linear(256, 1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)
        attn = torch.softmax(self.attention(x).squeeze(-1), dim=1)
        return torch.sum(x * attn.unsqueeze(-1), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))


class DualHeadCNNBiLSTMAttentionClassifier(nn.Module):
    def __init__(self, num_diagnosis_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.attention = nn.Linear(256, 1)
        self.diagnosis_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_diagnosis_classes),
        )
        self.noise_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)
        attn = torch.softmax(self.attention(x).squeeze(-1), dim=1)
        context = torch.sum(x * attn.unsqueeze(-1), dim=1)
        return {
            "diagnosis_logits": self.diagnosis_classifier(context),
            "noise_logit": self.noise_classifier(context).squeeze(-1),
        }


def build_classifier(
    strategy: str,
    num_classes: int,
    logger: TeeLogger,
    init_path: str | None = None,
    classifier_head: str = "single",
    noise_class_name: str = "Noise",
    class_order: list[str] | None = None,
) -> nn.Module:
    if classifier_head == "diagnosis_noise":
        if class_order is None or noise_class_name not in class_order:
            raise ValueError(f"--classifier-head diagnosis_noise requires noise class '{noise_class_name}' in class_order")
        num_diagnosis_classes = num_classes - 1
    elif classifier_head != "single":
        raise ValueError(f"Unsupported classifier_head: {classifier_head}")

    if strategy == "lightweight_pytorch_baseline":
        if classifier_head == "diagnosis_noise":
            model = DualHeadRhythmWindowClassifier(num_diagnosis_classes=num_diagnosis_classes)
        else:
            model = RhythmWindowClassifier(num_classes=num_classes)
    elif strategy == "pretrained_cnn_bilstm":
        if classifier_head == "diagnosis_noise":
            model = DualHeadCNNBiLSTMAttentionClassifier(num_diagnosis_classes=num_diagnosis_classes)
        else:
            model = CNNBiLSTMAttentionClassifier(num_classes=num_classes)
        ckpt_path = EXTERNAL_MODELS_DIR / "dheerajthuvara_ecg_arrhythmia_detection" / "models" / "best_model.pt"
        pretrained_sd = torch.load(ckpt_path, map_location="cpu")
        load_matching_state_dict(model, pretrained_sd, logger, f"[init] pretrained_cnn_bilstm from {ckpt_path.name}")
    else:
        raise ValueError(f"Unsupported classifier strategy: {strategy}")

    if init_path:
        state = torch.load(init_path, map_location="cpu")
        state_dict = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
        load_matching_state_dict(model, state_dict, logger, f"[init] fine-tune warm start from {Path(init_path).name}")
    return model
