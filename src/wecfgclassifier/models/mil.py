from __future__ import annotations

import math

import torch
import torch.nn as nn


class TopKMILClassifier(nn.Module):
    def __init__(
        self,
        window_classifier: nn.Module,
        topk_fraction: float = 0.25,
        min_topk: int = 1,
    ) -> None:
        super().__init__()
        self.window_classifier = window_classifier
        self.topk_fraction = topk_fraction
        self.min_topk = min_topk

    def pool_instance_logits(self, instance_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_windows = instance_logits.shape[0]
        k = max(self.min_topk, math.ceil(n_windows * self.topk_fraction))
        k = min(k, n_windows)
        values, indices = torch.topk(instance_logits, k=k, dim=0)
        return values.mean(dim=0), indices

    def forward_bag(self, windows: torch.Tensor) -> dict[str, torch.Tensor]:
        instance_logits = self.window_classifier(windows)
        bag_logits, topk_indices = self.pool_instance_logits(instance_logits)
        return {
            "bag_logits": bag_logits,
            "instance_logits": instance_logits,
            "topk_indices": topk_indices,
        }

    def forward(self, bags: list[torch.Tensor]) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        bag_outputs = [self.forward_bag(windows) for windows in bags]
        return {
            "bag_logits": torch.stack([item["bag_logits"] for item in bag_outputs]),
            "instance_logits": [item["instance_logits"] for item in bag_outputs],
            "topk_indices": [item["topk_indices"] for item in bag_outputs],
        }
