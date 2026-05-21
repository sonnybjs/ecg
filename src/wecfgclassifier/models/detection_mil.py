from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_GROUPS: dict[str, list[str]] = {
    "rhythm_state": ["AF", "SRhy", "SArrh", "PaRhy", "JuRhy", "Idiov"],
    "atrial_supraventricular_event": ["SVT", "AtRun", "AtCou", "EctAt", "EctAR"],
    "ventricular_event": ["VT", "VeCou", "EctVe"],
    "conduction_pause": ["21AvB", "Wenck", "ComHB", "Pause"],
    "quality_noise": ["Noise"],
}

EVENT_CLASSES = {"AF", "SVT", "VT", "Pause", "AtRun", "AtCou", "VeCou", "EctAt", "EctVe", "EctAR"}
RHYTHM_CLASSES = {"SRhy", "SArrh", "PaRhy", "JuRhy", "Idiov"}
CONDUCTION_CLASSES = {"21AvB", "Wenck", "ComHB"}

DEFAULT_PRIORITY_ORDER = [
    "VT",
    "ComHB",
    "Pause",
    "AF",
    "SVT",
    "AtRun",
    "21AvB",
    "Wenck",
    "VeCou",
    "AtCou",
    "EctVe",
    "EctAt",
    "EctAR",
    "Idiov",
    "JuRhy",
    "PaRhy",
    "SArrh",
    "SRhy",
]


@dataclass
class DetectionMILConfig:
    evidence_head_type: str = "linear"
    evidence_head_hidden_dim: int = 64
    evidence_head_dropout: float = 0.3
    event_pooling: str = "logsumexp"
    event_lse_tau: float = 1.0
    rhythm_pooling: str = "topk_mean"
    rhythm_topk_fraction: float = 0.25
    rhythm_min_topk: int = 1
    conduction_pooling: str = "topk_attention_mix"
    conduction_topk_fraction: float = 0.25
    conduction_min_topk: int = 1
    attention_mix_alpha: float = 0.5
    default_pooling: str = "topk_mean"
    default_topk_fraction: float = 0.25
    default_min_topk: int = 1
    noise_pooling: str = "topk_mean"
    noise_topk_fraction: float = 0.25
    noise_min_topk: int = 1
    ce_weight: float = 1.0
    pos_bce_weight: float = 0.3
    group_ce_weight: float = 0.2
    noise_weight: float = 0.2
    sparsity_weight: float = 0.0
    consistency_weight: float = 0.0
    non_noise_as_weak_negative: bool = False
    noise_weak_negative_weight: float = 0.1
    global_diagnosis_threshold: float = 0.3
    noise_threshold: float = 0.5
    class_thresholds: dict[str, float] = field(default_factory=dict)
    priority_order: list[str] = field(default_factory=lambda: list(DEFAULT_PRIORITY_ORDER))
    top_m_evidence_windows: int = 3


@dataclass
class DetectionMILMappings:
    class_order: list[str]
    noise_class: str = "Noise"

    def __post_init__(self) -> None:
        if self.noise_class not in self.class_order:
            raise ValueError(f"noise_class {self.noise_class!r} not found in class_order")
        self.class_to_index_19 = {name: idx for idx, name in enumerate(self.class_order)}
        self.noise_index_19 = self.class_to_index_19[self.noise_class]
        self.diagnosis_classes = [name for name in self.class_order if name != self.noise_class]
        self.diagnosis_to_index_18 = {name: idx for idx, name in enumerate(self.diagnosis_classes)}
        self.index18_to_index19 = {idx18: self.class_to_index_19[name] for idx18, name in enumerate(self.diagnosis_classes)}
        self.index19_to_index18 = {idx19: idx18 for idx18, idx19 in self.index18_to_index19.items()}
        self.group_names = list(DEFAULT_GROUPS)
        self.group_to_index = {name: idx for idx, name in enumerate(self.group_names)}
        self.class19_to_group_id = {}
        for group_name, classes in DEFAULT_GROUPS.items():
            group_id = self.group_to_index[group_name]
            for class_name in classes:
                if class_name in self.class_to_index_19:
                    self.class19_to_group_id[self.class_to_index_19[class_name]] = group_id
        self.diagnosis_class_to_group_id = {
            class_name: self.class19_to_group_id[self.class_to_index_19[class_name]]
            for class_name in self.diagnosis_classes
        }
        missing = [idx for idx in range(len(self.class_order)) if idx not in self.class19_to_group_id]
        if missing:
            raise ValueError(f"Missing group mapping for class indices: {missing}")


def pad_bag_list(bags: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not bags:
        raise ValueError("Empty bag list")
    max_windows = max(int(bag.shape[0]) for bag in bags)
    if max_windows <= 0:
        raise ValueError("At least one bag has no valid windows")
    length = int(bags[0].shape[-1])
    device = bags[0].device
    dtype = bags[0].dtype
    padded = torch.zeros((len(bags), max_windows, length), dtype=dtype, device=device)
    mask = torch.zeros((len(bags), max_windows), dtype=torch.bool, device=device)
    for idx, bag in enumerate(bags):
        n_windows = int(bag.shape[0])
        if n_windows <= 0:
            raise ValueError("A bag has no valid windows")
        padded[idx, :n_windows] = bag
        mask[idx, :n_windows] = True
    return padded, mask


def masked_fill_invalid(x: torch.Tensor, mask: torch.Tensor, value: float = -1.0e9) -> torch.Tensor:
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    return x.masked_fill(~mask, value)


def masked_softmax_over_windows(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = masked_fill_invalid(logits, mask, -1.0e9)
    weights = torch.softmax(masked, dim=1)
    return weights * mask.unsqueeze(-1).to(weights.dtype)


def topk_count(n_windows: int, fraction: float, min_topk: int) -> int:
    return max(min_topk, math.ceil(n_windows * fraction))


def masked_topk_mean(x: torch.Tensor, mask: torch.Tensor, fraction: float = 0.25, min_topk: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    x_masked = x.masked_fill(~mask, -1.0e9)
    valid_counts = mask.sum(dim=1).clamp_min(1)
    max_k = min(int(valid_counts.max().item()), x.shape[1])
    k = max(min_topk, math.ceil(x.shape[1] * fraction))
    k = max(1, min(k, max_k))
    values, indices = torch.topk(x_masked, k=k, dim=1)
    keep = torch.arange(k, device=x.device).unsqueeze(0) < valid_counts.unsqueeze(1)
    values = values.masked_fill(~keep, 0.0)
    denom = keep.sum(dim=1).clamp_min(1).to(x.dtype)
    return values.sum(dim=1) / denom, indices


def masked_max_pool(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x.masked_fill(~mask, -1.0e9).max(dim=1)


def masked_logsumexp_pool(x: torch.Tensor, mask: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    tau = max(float(tau), 1.0e-6)
    x_masked = x.masked_fill(~mask, -1.0e9)
    valid_counts = mask.sum(dim=1).clamp_min(1).to(x.dtype)
    return tau * (torch.logsumexp(x_masked / tau, dim=1) - torch.log(valid_counts))


def masked_attention_pool(x: torch.Tensor, attention_weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = attention_weights * mask.to(attention_weights.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0e-6)
    return (weights * x).sum(dim=1) / denom


def noisy_or_pool_from_logits(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(x).masked_fill(~mask, 0.0).clamp(1.0e-6, 1.0 - 1.0e-6)
    log_not = torch.log1p(-probs).masked_fill(~mask, 0.0)
    bag_prob = (1.0 - torch.exp(log_not.sum(dim=1))).clamp(1.0e-6, 1.0 - 1.0e-6)
    return torch.logit(bag_prob)


def pooling_by_name(
    x: torch.Tensor,
    mask: torch.Tensor,
    pooling: str,
    fraction: float,
    min_topk: int,
    lse_tau: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if pooling == "topk_mean":
        return masked_topk_mean(x, mask, fraction=fraction, min_topk=min_topk)
    if pooling == "max":
        values, indices = masked_max_pool(x, mask)
        return values, indices.unsqueeze(1)
    if pooling == "logsumexp":
        return masked_logsumexp_pool(x, mask, tau=lse_tau), None
    if pooling == "noisy_or":
        return noisy_or_pool_from_logits(x, mask), None
    raise ValueError(f"Unsupported pooling: {pooling}")


class PriorityAwareDetectionMIL(nn.Module):
    def __init__(
        self,
        window_encoder: nn.Module,
        class_order: list[str],
        hidden_dim: int = 256,
        config: DetectionMILConfig | None = None,
    ) -> None:
        super().__init__()
        self.window_encoder = window_encoder
        self.hidden_dim = hidden_dim
        self.config = config or DetectionMILConfig()
        self.mappings = DetectionMILMappings(class_order)
        self.diagnosis_evidence_head = self._build_evidence_head(hidden_dim, len(self.mappings.diagnosis_classes))
        self.group_attention_head = nn.Linear(hidden_dim, len(self.mappings.group_names))
        self.noise_head = self._build_evidence_head(hidden_dim, 1)

    def _build_evidence_head(self, in_dim: int, out_dim: int) -> nn.Module:
        if self.config.evidence_head_type == "linear":
            return nn.Linear(in_dim, out_dim)
        if self.config.evidence_head_type == "classifier_mlp":
            return nn.Sequential(
                nn.Linear(in_dim, self.config.evidence_head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.evidence_head_dropout),
                nn.Linear(self.config.evidence_head_hidden_dim, out_dim),
            )
        raise ValueError(f"Unsupported evidence_head_type: {self.config.evidence_head_type}")

    def encode_windows(self, windows: torch.Tensor) -> torch.Tensor:
        if hasattr(self.window_encoder, "encode"):
            return self.window_encoder.encode(windows)
        return self.window_encoder(windows)

    def pool_diagnosis_class(
        self,
        evidence_logits: torch.Tensor,
        group_attention_weights: torch.Tensor,
        mask: torch.Tensor,
        class_name: str,
        class_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cfg = self.config
        z = evidence_logits[:, :, class_idx]
        if class_name in EVENT_CLASSES:
            return pooling_by_name(z, mask, cfg.event_pooling, cfg.default_topk_fraction, cfg.default_min_topk, cfg.event_lse_tau)
        if class_name in RHYTHM_CLASSES:
            return pooling_by_name(z, mask, cfg.rhythm_pooling, cfg.rhythm_topk_fraction, cfg.rhythm_min_topk, cfg.event_lse_tau)
        if class_name in CONDUCTION_CLASSES or class_name == "Pause":
            if cfg.conduction_pooling == "topk_attention_mix":
                topk_score, topk_indices = masked_topk_mean(z, mask, cfg.conduction_topk_fraction, cfg.conduction_min_topk)
                group_id = self.mappings.diagnosis_class_to_group_id[class_name]
                attn_score = masked_attention_pool(z, group_attention_weights[:, :, group_id], mask)
                score = cfg.attention_mix_alpha * topk_score + (1.0 - cfg.attention_mix_alpha) * attn_score
                return score, topk_indices
            return pooling_by_name(z, mask, cfg.conduction_pooling, cfg.conduction_topk_fraction, cfg.conduction_min_topk, cfg.event_lse_tau)
        return pooling_by_name(z, mask, cfg.default_pooling, cfg.default_topk_fraction, cfg.default_min_topk, cfg.event_lse_tau)

    def pool_diagnosis(
        self,
        evidence_logits: torch.Tensor,
        group_attention_weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        scores = []
        top_indices: dict[str, torch.Tensor] = {}
        for class_idx, class_name in enumerate(self.mappings.diagnosis_classes):
            score, indices = self.pool_diagnosis_class(evidence_logits, group_attention_weights, mask, class_name, class_idx)
            scores.append(score)
            if indices is not None:
                top_indices[class_name] = indices
        return torch.stack(scores, dim=1), top_indices

    def pool_noise(self, noise_window_logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        score, indices = pooling_by_name(
            noise_window_logits,
            mask,
            self.config.noise_pooling,
            self.config.noise_topk_fraction,
            self.config.noise_min_topk,
            self.config.event_lse_tau,
        )
        return score.unsqueeze(-1), indices

    def build_bag_logits_19(self, diagnosis_bag_logits: torch.Tensor, noise_bag_logit: torch.Tensor) -> torch.Tensor:
        batch_size = diagnosis_bag_logits.shape[0]
        logits = diagnosis_bag_logits.new_zeros((batch_size, len(self.mappings.class_order)))
        for idx18, idx19 in self.mappings.index18_to_index19.items():
            logits[:, idx19] = diagnosis_bag_logits[:, idx18]
        logits[:, self.mappings.noise_index_19] = noise_bag_logit.squeeze(-1)
        return logits

    def forward(
        self,
        bags: list[torch.Tensor] | torch.Tensor,
        mask: torch.Tensor | None = None,
        return_evidence: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if isinstance(bags, list):
            windows, mask = pad_bag_list(bags)
        else:
            windows = bags
            if mask is None:
                mask = torch.ones(windows.shape[:2], dtype=torch.bool, device=windows.device)
        if mask is None:
            raise ValueError("mask is required when bags is not a list")
        batch_size, n_windows, length = windows.shape
        flat = windows.reshape(batch_size * n_windows, length)
        encoded = self.encode_windows(flat).reshape(batch_size, n_windows, self.hidden_dim)
        evidence_logits = self.diagnosis_evidence_head(encoded)
        group_attention_logits = self.group_attention_head(encoded)
        group_attention_weights = masked_softmax_over_windows(group_attention_logits, mask)
        noise_window_logits = self.noise_head(encoded).squeeze(-1)
        diagnosis_bag_logits, top_indices = self.pool_diagnosis(evidence_logits, group_attention_weights, mask)
        noise_bag_logit, noise_top_indices = self.pool_noise(noise_window_logits, mask)
        bag_logits_19 = self.build_bag_logits_19(diagnosis_bag_logits, noise_bag_logit)
        output: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "bag_logits": bag_logits_19,
            "bag_logits_19": bag_logits_19,
            "diagnosis_bag_logits": diagnosis_bag_logits,
            "noise_bag_logit": noise_bag_logit.squeeze(-1),
            "primary_softmax_probs": torch.softmax(bag_logits_19, dim=-1),
            "diagnosis_probs": torch.sigmoid(diagnosis_bag_logits),
            "noise_prob": torch.sigmoid(noise_bag_logit.squeeze(-1)),
            "mask": mask,
        }
        if return_evidence:
            output.update(
                {
                    "evidence_logits": evidence_logits,
                    "evidence_probs": torch.sigmoid(evidence_logits),
                    "group_attention_logits": group_attention_logits,
                    "group_attention_weights": group_attention_weights,
                    "noise_window_logits": noise_window_logits,
                    "noise_window_probs": torch.sigmoid(noise_window_logits),
                    "top_window_indices": top_indices,
                }
            )
            if noise_top_indices is not None:
                output["noise_top_window_indices"] = noise_top_indices
        return output


def initialize_detection_heads_from_classifier_state(
    model: PriorityAwareDetectionMIL,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, int]:
    if not isinstance(model.diagnosis_evidence_head, nn.Sequential) or not isinstance(model.noise_head, nn.Sequential):
        return {"copied_tensors": 0, "copied_class_rows": 0, "skipped": 1}
    if len(model.diagnosis_evidence_head) < 4 or len(model.noise_head) < 4:
        return {"copied_tensors": 0, "copied_class_rows": 0, "skipped": 1}
    diag_first = model.diagnosis_evidence_head[0]
    diag_last = model.diagnosis_evidence_head[3]
    noise_first = model.noise_head[0]
    noise_last = model.noise_head[3]
    if not all(isinstance(layer, nn.Linear) for layer in [diag_first, diag_last, noise_first, noise_last]):
        return {"copied_tensors": 0, "copied_class_rows": 0, "skipped": 1}

    first_weight = state_dict.get("window_classifier.classifier.0.weight", state_dict.get("classifier.0.weight"))
    first_bias = state_dict.get("window_classifier.classifier.0.bias", state_dict.get("classifier.0.bias"))
    last_weight = state_dict.get("window_classifier.classifier.3.weight", state_dict.get("classifier.3.weight"))
    last_bias = state_dict.get("window_classifier.classifier.3.bias", state_dict.get("classifier.3.bias"))
    copied_tensors = 0
    copied_rows = 0

    with torch.no_grad():
        if first_weight is not None and tuple(first_weight.shape) == tuple(diag_first.weight.shape):
            diag_first.weight.copy_(first_weight)
            noise_first.weight.copy_(first_weight)
            copied_tensors += 2
        if first_bias is not None and tuple(first_bias.shape) == tuple(diag_first.bias.shape):
            diag_first.bias.copy_(first_bias)
            noise_first.bias.copy_(first_bias)
            copied_tensors += 2
        if last_weight is not None and last_bias is not None:
            for class_name in model.mappings.diagnosis_classes:
                src_idx = model.mappings.class_to_index_19[class_name]
                dst_idx = model.mappings.diagnosis_to_index_18[class_name]
                if src_idx < last_weight.shape[0]:
                    diag_last.weight[dst_idx].copy_(last_weight[src_idx])
                    diag_last.bias[dst_idx].copy_(last_bias[src_idx])
                    copied_rows += 1
            noise_idx = model.mappings.noise_index_19
            if noise_idx < last_weight.shape[0] and last_weight.shape[1:] == noise_last.weight.shape[1:]:
                noise_last.weight[0].copy_(last_weight[noise_idx])
                noise_last.bias[0].copy_(last_bias[noise_idx])
                copied_rows += 1
    return {"copied_tensors": copied_tensors, "copied_class_rows": copied_rows, "skipped": 0}


def build_group_logits_from_outputs(outputs: dict, mappings: DetectionMILMappings) -> torch.Tensor:
    diagnosis_logits = outputs["diagnosis_bag_logits"]
    noise_logit = outputs["noise_bag_logit"]
    group_logits = diagnosis_logits.new_full((diagnosis_logits.shape[0], len(mappings.group_names)), -1.0e9)
    for group_id in range(len(mappings.group_names)):
        diag_indices = [
            mappings.diagnosis_to_index_18[name]
            for name in mappings.diagnosis_classes
            if mappings.diagnosis_class_to_group_id[name] == group_id
        ]
        pieces = []
        if diag_indices:
            pieces.append(torch.logsumexp(diagnosis_logits[:, diag_indices], dim=1))
        if group_id == mappings.group_to_index["quality_noise"]:
            pieces.append(noise_logit)
        if pieces:
            group_logits[:, group_id] = torch.stack(pieces, dim=1).max(dim=1).values
    return group_logits


def labels19_to_group(labels_19: torch.Tensor, mappings: DetectionMILMappings) -> torch.Tensor:
    group_ids = [mappings.class19_to_group_id[int(label.item())] for label in labels_19]
    return torch.tensor(group_ids, dtype=torch.long, device=labels_19.device)


def detection_mil_loss(
    outputs: dict,
    labels_19: torch.Tensor,
    mappings: DetectionMILMappings,
    config: DetectionMILConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    bag_logits_19 = outputs["bag_logits_19"]
    diagnosis_bag_logits = outputs["diagnosis_bag_logits"]
    noise_bag_logit = outputs["noise_bag_logit"]
    loss = bag_logits_19.new_tensor(0.0)
    loss_dict: dict[str, float] = {}

    ce = F.cross_entropy(bag_logits_19, labels_19)
    loss = loss + config.ce_weight * ce
    loss_dict["ce"] = float(ce.detach().cpu())

    pos_losses = []
    for batch_idx, y19 in enumerate(labels_19):
        label_idx = int(y19.item())
        if label_idx == mappings.noise_index_19:
            continue
        diag_idx = mappings.index19_to_index18[label_idx]
        pos_losses.append(F.binary_cross_entropy_with_logits(diagnosis_bag_logits[batch_idx, diag_idx], torch.ones((), device=labels_19.device)))
    if pos_losses:
        pos_bce = torch.stack(pos_losses).mean()
        loss = loss + config.pos_bce_weight * pos_bce
        loss_dict["pos_bce"] = float(pos_bce.detach().cpu())
    else:
        loss_dict["pos_bce"] = 0.0

    noise_losses = []
    for batch_idx, y19 in enumerate(labels_19):
        if int(y19.item()) == mappings.noise_index_19:
            noise_losses.append(F.binary_cross_entropy_with_logits(noise_bag_logit[batch_idx], torch.ones((), device=labels_19.device)))
        elif config.non_noise_as_weak_negative:
            weak_loss = F.binary_cross_entropy_with_logits(noise_bag_logit[batch_idx], torch.zeros((), device=labels_19.device))
            noise_losses.append(config.noise_weak_negative_weight * weak_loss)
    if noise_losses:
        noise_loss = torch.stack(noise_losses).mean()
        loss = loss + config.noise_weight * noise_loss
        loss_dict["noise"] = float(noise_loss.detach().cpu())
    else:
        loss_dict["noise"] = 0.0

    if config.group_ce_weight > 0:
        group_logits = build_group_logits_from_outputs(outputs, mappings)
        group_labels = labels19_to_group(labels_19, mappings)
        group_ce = F.cross_entropy(group_logits, group_labels)
        loss = loss + config.group_ce_weight * group_ce
        loss_dict["group_ce"] = float(group_ce.detach().cpu())
    else:
        loss_dict["group_ce"] = 0.0

    if config.sparsity_weight > 0 and "evidence_logits" in outputs:
        event_indices = [mappings.diagnosis_to_index_18[name] for name in mappings.diagnosis_classes if name in EVENT_CLASSES]
        evidence_logits = outputs["evidence_logits"][:, :, event_indices]
        mask = outputs["mask"].unsqueeze(-1).to(evidence_logits.dtype)
        sparsity = (torch.sigmoid(evidence_logits) * mask).sum() / mask.sum().clamp_min(1.0)
        loss = loss + config.sparsity_weight * sparsity
        loss_dict["sparsity"] = float(sparsity.detach().cpu())
    else:
        loss_dict["sparsity"] = 0.0

    loss_dict["total"] = float(loss.detach().cpu())
    return loss, loss_dict


def threshold_for_class(class_name: str, config: DetectionMILConfig) -> float:
    if class_name in config.class_thresholds:
        return config.class_thresholds[class_name]
    return config.global_diagnosis_threshold


def detection_inference(outputs: dict, mappings: DetectionMILMappings, config: DetectionMILConfig) -> list[dict]:
    diagnosis_probs = outputs["diagnosis_probs"].detach().cpu()
    noise_probs = outputs["noise_prob"].detach().cpu()
    softmax_probs = outputs["primary_softmax_probs"].detach().cpu()
    rows = []
    priority_rank = {name: idx for idx, name in enumerate(config.priority_order)}
    for batch_idx in range(diagnosis_probs.shape[0]):
        candidates = []
        for diag_idx, class_name in enumerate(mappings.diagnosis_classes):
            prob = float(diagnosis_probs[batch_idx, diag_idx])
            if prob >= threshold_for_class(class_name, config):
                candidates.append(class_name)
        noise_prob = float(noise_probs[batch_idx])
        noise_flag = noise_prob >= config.noise_threshold
        if noise_flag:
            candidates.append(mappings.noise_class)
        diagnosis_candidates = [name for name in candidates if name != mappings.noise_class]
        if diagnosis_candidates:
            primary = min(diagnosis_candidates, key=lambda name: priority_rank.get(name, len(priority_rank)))
        elif noise_flag:
            primary = mappings.noise_class
        else:
            top_idx = int(softmax_probs[batch_idx].argmax().item())
            primary = mappings.class_order[top_idx]
            candidates = [primary]
        rows.append(
            {
                "primary_label": primary,
                "candidate_labels": candidates,
                "noise_prob": noise_prob,
                "softmax_top1": mappings.class_order[int(softmax_probs[batch_idx].argmax().item())],
                "softmax_top1_prob": float(softmax_probs[batch_idx].max().item()),
            }
        )
    return rows


def multilabel_candidate_metrics(
    labels_19: list[int],
    candidate_labels: list[list[str]],
    primary_labels: list[str],
    mappings: DetectionMILMappings,
) -> dict:
    y_true = [mappings.class_order[idx] for idx in labels_19]
    contains = [truth in preds for truth, preds in zip(y_true, candidate_labels)]
    pred_counts = [len(preds) for preds in candidate_labels]
    sample_precisions = [(1.0 / len(preds)) if hit and preds else 0.0 for hit, preds in zip(contains, candidate_labels)]
    sample_recalls = [1.0 if hit else 0.0 for hit in contains]
    sample_f1s = [2 * p * r / (p + r) if (p + r) else 0.0 for p, r in zip(sample_precisions, sample_recalls)]

    per_class = []
    tp_total = fp_total = fn_total = 0
    for class_name in mappings.class_order:
        tp = fp = fn = tn = 0
        for truth, preds in zip(y_true, candidate_labels):
            actual = truth == class_name
            predicted = class_name in preds
            if actual and predicted:
                tp += 1
            elif not actual and predicted:
                fp += 1
            elif actual and not predicted:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total else 0.0
        per_class.append(
            {
                "class_name": class_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn,
                "predicted_count": tp + fp,
            }
        )
        tp_total += tp
        fp_total += fp
        fn_total += fn
    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0
    return {
        "contains_true_rate": sum(contains) / max(1, len(contains)),
        "threshold_accuracy": sum(contains) / max(1, len(contains)),
        "average_predicted_labels": sum(pred_counts) / max(1, len(pred_counts)),
        "sample_precision": sum(sample_precisions) / max(1, len(sample_precisions)),
        "sample_recall": sum(sample_recalls) / max(1, len(sample_recalls)),
        "sample_f1": sum(sample_f1s) / max(1, len(sample_f1s)),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_accuracy": sum(item["accuracy"] for item in per_class) / max(1, len(per_class)),
        "macro_f1": sum(item["f1"] for item in per_class) / max(1, len(per_class)),
        "priority_primary_accuracy": sum(int(t == p) for t, p in zip(y_true, primary_labels)) / max(1, len(y_true)),
        "per_class": per_class,
    }
