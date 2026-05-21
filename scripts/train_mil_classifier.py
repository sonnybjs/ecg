from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from wecfgclassifier.config import ASSUMED_INPUT_FS, DENOISER_FS, OUTPUT_ROOT, WINDOW_SAMPLES, WINDOW_SEC, ExperimentConfig, FileRecord  # noqa: E402
from wecfgclassifier.data.pipeline import build_split_file_records, load_mapping_preset, read_signal_cached  # noqa: E402
from wecfgclassifier.models.backbones import build_classifier  # noqa: E402
from wecfgclassifier.models.detection_mil import (  # noqa: E402
    DetectionMILConfig,
    PriorityAwareDetectionMIL,
    detection_inference,
    detection_mil_loss,
    initialize_detection_heads_from_classifier_state,
    multilabel_candidate_metrics,
)
from wecfgclassifier.models.mil import TopKMILClassifier  # noqa: E402
from wecfgclassifier.reporting.artifacts import write_csv  # noqa: E402
from wecfgclassifier.training.metrics import confusion_and_metrics  # noqa: E402
from wecfgclassifier.utils.runtime import TeeLogger, get_device, load_matching_state_dict, next_experiment_dir, resolve_days, save_json, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSV-level top-k MIL classifier training.")
    parser.add_argument("--days", nargs="+", default=["ALL"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mapping-preset", default="raw19_identity")
    parser.add_argument("--classifier-strategy", default="pretrained_cnn_bilstm", choices=["lightweight_pytorch_baseline", "pretrained_cnn_bilstm"])
    parser.add_argument(
        "--model-type",
        default="baseline_topk_mil",
        choices=[
            "baseline_topk_mil",
            "priority_detection_mil",
            "detection_mil_topk_only",
            "detection_mil_lse_event",
            "detection_mil_noisy_or_event",
            "detection_mil_group_attention",
            "detection_mil_noise_aux",
            "detection_mil_priority_inference",
        ],
    )
    parser.add_argument("--init-classifier-path", default=None)
    parser.add_argument("--resume-mil-path", default=None)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--bag-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--topk-fraction", type=float, default=0.25)
    parser.add_argument("--min-topk", type=int, default=1)
    parser.add_argument("--detection-head-type", default="linear", choices=["linear", "classifier_mlp"])
    parser.add_argument("--detection-head-hidden-dim", type=int, default=64)
    parser.add_argument("--detection-head-dropout", type=float, default=0.3)
    parser.add_argument("--init-detection-head-from-classifier", action="store_true")
    parser.add_argument("--event-pooling", default="logsumexp", choices=["topk_mean", "max", "logsumexp", "noisy_or"])
    parser.add_argument("--event-lse-tau", type=float, default=1.0)
    parser.add_argument("--rhythm-pooling", default="topk_mean", choices=["topk_mean", "max", "logsumexp", "noisy_or"])
    parser.add_argument("--conduction-pooling", default="topk_attention_mix", choices=["topk_mean", "max", "logsumexp", "noisy_or", "topk_attention_mix"])
    parser.add_argument("--attention-mix-alpha", type=float, default=0.5)
    parser.add_argument("--noise-pooling", default="topk_mean", choices=["topk_mean", "max", "logsumexp", "noisy_or"])
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--pos-bce-weight", type=float, default=0.3)
    parser.add_argument("--group-ce-weight", type=float, default=0.2)
    parser.add_argument("--noise-weight", type=float, default=0.2)
    parser.add_argument("--sparsity-weight", type=float, default=0.0)
    parser.add_argument("--non-noise-as-weak-negative", action="store_true")
    parser.add_argument("--noise-weak-negative-weight", type=float, default=0.1)
    parser.add_argument("--global-diagnosis-threshold", type=float, default=0.3)
    parser.add_argument("--noise-threshold", type=float, default=0.5)
    parser.add_argument("--top-m-evidence-windows", type=int, default=3)
    parser.add_argument("--max-files-per-class", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--validation-max-bags", type=int, default=760)
    parser.add_argument("--test-max-bags", type=int, default=760)
    parser.add_argument("--log-every-bags", type=int, default=200)
    parser.add_argument("--checkpoint-every-bags", type=int, default=4000)
    parser.add_argument("--validation-every-bags", type=int, default=0)
    parser.add_argument("--early-stop-patience-validations", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--early-stop-metric",
        default="macro_f1",
        choices=["macro_f1", "contains_true_rate", "softmax_macro_f1", "priority_primary_accuracy"],
    )
    parser.add_argument("--early-stop-min-validations", type=int, default=0)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--freeze-batchnorm", action="store_true")
    parser.add_argument("--class-balanced-train-sampling", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--max-windows-per-file", type=int, default=0)
    args = parser.parse_args()
    if not args.init_classifier_path and not args.resume_mil_path:
        parser.error("one of --init-classifier-path or --resume-mil-path is required")
    return args


def build_config(args: argparse.Namespace, mapping) -> ExperimentConfig:
    return ExperimentConfig(
        days=resolve_days(args.days),
        seed=args.seed,
        classifier_strategy=args.classifier_strategy,
        classifier_head="single",
        classifier_input="raw",
        classifier_only=True,
        mapping_preset=mapping.name,
        mapping_description=mapping.description,
        class_order=mapping.class_order,
        max_files_per_class=args.max_files_per_class,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        classifier_epochs=args.epochs,
        denoiser_epochs=0,
        batch_size=args.bag_batch_size,
        num_workers=0,
        file_batch_sampler=False,
        log_every_batches=args.log_every_bags,
        checkpoint_every_batches=args.checkpoint_every_bags,
        validation_max_batches=args.validation_max_bags,
        test_max_batches=args.test_max_bags,
        learning_rate_classifier=args.learning_rate,
        learning_rate_denoiser=0.0,
        lambda_anchor=0.0,
        lambda_slope=0.0,
        lambda_noise=0.0,
        noise_class_name="Noise",
        noise_threshold=0.9,
        diagnosis_confidence_threshold=0.7,
        window_sec=WINDOW_SEC,
        input_fs=ASSUMED_INPUT_FS,
        denoiser_fs=DENOISER_FS,
        init_classifier_path=args.init_classifier_path or args.resume_mil_path,
        init_denoiser_path=None,
    )


def build_detection_config(args: argparse.Namespace) -> DetectionMILConfig:
    event_pooling = args.event_pooling
    conduction_pooling = args.conduction_pooling
    group_ce_weight = args.group_ce_weight
    if args.model_type == "detection_mil_topk_only":
        event_pooling = "topk_mean"
        conduction_pooling = "topk_mean"
        group_ce_weight = 0.0
    elif args.model_type in {"priority_detection_mil", "detection_mil_lse_event", "detection_mil_group_attention", "detection_mil_noise_aux", "detection_mil_priority_inference"}:
        event_pooling = "logsumexp" if args.event_pooling == "logsumexp" else args.event_pooling
    elif args.model_type == "detection_mil_noisy_or_event":
        event_pooling = "noisy_or"

    return DetectionMILConfig(
        evidence_head_type=args.detection_head_type,
        evidence_head_hidden_dim=args.detection_head_hidden_dim,
        evidence_head_dropout=args.detection_head_dropout,
        event_pooling=event_pooling,
        event_lse_tau=args.event_lse_tau,
        rhythm_pooling=args.rhythm_pooling,
        rhythm_topk_fraction=args.topk_fraction,
        rhythm_min_topk=args.min_topk,
        conduction_pooling=conduction_pooling,
        conduction_topk_fraction=args.topk_fraction,
        conduction_min_topk=args.min_topk,
        attention_mix_alpha=args.attention_mix_alpha,
        default_topk_fraction=args.topk_fraction,
        default_min_topk=args.min_topk,
        noise_pooling=args.noise_pooling,
        noise_topk_fraction=args.topk_fraction,
        noise_min_topk=args.min_topk,
        ce_weight=args.ce_weight,
        pos_bce_weight=args.pos_bce_weight,
        group_ce_weight=group_ce_weight,
        noise_weight=args.noise_weight,
        sparsity_weight=args.sparsity_weight,
        non_noise_as_weak_negative=args.non_noise_as_weak_negative,
        noise_weak_negative_weight=args.noise_weak_negative_weight,
        global_diagnosis_threshold=args.global_diagnosis_threshold,
        noise_threshold=args.noise_threshold,
        top_m_evidence_windows=args.top_m_evidence_windows,
    )


def encoder_hidden_dim(strategy: str) -> int:
    if strategy == "lightweight_pytorch_baseline":
        return 96
    if strategy == "pretrained_cnn_bilstm":
        return 256
    raise ValueError(f"Unsupported classifier strategy: {strategy}")


def balanced_file_subset(records: list[FileRecord], max_records: int, class_order: list[str], seed: int) -> list[FileRecord]:
    if max_records <= 0 or len(records) <= max_records:
        return records
    by_class: dict[str, list[FileRecord]] = defaultdict(list)
    for rec in records:
        by_class[rec.mapped_label].append(rec)
    rng = random.Random(seed)
    for rows in by_class.values():
        rng.shuffle(rows)
    selected: list[FileRecord] = []
    active = [name for name in class_order if by_class.get(name)]
    while len(selected) < max_records and active:
        next_active = []
        for name in active:
            rows = by_class[name]
            if rows and len(selected) < max_records:
                selected.append(rows.pop())
            if rows:
                next_active.append(name)
        active = next_active
    rng.shuffle(selected)
    return selected


def class_counts(records: list[FileRecord], class_order: list[str]) -> dict[str, int]:
    counts = Counter(rec.mapped_label for rec in records)
    return {name: counts.get(name, 0) for name in class_order}


def class_balanced_sampler(records: list[FileRecord], class_order: list[str]) -> WeightedRandomSampler:
    counts = Counter(rec.mapped_label for rec in records)
    weights = []
    for rec in records:
        count = max(1, counts.get(rec.mapped_label, 0))
        weights.append(1.0 / count)
    return WeightedRandomSampler(weights=weights, num_samples=len(records), replacement=True)


class CsvBagDataset(Dataset):
    def __init__(
        self,
        records: list[FileRecord],
        class_to_index: dict[str, int],
        max_windows_per_file: int = 0,
        seed: int = 42,
    ) -> None:
        self.records = records
        self.class_to_index = class_to_index
        self.max_windows_per_file = max_windows_per_file
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        rec = self.records[index]
        signal = read_signal_cached(rec.file_path)
        n_windows = int(len(signal) // WINDOW_SAMPLES)
        window_indices = list(range(n_windows))
        if self.max_windows_per_file > 0 and len(window_indices) > self.max_windows_per_file:
            rng = random.Random(self.seed + index)
            window_indices = sorted(rng.sample(window_indices, self.max_windows_per_file))
        windows = []
        for window_index in window_indices:
            start = window_index * WINDOW_SAMPLES
            end = start + WINDOW_SAMPLES
            windows.append(torch.from_numpy(signal[start:end].copy()).float())
        bag = torch.stack(windows)
        label = torch.tensor(self.class_to_index[rec.mapped_label], dtype=torch.long)
        meta = {
            "file_path": rec.file_path,
            "file_name": rec.file_name,
            "mapped_label": rec.mapped_label,
            "n_windows": len(window_indices),
            "window_indices": window_indices,
        }
        return bag, label, meta


def collate_bags(batch):
    bags, labels, metas = zip(*batch)
    return list(bags), torch.stack(labels), list(metas)


def move_bags_to_device(bags: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [bag.to(device) for bag in bags]


def run_eval(
    model: TopKMILClassifier,
    loader: DataLoader,
    device: torch.device,
    criterion,
    class_order: list[str],
    logger: TeeLogger,
    prefix: str,
    log_every: int,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_bags = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch_idx, (bags, labels, _) in enumerate(loader, start=1):
            labels = labels.to(device)
            output = model(move_bags_to_device(bags, device))
            loss = criterion(output["bag_logits"], labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_bags += labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(output["bag_logits"].argmax(dim=1).cpu().tolist())
            if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == len(loader):
                logger.log(f"[{prefix}][batch {batch_idx}/{len(loader)}] bags={total_bags}")
    metrics = confusion_and_metrics(y_true, y_pred, class_order)
    metrics["loss"] = total_loss / max(1, total_bags)
    metrics["evaluated_bags"] = total_bags
    return metrics


def save_mil_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    global_step: int,
    metrics: dict,
    config: ExperimentConfig,
    model_config: dict | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
        "config": asdict(config),
    }
    if model_config is not None:
        payload["model_config"] = model_config
    torch.save(payload, path)


def move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def export_evidence(
    path: Path,
    model: TopKMILClassifier,
    dataset: CsvBagDataset,
    device: torch.device,
    class_order: list[str],
    max_files: int = 200,
) -> None:
    rows = []
    model.eval()
    with torch.no_grad():
        for index in range(min(max_files, len(dataset))):
            bag, label, meta = dataset[index]
            output = model.forward_bag(bag.to(device))
            instance_probs = torch.softmax(output["instance_logits"], dim=1).cpu().numpy()
            bag_probs = torch.softmax(output["bag_logits"], dim=0).cpu().numpy()
            pred_idx = int(np.argmax(bag_probs))
            true_idx = int(label.item())
            for local_idx, window_index in enumerate(meta["window_indices"]):
                top_class = int(instance_probs[local_idx].argmax())
                rows.append(
                    {
                        "file_name": meta["file_name"],
                        "file_path": meta["file_path"],
                        "true_class": class_order[true_idx],
                        "bag_pred_class": class_order[pred_idx],
                        "window_index": window_index,
                        "start_sec": window_index * WINDOW_SEC,
                        "end_sec": (window_index + 1) * WINDOW_SEC,
                        "window_pred_class": class_order[top_class],
                        "window_pred_prob": float(instance_probs[local_idx, top_class]),
                        "true_class_window_prob": float(instance_probs[local_idx, true_idx]),
                        "bag_true_class_prob": float(bag_probs[true_idx]),
                        "bag_pred_class_prob": float(bag_probs[pred_idx]),
                    }
                )
    write_csv(path, rows)


def load_window_encoder_init(window_encoder, init_path: str | None, logger: TeeLogger) -> None:
    if not init_path:
        return
    path = Path(init_path).expanduser().resolve()
    state = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    if any(key.startswith("window_classifier.") for key in state_dict):
        state_dict = {key.removeprefix("window_classifier."): value for key, value in state_dict.items() if key.startswith("window_classifier.")}
    elif any(key.startswith("window_encoder.") for key in state_dict):
        state_dict = {key.removeprefix("window_encoder."): value for key, value in state_dict.items() if key.startswith("window_encoder.")}
    load_matching_state_dict(window_encoder, state_dict, logger, f"[init] detection window encoder from {path.name}")


def load_checkpoint_state_dict(path_like: str | None) -> dict[str, torch.Tensor] | None:
    if not path_like:
        return None
    path = Path(path_like).expanduser().resolve()
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state["model_state"] if isinstance(state, dict) and "model_state" in state else state


def build_detection_model(args: argparse.Namespace, mapping, logger: TeeLogger) -> PriorityAwareDetectionMIL:
    window_encoder = build_classifier(
        strategy=args.classifier_strategy,
        num_classes=len(mapping.class_order),
        logger=logger,
        init_path=None,
    )
    load_window_encoder_init(window_encoder, args.init_classifier_path, logger)
    detection_config = build_detection_config(args)
    model = PriorityAwareDetectionMIL(
        window_encoder=window_encoder,
        class_order=mapping.class_order,
        hidden_dim=encoder_hidden_dim(args.classifier_strategy),
        config=detection_config,
    )
    if args.init_detection_head_from_classifier:
        state_dict = load_checkpoint_state_dict(args.init_classifier_path or args.resume_mil_path)
        if state_dict is None:
            logger.log("[init] detection head classifier init skipped: no checkpoint state")
        else:
            report = initialize_detection_heads_from_classifier_state(model, state_dict)
            logger.log(f"[init] detection heads from classifier checkpoint: {json.dumps(report, sort_keys=True)}")
    return model


def set_batchnorm_eval(module: torch.nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()


def run_detection_eval(
    model: PriorityAwareDetectionMIL,
    loader: DataLoader,
    device: torch.device,
    class_order: list[str],
    logger: TeeLogger,
    prefix: str,
    log_every: int,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_bags = 0
    y_true: list[int] = []
    y_softmax: list[int] = []
    primary_labels: list[str] = []
    candidate_labels: list[list[str]] = []
    loss_parts: Counter = Counter()
    with torch.no_grad():
        for batch_idx, (bags, labels, _) in enumerate(loader, start=1):
            labels = labels.to(device)
            output = model(move_bags_to_device(bags, device), return_evidence=False)
            loss, parts = detection_mil_loss(output, labels, model.mappings, model.config)
            total_loss += float(loss.item()) * labels.size(0)
            total_bags += labels.size(0)
            for key, value in parts.items():
                loss_parts[key] += float(value) * labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_softmax.extend(output["bag_logits_19"].argmax(dim=1).cpu().tolist())
            inference_rows = detection_inference(output, model.mappings, model.config)
            primary_labels.extend(row["primary_label"] for row in inference_rows)
            candidate_labels.extend(row["candidate_labels"] for row in inference_rows)
            if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == len(loader):
                logger.log(f"[{prefix}][batch {batch_idx}/{len(loader)}] bags={total_bags}")
    softmax_metrics = confusion_and_metrics(y_true, y_softmax, class_order)
    candidate_metrics = multilabel_candidate_metrics(y_true, candidate_labels, primary_labels, model.mappings)
    metrics = {
        "loss": total_loss / max(1, total_bags),
        "evaluated_bags": total_bags,
        "softmax_top1_accuracy": softmax_metrics["accuracy"],
        "softmax_macro_f1": softmax_metrics["macro_f1"],
        "priority_primary_accuracy": candidate_metrics["priority_primary_accuracy"],
        "contains_true_rate": candidate_metrics["contains_true_rate"],
        "threshold_accuracy": candidate_metrics["threshold_accuracy"],
        "average_predicted_labels": candidate_metrics["average_predicted_labels"],
        "sample_precision": candidate_metrics["sample_precision"],
        "sample_recall": candidate_metrics["sample_recall"],
        "sample_f1": candidate_metrics["sample_f1"],
        "micro_precision": candidate_metrics["micro_precision"],
        "micro_recall": candidate_metrics["micro_recall"],
        "micro_f1": candidate_metrics["micro_f1"],
        "macro_accuracy": candidate_metrics["macro_accuracy"],
        "macro_f1": candidate_metrics["macro_f1"],
        "softmax_confusion_matrix": softmax_metrics["confusion_matrix"],
        "per_class": candidate_metrics["per_class"],
    }
    for key, value in loss_parts.items():
        metrics[f"loss_{key}"] = value / max(1, total_bags)
    return metrics


def export_detection_evidence(
    path: Path,
    model: PriorityAwareDetectionMIL,
    dataset: CsvBagDataset,
    device: torch.device,
    max_files: int = 200,
) -> None:
    rows = []
    model.eval()
    with torch.no_grad():
        for index in range(min(max_files, len(dataset))):
            bag, label, meta = dataset[index]
            output = model([bag.to(device)], return_evidence=True)
            inference = detection_inference(output, model.mappings, model.config)[0]
            true_class = model.mappings.class_order[int(label.item())]
            labels_to_export = set(inference["candidate_labels"])
            labels_to_export.add(true_class)
            labels_to_export.add(inference["softmax_top1"])
            evidence_probs = output["evidence_probs"][0].cpu()
            attention = output["group_attention_weights"][0].cpu()
            noise_probs = output["noise_window_probs"][0].cpu()
            for class_name in sorted(labels_to_export):
                if class_name == model.mappings.noise_class:
                    values = noise_probs
                    top_values, top_indices = torch.topk(values, k=min(model.config.top_m_evidence_windows, values.numel()))
                    for rank, (value, window_index) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
                        rows.append(
                            {
                                "file_name": meta["file_name"],
                                "file_path": meta["file_path"],
                                "true_class": true_class,
                                "primary_label": inference["primary_label"],
                                "candidate_labels": "|".join(inference["candidate_labels"]),
                                "evidence_class": class_name,
                                "rank": rank,
                                "window_index": meta["window_indices"][window_index],
                                "start_sec": meta["window_indices"][window_index] * WINDOW_SEC,
                                "end_sec": (meta["window_indices"][window_index] + 1) * WINDOW_SEC,
                                "evidence_prob": float(value),
                                "attention_weight": "",
                            }
                        )
                    continue
                if class_name not in model.mappings.diagnosis_to_index_18:
                    continue
                diag_idx = model.mappings.diagnosis_to_index_18[class_name]
                group_id = model.mappings.diagnosis_class_to_group_id[class_name]
                values = evidence_probs[:, diag_idx]
                top_values, top_indices = torch.topk(values, k=min(model.config.top_m_evidence_windows, values.numel()))
                for rank, (value, window_index) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
                    rows.append(
                        {
                            "file_name": meta["file_name"],
                            "file_path": meta["file_path"],
                            "true_class": true_class,
                            "primary_label": inference["primary_label"],
                            "candidate_labels": "|".join(inference["candidate_labels"]),
                            "evidence_class": class_name,
                            "rank": rank,
                            "window_index": meta["window_indices"][window_index],
                            "start_sec": meta["window_indices"][window_index] * WINDOW_SEC,
                            "end_sec": (meta["window_indices"][window_index] + 1) * WINDOW_SEC,
                            "evidence_prob": float(value),
                            "attention_weight": float(attention[window_index, group_id]),
                        }
                    )
    write_csv(path, rows)


def run_detection_training(
    args: argparse.Namespace,
    config: ExperimentConfig,
    mapping,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    val_dataset: CsvBagDataset,
    device: torch.device,
    exp_dir: Path,
    logger: TeeLogger,
) -> None:
    model = build_detection_model(args, mapping, logger).to(device)
    resume_ckpt = None
    if args.resume_mil_path:
        resume_path = Path(args.resume_mil_path).expanduser().resolve()
        resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(resume_ckpt["model_state"], strict=False)
        logger.log(
            f"[resume] loaded detection MIL model from {resume_path} "
            f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
        )
        if missing:
            logger.log(f"[resume] missing_sample={missing[:10]}")
        if unexpected:
            logger.log(f"[resume] unexpected_sample={unexpected[:10]}")
    if args.freeze_encoder:
        for param in model.window_encoder.parameters():
            param.requires_grad = False
        logger.log("[init] frozen detection window encoder")
    if args.freeze_batchnorm:
        set_batchnorm_eval(model)
        logger.log("[init] frozen BatchNorm running stats")

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    history: list[dict] = []
    interval_history: list[dict] = []
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_config = {"model_type": args.model_type, "detection_mil": asdict(model.config)}
    best_score = -1.0
    global_step = 0
    no_improve_count = 0
    should_stop = False
    if resume_ckpt is not None:
        global_step = int(resume_ckpt.get("global_step", 0))
        resume_metrics = resume_ckpt.get("metrics") or {}
        best_score = float(resume_metrics.get("macro_f1", -1.0))
        if args.resume_optimizer and resume_ckpt.get("optimizer_state") is not None:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            move_optimizer_state_to_device(optimizer, device)
            logger.log("[resume] loaded optimizer state")
        logger.log(f"[resume] starting global_step={global_step} prior_score={best_score}")

    log_every_eval = max(1, math.ceil(args.log_every_bags / max(1, args.bag_batch_size)))
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_batchnorm:
            set_batchnorm_eval(model)
        total_loss = 0.0
        total_bags = 0
        y_true: list[int] = []
        y_softmax: list[int] = []
        loss_parts: Counter = Counter()
        for batch_idx, (bags, labels, _) in enumerate(train_loader, start=1):
            labels = labels.to(device)
            output = model(move_bags_to_device(bags, device), return_evidence=args.sparsity_weight > 0)
            loss, parts = detection_mil_loss(output, labels, model.mappings, model.config)
            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            global_step += labels.size(0)
            total_loss += float(loss.item()) * labels.size(0)
            total_bags += labels.size(0)
            for key, value in parts.items():
                loss_parts[key] += float(value) * labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_softmax.extend(output["bag_logits_19"].argmax(dim=1).detach().cpu().tolist())
            if batch_idx == 1 or total_bags % args.log_every_bags < labels.size(0) or batch_idx == len(train_loader):
                part_msg = " ".join(f"{key}={loss_parts[key] / max(1, total_bags):.4f}" for key in ["ce", "pos_bce", "group_ce", "noise"])
                logger.log(
                    f"[detection_mil][epoch {epoch}/{args.epochs}][batch {batch_idx}/{len(train_loader)}] "
                    f"bags={total_bags} loss={loss.item():.4f} running_loss={total_loss / max(1, total_bags):.4f} {part_msg}"
                )
            if args.checkpoint_every_bags > 0 and global_step % args.checkpoint_every_bags < labels.size(0):
                metrics = {"running_loss": total_loss / max(1, total_bags)}
                save_mil_checkpoint(ckpt_dir / f"step_{global_step:08d}.ckpt", model, optimizer, epoch, global_step, metrics, config, model_config)
                save_mil_checkpoint(ckpt_dir / "interval_latest.ckpt", model, optimizer, epoch, global_step, metrics, config, model_config)
                logger.log(f"[detection_mil] saved interval checkpoint global_step={global_step}")
            if args.validation_every_bags > 0 and global_step % args.validation_every_bags < labels.size(0):
                val_metrics = run_detection_eval(
                    model,
                    val_loader,
                    device,
                    mapping.class_order,
                    logger,
                    f"detection_mil][step {global_step}][val",
                    log_every_eval,
                )
                interval_row = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_running_loss": total_loss / max(1, total_bags),
                    "val_loss": val_metrics["loss"],
                    "val_softmax_accuracy": val_metrics["softmax_top1_accuracy"],
                    "val_softmax_macro_f1": val_metrics["softmax_macro_f1"],
                    "val_contains_true_rate": val_metrics["contains_true_rate"],
                    "val_priority_primary_accuracy": val_metrics["priority_primary_accuracy"],
                    "val_candidate_macro_accuracy": val_metrics["macro_accuracy"],
                    "val_candidate_macro_f1": val_metrics["macro_f1"],
                    "val_avg_predicted_labels": val_metrics["average_predicted_labels"],
                }
                interval_history.append(interval_row)
                write_csv(exp_dir / "mil_interval_history.csv", interval_history)
                save_json(exp_dir / "last_interval_val_metrics.json", val_metrics)
                score = float(val_metrics[args.early_stop_metric])
                improved = score > best_score + args.early_stop_min_delta
                logger.log(
                    f"[detection_mil][step {global_step}] val_{args.early_stop_metric}={score:.4f} "
                    f"best={best_score:.4f} improved={improved}"
                )
                validation_count = len(interval_history)
                if improved:
                    best_score = score
                    no_improve_count = 0
                    save_mil_checkpoint(ckpt_dir / "best.ckpt", model, optimizer, epoch, global_step, val_metrics, config, model_config)
                    save_json(exp_dir / "best_val_metrics.json", val_metrics)
                else:
                    no_improve_count += 1
                    if (
                        args.early_stop_patience_validations > 0
                        and validation_count >= args.early_stop_min_validations
                        and no_improve_count >= args.early_stop_patience_validations
                    ):
                        logger.log(
                            f"[detection_mil] early stopping after {no_improve_count} validation checks "
                            f"without improvement on {args.early_stop_metric}"
                        )
                        should_stop = True
                        break
                model.train()
                if args.freeze_batchnorm:
                    set_batchnorm_eval(model)
        if should_stop:
            break

        train_softmax = confusion_and_metrics(y_true, y_softmax, mapping.class_order)
        val_metrics = run_detection_eval(model, val_loader, device, mapping.class_order, logger, f"detection_mil][epoch {epoch}][val", log_every_eval)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_bags),
            "train_softmax_accuracy": train_softmax["accuracy"],
            "train_softmax_macro_f1": train_softmax["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_softmax_accuracy": val_metrics["softmax_top1_accuracy"],
            "val_softmax_macro_f1": val_metrics["softmax_macro_f1"],
            "val_contains_true_rate": val_metrics["contains_true_rate"],
            "val_priority_primary_accuracy": val_metrics["priority_primary_accuracy"],
            "val_candidate_macro_accuracy": val_metrics["macro_accuracy"],
            "val_candidate_macro_f1": val_metrics["macro_f1"],
            "val_avg_predicted_labels": val_metrics["average_predicted_labels"],
        }
        history.append(row)
        write_csv(exp_dir / "mil_history.csv", history)
        save_json(exp_dir / "last_val_metrics.json", val_metrics)
        save_mil_checkpoint(ckpt_dir / "last.ckpt", model, optimizer, epoch, global_step, val_metrics, config, model_config)
        logger.log(
            f"[detection_mil][epoch {epoch}/{args.epochs}] train_loss={row['train_loss']:.4f} "
            f"val_softmax_f1={row['val_softmax_macro_f1']:.4f} val_contains={row['val_contains_true_rate']:.4f} "
            f"val_primary_acc={row['val_priority_primary_accuracy']:.4f}"
        )
        score = float(val_metrics[args.early_stop_metric])
        if score > best_score + args.early_stop_min_delta:
            best_score = score
            no_improve_count = 0
            save_mil_checkpoint(ckpt_dir / "best.ckpt", model, optimizer, epoch, global_step, val_metrics, config, model_config)
            save_json(exp_dir / "best_val_metrics.json", val_metrics)
        else:
            no_improve_count += 1
            validation_count = len(interval_history) + len(history)
            if (
                args.early_stop_patience_validations > 0
                and validation_count >= args.early_stop_min_validations
                and no_improve_count >= args.early_stop_patience_validations
            ):
                logger.log(
                    f"[detection_mil] early stopping after {no_improve_count} validation checks "
                    f"without improvement on {args.early_stop_metric}"
                )
                break

    test_metrics = run_detection_eval(model, test_loader, device, mapping.class_order, logger, "detection_mil][test", log_every_eval)
    save_json(exp_dir / "test_metrics.json", test_metrics)
    export_detection_evidence(exp_dir / "val_detection_evidence.csv", model, val_dataset, device)
    save_json(exp_dir / "experiment_config.json", asdict(config))
    save_json(exp_dir / "detection_model_config.json", model_config)
    logger.log(json.dumps({"exp_dir": str(exp_dir), "best_val_candidate_macro_f1": best_score, "test_candidate_macro_f1": test_metrics["macro_f1"]}, indent=2))


def main() -> None:
    args = parse_args()
    mapping = load_mapping_preset(args.mapping_preset)
    config = build_config(args, mapping)
    set_seed(config.seed)
    device = get_device(args.device)
    exp_dir = next_experiment_dir(OUTPUT_ROOT / "mil_experiments")
    logger = TeeLogger(exp_dir / "train.log")
    logger.log(json.dumps({"exp_dir": str(exp_dir), "device": str(device), "config": asdict(config)}, indent=2))

    file_records, _ = build_split_file_records(config, mapping, logger)
    train_records = [rec for rec in file_records if rec.split == "train"]
    val_records_full = [rec for rec in file_records if rec.split == "val"]
    test_records_full = [rec for rec in file_records if rec.split == "test"]
    val_records = balanced_file_subset(val_records_full, args.validation_max_bags, mapping.class_order, args.seed + 11)
    test_records = balanced_file_subset(test_records_full, args.test_max_bags, mapping.class_order, args.seed + 22)
    logger.log(f"[train] files={len(train_records)} class_counts={json.dumps(class_counts(train_records, mapping.class_order), sort_keys=True)}")
    logger.log(f"[val] files={len(val_records)}/{len(val_records_full)} class_counts={json.dumps(class_counts(val_records, mapping.class_order), sort_keys=True)}")
    logger.log(f"[test] files={len(test_records)}/{len(test_records_full)} class_counts={json.dumps(class_counts(test_records, mapping.class_order), sort_keys=True)}")

    class_to_index = {name: idx for idx, name in enumerate(mapping.class_order)}
    train_dataset = CsvBagDataset(train_records, class_to_index, args.max_windows_per_file, args.seed)
    val_dataset = CsvBagDataset(val_records, class_to_index, args.max_windows_per_file, args.seed + 1000)
    test_dataset = CsvBagDataset(test_records, class_to_index, args.max_windows_per_file, args.seed + 2000)
    if args.class_balanced_train_sampling:
        train_sampler = class_balanced_sampler(train_records, mapping.class_order)
        train_loader = DataLoader(train_dataset, batch_size=args.bag_batch_size, sampler=train_sampler, collate_fn=collate_bags)
        logger.log("[train] using inverse-frequency class-balanced sampling")
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.bag_batch_size, shuffle=True, collate_fn=collate_bags)
    val_loader = DataLoader(val_dataset, batch_size=args.bag_batch_size, shuffle=False, collate_fn=collate_bags)
    test_loader = DataLoader(test_dataset, batch_size=args.bag_batch_size, shuffle=False, collate_fn=collate_bags)

    if args.model_type != "baseline_topk_mil":
        run_detection_training(
            args=args,
            config=config,
            mapping=mapping,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            val_dataset=val_dataset,
            device=device,
            exp_dir=exp_dir,
            logger=logger,
        )
        return

    window_classifier = build_classifier(
        strategy=args.classifier_strategy,
        num_classes=len(mapping.class_order),
        logger=logger,
        init_path=args.init_classifier_path,
    )
    model = TopKMILClassifier(window_classifier, topk_fraction=args.topk_fraction, min_topk=args.min_topk).to(device)
    resume_ckpt = None
    if args.resume_mil_path:
        resume_path = Path(args.resume_mil_path).expanduser().resolve()
        resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(resume_ckpt["model_state"], strict=False)
        logger.log(
            f"[resume] loaded MIL model from {resume_path} "
            f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
        )
        if missing:
            logger.log(f"[resume] missing_sample={missing[:10]}")
        if unexpected:
            logger.log(f"[resume] unexpected_sample={unexpected[:10]}")
    if args.freeze_encoder:
        for name, param in model.window_classifier.named_parameters():
            if "classifier" not in name and "head" not in name:
                param.requires_grad = False
        logger.log("[init] frozen encoder parameters; classifier head remains trainable")

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    history: list[dict] = []
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    global_step = 0
    if resume_ckpt is not None:
        global_step = int(resume_ckpt.get("global_step", 0))
        resume_metrics = resume_ckpt.get("metrics") or {}
        best_score = float(resume_metrics.get("macro_f1", -1.0))
        if args.resume_optimizer and resume_ckpt.get("optimizer_state") is not None:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            move_optimizer_state_to_device(optimizer, device)
            logger.log("[resume] loaded optimizer state")
        logger.log(f"[resume] starting global_step={global_step} prior_score={best_score}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_bags = 0
        y_true: list[int] = []
        y_pred: list[int] = []
        for batch_idx, (bags, labels, _) in enumerate(train_loader, start=1):
            labels = labels.to(device)
            output = model(move_bags_to_device(bags, device))
            loss = criterion(output["bag_logits"], labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += labels.size(0)
            total_loss += float(loss.item()) * labels.size(0)
            total_bags += labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(output["bag_logits"].argmax(dim=1).detach().cpu().tolist())
            if batch_idx == 1 or total_bags % args.log_every_bags < labels.size(0) or batch_idx == len(train_loader):
                logger.log(f"[mil][epoch {epoch}/{args.epochs}][batch {batch_idx}/{len(train_loader)}] bags={total_bags} loss={loss.item():.4f}")
            if args.checkpoint_every_bags > 0 and global_step % args.checkpoint_every_bags < labels.size(0):
                save_mil_checkpoint(ckpt_dir / f"step_{global_step:08d}.ckpt", model, optimizer, epoch, global_step, {"running_loss": total_loss / max(1, total_bags)}, config)
                save_mil_checkpoint(ckpt_dir / "interval_latest.ckpt", model, optimizer, epoch, global_step, {"running_loss": total_loss / max(1, total_bags)}, config)
                logger.log(f"[mil] saved interval checkpoint global_step={global_step}")

        train_metrics = confusion_and_metrics(y_true, y_pred, mapping.class_order)
        train_metrics["loss"] = total_loss / max(1, total_bags)
        val_metrics = run_eval(model, val_loader, device, criterion, mapping.class_order, logger, f"mil][epoch {epoch}][val", max(1, math.ceil(args.log_every_bags / max(1, args.bag_batch_size))))
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        write_csv(exp_dir / "mil_history.csv", history)
        save_json(exp_dir / "last_val_metrics.json", val_metrics)
        save_mil_checkpoint(ckpt_dir / "last.ckpt", model, optimizer, epoch, global_step, val_metrics, config)
        logger.log(
            f"[mil][epoch {epoch}/{args.epochs}] train_loss={row['train_loss']:.4f} "
            f"train_f1={row['train_macro_f1']:.4f} val_loss={row['val_loss']:.4f} val_f1={row['val_macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            save_mil_checkpoint(ckpt_dir / "best.ckpt", model, optimizer, epoch, global_step, val_metrics, config)
            save_json(exp_dir / "best_val_metrics.json", val_metrics)

    test_metrics = run_eval(model, test_loader, device, criterion, mapping.class_order, logger, "mil][test", max(1, math.ceil(args.log_every_bags / max(1, args.bag_batch_size))))
    save_json(exp_dir / "test_metrics.json", test_metrics)
    export_evidence(exp_dir / "val_window_evidence.csv", model, val_dataset, device, mapping.class_order)
    save_json(exp_dir / "experiment_config.json", asdict(config))
    logger.log(json.dumps({"exp_dir": str(exp_dir), "best_val_macro_f1": best_score, "test_macro_f1": test_metrics["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
