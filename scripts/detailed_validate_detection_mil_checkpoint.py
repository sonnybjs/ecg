from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from train_mil_classifier import CsvBagDataset, balanced_file_subset, collate_bags, encoder_hidden_dim, move_bags_to_device  # noqa: E402
from wecfgclassifier.config import ASSUMED_INPUT_FS, DENOISER_FS, OUTPUT_ROOT, WINDOW_SEC, ExperimentConfig  # noqa: E402
from wecfgclassifier.data.pipeline import build_split_file_records, load_mapping_preset  # noqa: E402
from wecfgclassifier.models.backbones import build_classifier  # noqa: E402
from wecfgclassifier.models.detection_mil import (  # noqa: E402
    DetectionMILConfig,
    PriorityAwareDetectionMIL,
    detection_inference,
    detection_mil_loss,
    multilabel_candidate_metrics,
)
from wecfgclassifier.reporting.artifacts import write_csv  # noqa: E402
from wecfgclassifier.training.metrics import confusion_and_metrics  # noqa: E402
from wecfgclassifier.utils.runtime import TeeLogger, get_device, resolve_days, save_json, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detailed full validation for a detection MIL checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--days", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mapping-preset", default=None)
    parser.add_argument("--classifier-strategy", default=None, choices=[None, "lightweight_pytorch_baseline", "pretrained_cnn_bilstm"])
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--bag-batch-size", type=int, default=4)
    parser.add_argument("--max-bags", type=int, default=0, help="0 means full split.")
    parser.add_argument("--log-every-batches", type=int, default=250)
    parser.add_argument("--max-files-per-class", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument("--max-windows-per-file", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def next_eval_dir() -> Path:
    root = OUTPUT_ROOT / "mil_evaluations"
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in root.glob("detailed_detection_mil_eval_*") if p.is_dir())
    next_idx = max((int(p.name.split("_")[-1]) for p in existing), default=0) + 1
    path = root / f"detailed_detection_mil_eval_{next_idx:03d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def checkpoint_config_value(ckpt: dict, key: str, default=None):
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    return config.get(key, default)


def class_count_summary(records: list, class_order: list[str]) -> dict[str, int]:
    counts = Counter(rec.mapped_label for rec in records)
    return {name: counts.get(name, 0) for name in class_order}


def write_matrix_csv(path: Path, matrix: list[list[int]], class_order: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth\\pred", *class_order])
        for name, row in zip(class_order, matrix):
            writer.writerow([name, *row])


def one_vs_rest_rows_from_pred_sets(y_true: list[int], y_pred_sets: list[list[int]], class_order: list[str]) -> list[dict]:
    rows: list[dict] = []
    for idx, name in enumerate(class_order):
        tp = fp = fn = tn = 0
        for truth, pred_set in zip(y_true, y_pred_sets):
            actual = truth == idx
            predicted = idx in pred_set
            if actual and predicted:
                tp += 1
            elif not actual and predicted:
                fp += 1
            elif actual and not predicted:
                fn += 1
            else:
                tn += 1
        support = tp + fn
        predicted_count = tp + fp
        total = tp + fp + fn + tn
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / total if total else 0.0
        rows.append(
            {
                "class_name": name,
                "support": support,
                "predicted_count": predicted_count,
                "accuracy": accuracy,
                "one_vs_rest_accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "realTrue_TP": tp,
                "fakeTrue_FP": fp,
                "fakeFalse_FN": fn,
                "realFalse_TN": tn,
            }
        )
    return rows


def threshold_prediction_matrix(y_true: list[int], y_pred_sets: list[list[int]], class_order: list[str]) -> list[list[int]]:
    matrix = np.zeros((len(class_order), len(class_order)), dtype=np.int64)
    for truth, pred_set in zip(y_true, y_pred_sets):
        for pred in pred_set:
            matrix[truth, pred] += 1
    return matrix.tolist()


def build_eval_config(args: argparse.Namespace, ckpt: dict, mapping) -> ExperimentConfig:
    days = resolve_days(args.days or checkpoint_config_value(ckpt, "days", ["ALL"]))
    seed = args.seed if args.seed is not None else checkpoint_config_value(ckpt, "seed", 42)
    strategy = args.classifier_strategy or checkpoint_config_value(ckpt, "classifier_strategy", "pretrained_cnn_bilstm")
    max_files_per_class = args.max_files_per_class
    if max_files_per_class is None:
        max_files_per_class = checkpoint_config_value(ckpt, "max_files_per_class", 0)
    val_fraction = args.val_fraction if args.val_fraction is not None else checkpoint_config_value(ckpt, "val_fraction", 0.1)
    test_fraction = args.test_fraction if args.test_fraction is not None else checkpoint_config_value(ckpt, "test_fraction", 0.1)
    return ExperimentConfig(
        days=days,
        seed=seed,
        classifier_strategy=strategy,
        classifier_head="single",
        classifier_input="raw",
        classifier_only=True,
        mapping_preset=mapping.name,
        mapping_description=mapping.description,
        class_order=mapping.class_order,
        max_files_per_class=max_files_per_class,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        classifier_epochs=0,
        denoiser_epochs=0,
        batch_size=args.bag_batch_size,
        num_workers=0,
        file_batch_sampler=False,
        log_every_batches=args.log_every_batches,
        checkpoint_every_batches=0,
        validation_max_batches=args.max_bags if args.split == "val" else 0,
        test_max_batches=args.max_bags if args.split == "test" else 0,
        learning_rate_classifier=0.0,
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
        init_classifier_path=str(args.checkpoint),
        init_denoiser_path=None,
    )


def build_model(ckpt: dict, config: ExperimentConfig, class_order: list[str], logger: TeeLogger) -> PriorityAwareDetectionMIL:
    model_cfg = ckpt.get("model_config", {}).get("detection_mil", {})
    detection_config = DetectionMILConfig(**model_cfg)
    window_encoder = build_classifier(
        strategy=config.classifier_strategy,
        num_classes=len(class_order),
        logger=logger,
        init_path=None,
    )
    model = PriorityAwareDetectionMIL(
        window_encoder=window_encoder,
        class_order=class_order,
        hidden_dim=encoder_hidden_dim(config.classifier_strategy),
        config=detection_config,
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        logger.log(f"[checkpoint] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
        if missing:
            logger.log(f"[checkpoint] missing_sample={missing[:10]}")
        if unexpected:
            logger.log(f"[checkpoint] unexpected_sample={unexpected[:10]}")
    return model


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mapping_name = args.mapping_preset or checkpoint_config_value(ckpt, "mapping_preset", "raw19_identity")
    mapping = load_mapping_preset(mapping_name)
    config = build_eval_config(args, ckpt, mapping)
    max_bags = args.max_bags

    set_seed(config.seed)
    device = get_device(args.device)
    eval_dir = next_eval_dir()
    logger = TeeLogger(eval_dir / "eval.log")
    logger.log(
        json.dumps(
            {
                "eval_dir": str(eval_dir),
                "checkpoint": str(checkpoint),
                "checkpoint_epoch": ckpt.get("epoch"),
                "checkpoint_global_step": ckpt.get("global_step"),
                "device": str(device),
                "split": args.split,
                "mapping_preset": mapping.name,
                "bag_batch_size": args.bag_batch_size,
                "max_bags": max_bags,
                "full_split": max_bags <= 0,
                "class_order": mapping.class_order,
                "model_config": ckpt.get("model_config", {}),
            },
            indent=2,
        )
    )

    logger.log(f"[manifest] resolving {len(config.days)} day folders")
    file_records, _ = build_split_file_records(config, mapping, logger)
    split_records = [rec for rec in file_records if rec.split == args.split]
    eval_records = balanced_file_subset(split_records, max_bags, mapping.class_order, config.seed + (11 if args.split == "val" else 22))
    logger.log(
        f"[{args.split}] using bags={len(eval_records)}/{len(split_records)} "
        f"mode={'full' if max_bags <= 0 else 'class-balanced-subset'}"
    )
    logger.log(f"[{args.split}] class_counts={json.dumps(class_count_summary(eval_records, mapping.class_order), sort_keys=True)}")

    class_to_index = {name: idx for idx, name in enumerate(mapping.class_order)}
    dataset = CsvBagDataset(eval_records, class_to_index, args.max_windows_per_file, config.seed + 1000)
    loader = DataLoader(dataset, batch_size=args.bag_batch_size, shuffle=False, collate_fn=collate_bags)

    model = build_model(ckpt, config, mapping.class_order, logger).to(device)
    model.eval()

    total_loss = 0.0
    loss_parts: Counter = Counter()
    total_bags = 0
    y_true: list[int] = []
    y_softmax: list[int] = []
    y_primary: list[int] = []
    y_candidate_sets: list[list[int]] = []
    prediction_rows: list[dict] = []

    with torch.no_grad():
        for batch_idx, (bags, labels, metas) in enumerate(loader, start=1):
            labels = labels.to(device)
            output = model(move_bags_to_device(bags, device), return_evidence=False)
            loss, parts = detection_mil_loss(output, labels, model.mappings, model.config)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_bags += batch_size
            for key, value in parts.items():
                loss_parts[key] += float(value) * batch_size

            softmax_probs = output["primary_softmax_probs"].detach().cpu()
            diagnosis_probs = output["diagnosis_probs"].detach().cpu()
            noise_probs = output["noise_prob"].detach().cpu()
            softmax_conf, softmax_pred = softmax_probs.max(dim=1)
            inference_rows = detection_inference(output, model.mappings, model.config)

            for row_idx, meta in enumerate(metas):
                true_idx = int(labels[row_idx].detach().cpu().item())
                pred_idx = int(softmax_pred[row_idx].item())
                primary_label = inference_rows[row_idx]["primary_label"]
                primary_idx = class_to_index[primary_label]
                candidate_names = inference_rows[row_idx]["candidate_labels"]
                candidate_indices = [class_to_index[name] for name in candidate_names]
                true_class = mapping.class_order[true_idx]
                y_true.append(true_idx)
                y_softmax.append(pred_idx)
                y_primary.append(primary_idx)
                y_candidate_sets.append(candidate_indices)

                record = {
                    "file_path": meta["file_path"],
                    "file_name": meta["file_name"],
                    "n_windows": meta["n_windows"],
                    "window_indices": " ".join(str(idx) for idx in meta["window_indices"]),
                    "true_class": true_class,
                    "pred_class": mapping.class_order[pred_idx],
                    "correct": int(true_idx == pred_idx),
                    "confidence": float(softmax_conf[row_idx].item()),
                    "true_class_prob": float(softmax_probs[row_idx, true_idx].item()),
                    "primary_label": primary_label,
                    "primary_correct": int(primary_idx == true_idx),
                    "candidate_labels": "|".join(candidate_names),
                    "contains_true": int(true_class in candidate_names),
                    "num_candidate_labels": len(candidate_names),
                    "noise_prob": float(noise_probs[row_idx].item()),
                }
                for class_idx, class_name in enumerate(mapping.class_order):
                    record[f"prob_{class_name}"] = float(softmax_probs[row_idx, class_idx].item())
                for class_name in model.mappings.diagnosis_classes:
                    diag_idx = model.mappings.diagnosis_to_index_18[class_name]
                    record[f"diag_prob_{class_name}"] = float(diagnosis_probs[row_idx, diag_idx].item())
                prediction_rows.append(record)

            if batch_idx == 1 or batch_idx % args.log_every_batches == 0 or batch_idx == len(loader):
                logger.log(f"[{args.split}][batch {batch_idx}/{len(loader)}] bags={total_bags}")

    softmax_metrics = confusion_and_metrics(y_true, y_softmax, mapping.class_order)
    primary_metrics = confusion_and_metrics(y_true, y_primary, mapping.class_order)
    candidate_names = [[mapping.class_order[idx] for idx in pred_set] for pred_set in y_candidate_sets]
    primary_names = [mapping.class_order[idx] for idx in y_primary]
    candidate_metrics = multilabel_candidate_metrics(y_true, candidate_names, primary_names, model.mappings)
    candidate_matrix = threshold_prediction_matrix(y_true, y_candidate_sets, mapping.class_order)
    candidate_per_class = one_vs_rest_rows_from_pred_sets(y_true, y_candidate_sets, mapping.class_order)

    summary = {
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
        "primary_top1_accuracy": primary_metrics["accuracy"],
    }
    for key, value in loss_parts.items():
        summary[f"loss_{key}"] = value / max(1, total_bags)

    split = args.split
    write_csv(eval_dir / f"{split}_file_manifest.csv", [rec.__dict__ for rec in eval_records])
    write_csv(eval_dir / "file_manifest.csv", [rec.__dict__ for rec in eval_records])
    write_csv(eval_dir / f"{split}_predictions.csv", prediction_rows)
    write_csv(eval_dir / f"{split}_per_class_detailed.csv", candidate_per_class)
    write_csv(eval_dir / f"{split}_softmax_per_class_detailed.csv", softmax_metrics["per_class"])
    write_csv(eval_dir / f"{split}_primary_per_class_detailed.csv", primary_metrics["per_class"])
    save_json(eval_dir / f"{split}_summary.json", summary)
    save_json(eval_dir / f"{split}_confusion_matrix.json", softmax_metrics["confusion_matrix"])
    save_json(eval_dir / f"{split}_primary_confusion_matrix.json", primary_metrics["confusion_matrix"])
    save_json(eval_dir / f"{split}_threshold_prediction_matrix.json", candidate_matrix)
    write_matrix_csv(eval_dir / f"{split}_confusion_matrix.csv", softmax_metrics["confusion_matrix"], mapping.class_order)
    write_matrix_csv(eval_dir / f"{split}_primary_confusion_matrix.csv", primary_metrics["confusion_matrix"], mapping.class_order)
    write_matrix_csv(eval_dir / f"{split}_threshold_prediction_matrix.csv", candidate_matrix, mapping.class_order)
    save_json(eval_dir / "config.json", {"config": asdict(config), "checkpoint": str(checkpoint), "model_config": ckpt.get("model_config", {})})
    logger.log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
