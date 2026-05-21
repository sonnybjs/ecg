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

from train_mil_classifier import CsvBagDataset, balanced_file_subset, collate_bags, move_bags_to_device  # noqa: E402
from wecfgclassifier.config import ASSUMED_INPUT_FS, DENOISER_FS, OUTPUT_ROOT, WINDOW_SEC, ExperimentConfig  # noqa: E402
from wecfgclassifier.data.pipeline import build_split_file_records, load_mapping_preset  # noqa: E402
from wecfgclassifier.models.backbones import build_classifier  # noqa: E402
from wecfgclassifier.models.mil import TopKMILClassifier  # noqa: E402
from wecfgclassifier.reporting.artifacts import write_csv  # noqa: E402
from wecfgclassifier.utils.runtime import TeeLogger, get_device, resolve_days, save_json, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold multi-label style validation for one MIL checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--ensure-at-least-one", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--days", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mapping-preset", default=None)
    parser.add_argument("--classifier-strategy", default=None, choices=[None, "lightweight_pytorch_baseline", "pretrained_cnn_bilstm"])
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--bag-batch-size", type=int, default=4)
    parser.add_argument("--max-bags", type=int, default=0)
    parser.add_argument("--log-every-batches", type=int, default=250)
    parser.add_argument("--max-files-per-class", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument("--topk-fraction", type=float, default=None)
    parser.add_argument("--min-topk", type=int, default=None)
    parser.add_argument("--max-windows-per-file", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def next_threshold_eval_dir() -> Path:
    root = OUTPUT_ROOT / "mil_evaluations"
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in root.glob("threshold_mil_eval_*") if p.is_dir())
    next_idx = max((int(p.name.split("_")[-1]) for p in existing), default=0) + 1
    path = root / f"threshold_mil_eval_{next_idx:03d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def checkpoint_config_value(ckpt: dict, key: str, default=None):
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    return config.get(key, default)


def class_count_summary(records: list, class_order: list[str]) -> dict[str, int]:
    counts = Counter(rec.mapped_label for rec in records)
    return {name: counts.get(name, 0) for name in class_order}


def build_eval_config(args: argparse.Namespace, ckpt: dict, mapping) -> ExperimentConfig:
    days = resolve_days(args.days or checkpoint_config_value(ckpt, "days", ["ALL"]))
    seed = args.seed if args.seed is not None else checkpoint_config_value(ckpt, "seed", 42)
    strategy = args.classifier_strategy or checkpoint_config_value(ckpt, "classifier_strategy", "pretrained_cnn_bilstm")
    max_files_per_class = args.max_files_per_class
    if max_files_per_class is None:
        max_files_per_class = checkpoint_config_value(ckpt, "max_files_per_class", 0)
    val_fraction = args.val_fraction
    if val_fraction is None:
        val_fraction = checkpoint_config_value(ckpt, "val_fraction", 0.1)
    test_fraction = args.test_fraction
    if test_fraction is None:
        test_fraction = checkpoint_config_value(ckpt, "test_fraction", 0.1)

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


def write_matrix_csv(path: Path, matrix: list[list[int]], class_order: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth\\pred", *class_order])
        for name, row in zip(class_order, matrix):
            writer.writerow([name, *row])


def top1_confusion_stats(y_true: list[int], y_pred: list[int], class_order: list[str]) -> tuple[list[list[int]], list[dict]]:
    num_classes = len(class_order)
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        conf[truth, pred] += 1
    return conf.tolist(), one_vs_rest_rows_from_counts(conf, class_order)


def threshold_matrix_and_rows(
    y_true: list[int],
    y_pred_sets: list[list[int]],
    class_order: list[str],
) -> tuple[list[list[int]], list[dict]]:
    num_classes = len(class_order)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred_set in zip(y_true, y_pred_sets):
        for pred in pred_set:
            matrix[truth, pred] += 1
    return matrix.tolist(), one_vs_rest_rows_from_pred_sets(y_true, y_pred_sets, class_order)


def one_vs_rest_rows_from_counts(conf: np.ndarray, class_order: list[str]) -> list[dict]:
    total = int(conf.sum())
    rows: list[dict] = []
    for idx, name in enumerate(class_order):
        tp = int(conf[idx, idx])
        fp = int(conf[:, idx].sum() - tp)
        fn = int(conf[idx, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        rows.append(metric_row(name, tp, fp, fn, tn))
    return rows


def one_vs_rest_rows_from_pred_sets(y_true: list[int], y_pred_sets: list[list[int]], class_order: list[str]) -> list[dict]:
    rows: list[dict] = []
    for idx, name in enumerate(class_order):
        tp = fp = fn = tn = 0
        for truth, pred_set in zip(y_true, y_pred_sets):
            predicted = idx in pred_set
            actual = truth == idx
            if actual and predicted:
                tp += 1
            elif not actual and predicted:
                fp += 1
            elif actual and not predicted:
                fn += 1
            else:
                tn += 1
        rows.append(metric_row(name, tp, fp, fn, tn))
    return rows


def metric_row(class_name: str, tp: int, fp: int, fn: int, tn: int) -> dict:
    support = tp + fn
    predicted = tp + fp
    total = tp + fp + fn + tn
    precision = tp / predicted if predicted else 0.0
    recall = tp / support if support else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    one_vs_rest_accuracy = (tp + tn) / total if total else 0.0
    return {
        "class_name": class_name,
        "support": support,
        "predicted_count": predicted,
        "accuracy": one_vs_rest_accuracy,
        "one_vs_rest_accuracy": one_vs_rest_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "realTrue_TP": tp,
        "fakeTrue_FP": fp,
        "fakeFalse_FN": fn,
        "realFalse_TN": tn,
    }


def summarize_threshold_predictions(
    y_true: list[int],
    y_top1: list[int],
    y_pred_sets: list[list[int]],
    per_class_rows: list[dict],
) -> dict:
    total = len(y_true)
    set_sizes = [len(pred_set) for pred_set in y_pred_sets]
    contains = [truth in pred_set for truth, pred_set in zip(y_true, y_pred_sets)]
    exact_single = [pred_set == [truth] for truth, pred_set in zip(y_true, y_pred_sets)]
    sample_precisions = [(1.0 / len(pred_set)) if hit and pred_set else 0.0 for hit, pred_set in zip(contains, y_pred_sets)]
    sample_recalls = [1.0 if hit else 0.0 for hit in contains]
    sample_f1s = []
    for precision, recall in zip(sample_precisions, sample_recalls):
        sample_f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)

    tp_total = sum(row["realTrue_TP"] for row in per_class_rows)
    fp_total = sum(row["fakeTrue_FP"] for row in per_class_rows)
    fn_total = sum(row["fakeFalse_FN"] for row in per_class_rows)
    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0
    threshold_accuracy = sum(contains) / max(1, total)

    return {
        "accuracy": threshold_accuracy,
        "threshold_accuracy": threshold_accuracy,
        "top1_accuracy": sum(int(truth == pred) for truth, pred in zip(y_true, y_top1)) / max(1, total),
        "threshold_contains_true_rate": threshold_accuracy,
        "threshold_exact_single_label_accuracy": sum(exact_single) / max(1, total),
        "sample_precision": float(np.mean(sample_precisions)) if sample_precisions else 0.0,
        "sample_recall": float(np.mean(sample_recalls)) if sample_recalls else 0.0,
        "sample_f1": float(np.mean(sample_f1s)) if sample_f1s else 0.0,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": float(np.mean([row["precision"] for row in per_class_rows])) if per_class_rows else 0.0,
        "macro_recall": float(np.mean([row["recall"] for row in per_class_rows])) if per_class_rows else 0.0,
        "macro_f1": float(np.mean([row["f1"] for row in per_class_rows])) if per_class_rows else 0.0,
        "average_predicted_labels": float(np.mean(set_sizes)) if set_sizes else 0.0,
        "max_predicted_labels": max(set_sizes, default=0),
        "empty_prediction_count": sum(1 for size in set_sizes if size == 0),
    }


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mapping_name = args.mapping_preset or checkpoint_config_value(ckpt, "mapping_preset", "raw19_identity")
    mapping = load_mapping_preset(mapping_name)
    config = build_eval_config(args, ckpt, mapping)
    topk_fraction = args.topk_fraction
    if topk_fraction is None:
        topk_fraction = checkpoint_config_value(ckpt, "topk_fraction", 0.25)
    min_topk = args.min_topk
    if min_topk is None:
        min_topk = checkpoint_config_value(ckpt, "min_topk", 1)
    max_bags = config.validation_max_batches if args.split == "val" else config.test_max_batches

    set_seed(config.seed)
    device = get_device(args.device)
    eval_dir = next_threshold_eval_dir()
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
                "threshold": args.threshold,
                "ensure_at_least_one": args.ensure_at_least_one,
                "mapping_preset": mapping.name,
                "bag_batch_size": args.bag_batch_size,
                "max_bags": max_bags,
                "topk_fraction": topk_fraction,
                "min_topk": min_topk,
                "class_order": mapping.class_order,
            },
            indent=2,
        )
    )

    logger.log(f"[manifest] resolving {len(config.days)} day folders")
    file_records, _ = build_split_file_records(config, mapping, logger)
    split_records = [rec for rec in file_records if rec.split == args.split]
    eval_records = balanced_file_subset(split_records, max_bags, mapping.class_order, config.seed + (11 if args.split == "val" else 22))
    logger.log(
        f"[{args.split}] using subset bags={len(eval_records)}/{len(split_records)} "
        f"max_bags={max_bags}"
    )
    logger.log(f"[{args.split}] subset_class_counts={json.dumps(class_count_summary(eval_records, mapping.class_order), sort_keys=True)}")

    class_to_index = {name: idx for idx, name in enumerate(mapping.class_order)}
    dataset = CsvBagDataset(eval_records, class_to_index, args.max_windows_per_file, config.seed + 1000)
    loader = DataLoader(dataset, batch_size=args.bag_batch_size, shuffle=False, collate_fn=collate_bags)
    window_classifier = build_classifier(
        strategy=config.classifier_strategy,
        num_classes=len(mapping.class_order),
        logger=logger,
        init_path=None,
    )
    model = TopKMILClassifier(window_classifier, topk_fraction=topk_fraction, min_topk=min_topk).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        logger.log(f"[checkpoint] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
        if missing:
            logger.log(f"[checkpoint] missing_sample={missing[:10]}")
        if unexpected:
            logger.log(f"[checkpoint] unexpected_sample={unexpected[:10]}")
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    y_true: list[int] = []
    y_top1: list[int] = []
    y_pred_sets: list[list[int]] = []
    prediction_rows: list[dict] = []
    with torch.no_grad():
        for batch_idx, (bags, labels, metas) in enumerate(loader, start=1):
            labels = labels.to(device)
            output = model(move_bags_to_device(bags, device))
            logits = output["bag_logits"]
            total_loss += float(criterion(logits, labels).item())
            probs = torch.softmax(logits, dim=1)
            top_conf, top_pred = probs.max(dim=1)

            batch_true = labels.cpu().tolist()
            batch_top_pred = top_pred.cpu().tolist()
            batch_top_conf = top_conf.cpu().tolist()
            batch_probs = probs.cpu()
            y_true.extend(batch_true)
            y_top1.extend(batch_top_pred)

            for row_idx, (meta, truth, top_idx, top_confidence) in enumerate(zip(metas, batch_true, batch_top_pred, batch_top_conf)):
                prob_row = batch_probs[row_idx]
                pred_set = [idx for idx, value in enumerate(prob_row.tolist()) if value >= args.threshold]
                if args.ensure_at_least_one and not pred_set:
                    pred_set = [top_idx]
                pred_set = sorted(pred_set, key=lambda idx: float(prob_row[idx]), reverse=True)
                y_pred_sets.append(pred_set)

                pred_classes = [mapping.class_order[idx] for idx in pred_set]
                pred_probs = [float(prob_row[idx]) for idx in pred_set]
                row = {
                    "file_path": meta["file_path"],
                    "file_name": meta["file_name"],
                    "n_windows": meta["n_windows"],
                    "window_indices": " ".join(str(item) for item in meta["window_indices"]),
                    "true_class": mapping.class_order[truth],
                    "top1_class": mapping.class_order[top_idx],
                    "top1_confidence": top_confidence,
                    "threshold_pred_classes": "|".join(pred_classes),
                    "threshold_pred_probs": "|".join(f"{value:.8f}" for value in pred_probs),
                    "threshold_pred_count": len(pred_set),
                    "contains_true": int(truth in pred_set),
                    "exact_single_label_match": int(pred_set == [truth]),
                    "true_class_prob": float(prob_row[truth]),
                }
                for class_idx, class_name in enumerate(mapping.class_order):
                    row[f"prob_{class_name}"] = float(prob_row[class_idx])
                    row[f"pred_{class_name}"] = int(class_idx in pred_set)
                prediction_rows.append(row)

            if batch_idx == 1 or batch_idx % args.log_every_batches == 0 or batch_idx == len(loader):
                logger.log(f"[{args.split}][batch {batch_idx}/{len(loader)}] bags={len(y_true)}")

    threshold_matrix, threshold_rows = threshold_matrix_and_rows(y_true, y_pred_sets, mapping.class_order)
    top1_matrix, top1_rows = top1_confusion_stats(y_true, y_top1, mapping.class_order)
    threshold_summary = summarize_threshold_predictions(y_true, y_top1, y_pred_sets, threshold_rows)
    threshold_summary.update(
        {
            "loss": total_loss / max(1, len(y_true)),
            "threshold": args.threshold,
            "ensure_at_least_one": args.ensure_at_least_one,
            "evaluated_batches": len(loader),
            "evaluated_bags": len(y_true),
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_global_step": ckpt.get("global_step"),
            "eval_dir": str(eval_dir),
            "split": args.split,
            "matrix_note": "threshold_prediction_matrix rows may sum above support because one CSV can emit multiple predicted labels.",
        }
    )
    top1_summary = {
        "accuracy": threshold_summary["top1_accuracy"],
        "macro_f1": float(np.mean([row["f1"] for row in top1_rows])) if top1_rows else 0.0,
        "evaluated_bags": len(y_true),
    }

    save_json(eval_dir / "config.json", asdict(config))
    save_json(eval_dir / f"{args.split}_threshold_summary.json", threshold_summary)
    save_json(eval_dir / f"{args.split}_threshold_prediction_matrix.json", {"class_order": mapping.class_order, "matrix": threshold_matrix})
    save_json(eval_dir / f"{args.split}_top1_confusion_matrix.json", {"class_order": mapping.class_order, "matrix": top1_matrix})
    save_json(eval_dir / f"{args.split}_top1_summary.json", top1_summary)
    write_csv(eval_dir / f"{args.split}_threshold_per_class_detailed.csv", threshold_rows)
    write_csv(eval_dir / f"{args.split}_top1_per_class_detailed.csv", top1_rows)
    write_csv(eval_dir / f"{args.split}_threshold_predictions.csv", prediction_rows)
    write_csv(eval_dir / "file_manifest.csv", [asdict(rec) for rec in file_records])
    write_csv(eval_dir / f"{args.split}_file_manifest.csv", [asdict(rec) for rec in eval_records])
    write_matrix_csv(eval_dir / f"{args.split}_threshold_prediction_matrix.csv", threshold_matrix, mapping.class_order)
    write_matrix_csv(eval_dir / f"{args.split}_top1_confusion_matrix.csv", top1_matrix, mapping.class_order)
    logger.log(json.dumps(threshold_summary, indent=2))
    logger.log(f"[done] wrote threshold outputs to {eval_dir}")


if __name__ == "__main__":
    main()
