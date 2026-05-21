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

from wecfgclassifier.config import ASSUMED_INPUT_FS, DENOISER_FS, OUTPUT_ROOT, WINDOW_SEC, ExperimentConfig  # noqa: E402
from wecfgclassifier.data.pipeline import WindowDataset, build_split_file_records, load_mapping_preset  # noqa: E402
from wecfgclassifier.models.backbones import build_classifier  # noqa: E402
from wecfgclassifier.pipeline import _balanced_eval_records  # noqa: E402
from wecfgclassifier.reporting.artifacts import write_csv  # noqa: E402
from wecfgclassifier.utils.runtime import TeeLogger, get_device, resolve_days, save_json, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detailed validation for one classifier checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--days", nargs="+", default=["ALL"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mapping-preset", default="raw19_identity")
    parser.add_argument("--classifier-strategy", default="pretrained_cnn_bilstm", choices=["lightweight_pytorch_baseline", "pretrained_cnn_bilstm"])
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=400)
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--max-files-per-class", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def next_eval_dir() -> Path:
    root = OUTPUT_ROOT / "evaluations"
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in root.glob("detailed_eval_*") if p.is_dir())
    next_idx = 1
    if existing:
        next_idx = max(int(p.name.split("_")[-1]) for p in existing) + 1
    path = root / f"detailed_eval_{next_idx:03d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def class_count_summary(records: list, class_order: list[str]) -> dict[str, int]:
    counts = Counter(rec.class_index for rec in records)
    return {name: counts.get(index, 0) for index, name in enumerate(class_order)}


def confusion_stats(y_true: list[int], y_pred: list[int], class_order: list[str]) -> tuple[list[list[int]], list[dict]]:
    num_classes = len(class_order)
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        conf[truth, pred] += 1

    total = int(conf.sum())
    rows: list[dict] = []
    for idx, name in enumerate(class_order):
        tp = int(conf[idx, idx])
        fp = int(conf[:, idx].sum() - tp)
        fn = int(conf[idx, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        support = tp + fn
        predicted = tp + fp
        one_vs_rest_accuracy = (tp + tn) / total if total else 0.0
        class_accuracy = one_vs_rest_accuracy
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "class_name": name,
                "support": support,
                "predicted_count": predicted,
                "accuracy": class_accuracy,
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
        )
    return conf.tolist(), rows


def write_matrix_csv(path: Path, matrix: list[list[int]], class_order: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth\\pred", *class_order])
        for name, row in zip(class_order, matrix):
            writer.writerow([name, *row])


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    mapping = load_mapping_preset(args.mapping_preset)
    config = ExperimentConfig(
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
        classifier_epochs=0,
        denoiser_epochs=0,
        batch_size=args.batch_size,
        num_workers=0,
        file_batch_sampler=False,
        log_every_batches=args.log_every_batches,
        checkpoint_every_batches=0,
        validation_max_batches=args.max_batches if args.split == "val" else 0,
        test_max_batches=args.max_batches if args.split == "test" else 0,
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
        init_classifier_path=str(checkpoint),
        init_denoiser_path=None,
    )

    set_seed(config.seed)
    device = get_device(args.device)
    eval_dir = next_eval_dir()
    logger = TeeLogger(eval_dir / "eval.log")
    logger.log(
        json.dumps(
            {
                "eval_dir": str(eval_dir),
                "checkpoint": str(checkpoint),
                "device": str(device),
                "split": args.split,
                "mapping_preset": mapping.name,
                "batch_size": args.batch_size,
                "max_batches": args.max_batches,
                "class_order": mapping.class_order,
            },
            indent=2,
        )
    )

    logger.log(f"[manifest] resolving {len(config.days)} day folders")
    file_records, window_records = build_split_file_records(config, mapping, logger)
    split_records = [rec for rec in window_records if rec.split == args.split]
    full_count = len(split_records)
    max_records = args.max_batches * args.batch_size if args.max_batches > 0 else 0
    eval_records = _balanced_eval_records(split_records, max_records, args.seed + 404)
    logger.log(
        f"[{args.split}] using class-balanced subset windows={len(eval_records)}/{full_count} "
        f"max_batches={args.max_batches}"
    )
    logger.log(f"[{args.split}] subset_class_counts={json.dumps(class_count_summary(eval_records, mapping.class_order), sort_keys=True)}")

    dataset = WindowDataset(eval_records)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    classifier = build_classifier(
        strategy=args.classifier_strategy,
        num_classes=len(mapping.class_order),
        logger=logger,
        init_path=str(checkpoint),
    ).to(device)
    classifier.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict] = []
    with torch.no_grad():
        for batch_idx, (raw_batch, labels) in enumerate(loader, start=1):
            raw_batch = raw_batch.to(device)
            labels = labels.to(device)
            logits = classifier(raw_batch)
            total_loss += float(criterion(logits, labels).item())
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            batch_true = labels.cpu().tolist()
            batch_pred = pred.cpu().tolist()
            batch_conf = conf.cpu().tolist()
            y_true.extend(batch_true)
            y_pred.extend(batch_pred)
            start = (batch_idx - 1) * args.batch_size
            for offset, (truth, pred_idx, confidence) in enumerate(zip(batch_true, batch_pred, batch_conf)):
                rec = eval_records[start + offset]
                prediction_rows.append(
                    {
                        "index": start + offset,
                        "file_path": rec.file_path,
                        "file_name": rec.file_name,
                        "window_index": rec.window_index,
                        "true_class": mapping.class_order[truth],
                        "pred_class": mapping.class_order[pred_idx],
                        "correct": int(truth == pred_idx),
                        "confidence": confidence,
                    }
                )
            if batch_idx == 1 or batch_idx % args.log_every_batches == 0 or batch_idx == len(loader):
                logger.log(f"[{args.split}][batch {batch_idx}/{len(loader)}] samples={len(y_true)}")

    confusion, per_class_rows = confusion_stats(y_true, y_pred, mapping.class_order)
    overall_accuracy = sum(int(t == p) for t, p in zip(y_true, y_pred)) / max(1, len(y_true))
    macro_f1 = sum(row["f1"] for row in per_class_rows) / max(1, len(per_class_rows))
    summary = {
        "loss": total_loss / max(1, len(y_true)),
        "accuracy": overall_accuracy,
        "macro_f1": macro_f1,
        "evaluated_batches": len(loader),
        "evaluated_samples": len(y_true),
        "eval_dir": str(eval_dir),
    }

    save_json(eval_dir / "config.json", asdict(config))
    save_json(eval_dir / f"{args.split}_summary.json", summary)
    save_json(eval_dir / f"{args.split}_confusion_matrix.json", {"class_order": mapping.class_order, "matrix": confusion})
    write_csv(eval_dir / f"{args.split}_per_class_detailed.csv", per_class_rows)
    write_csv(eval_dir / f"{args.split}_predictions.csv", prediction_rows)
    write_csv(eval_dir / "file_manifest.csv", [asdict(rec) for rec in file_records])
    write_csv(eval_dir / f"{args.split}_window_manifest.csv", [asdict(rec) for rec in eval_records])
    write_matrix_csv(eval_dir / f"{args.split}_confusion_matrix.csv", confusion, mapping.class_order)
    logger.log(json.dumps(summary, indent=2))
    logger.log(f"[done] wrote detailed outputs to {eval_dir}")


if __name__ == "__main__":
    main()
