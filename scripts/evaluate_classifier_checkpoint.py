from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

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
from wecfgclassifier.training.loops import evaluate_classifier  # noqa: E402
from wecfgclassifier.utils.runtime import TeeLogger, get_device, resolve_days, save_json, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one classifier checkpoint on a balanced validation subset.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--days", nargs="+", default=["ALL"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mapping-preset", default="raw19_identity")
    parser.add_argument("--classifier-strategy", default="pretrained_cnn_bilstm", choices=["lightweight_pytorch_baseline", "pretrained_cnn_bilstm"])
    parser.add_argument("--classifier-input", default="raw", choices=["raw"])
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--max-files-per-class", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def next_eval_dir() -> Path:
    root = OUTPUT_ROOT / "evaluations"
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in root.glob("eval_*") if p.is_dir())
    next_idx = 1
    if existing:
        next_idx = max(int(p.name.split("_")[-1]) for p in existing) + 1
    path = root / f"eval_{next_idx:03d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def class_count_summary(records: list, class_order: list[str]) -> dict[str, int]:
    counts = Counter(rec.class_index for rec in records)
    return {name: counts.get(index, 0) for index, name in enumerate(class_order)}


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
        classifier_input=args.classifier_input,
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
    eval_records = _balanced_eval_records(split_records, max_records, args.seed + 303)
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
    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate_classifier(
        denoiser=None,
        classifier=classifier,
        loader=loader,
        device=device,
        criterion=criterion,
        class_order=mapping.class_order,
        classifier_input=args.classifier_input,
        config=config,
        logger=logger,
        log_prefix=f"{args.split}",
        log_every_batches=args.log_every_batches,
    )

    save_json(eval_dir / "config.json", asdict(config))
    save_json(eval_dir / f"{args.split}_metrics.json", metrics)
    write_csv(eval_dir / f"{args.split}_per_class.csv", metrics["per_class"])
    write_csv(eval_dir / "file_manifest.csv", [asdict(rec) for rec in file_records])
    write_csv(eval_dir / f"{args.split}_window_manifest.csv", [asdict(rec) for rec in eval_records])
    logger.log(
        json.dumps(
            {
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "evaluated_batches": metrics["evaluated_batches"],
                "evaluated_samples": metrics["evaluated_samples"],
                "eval_dir": str(eval_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
