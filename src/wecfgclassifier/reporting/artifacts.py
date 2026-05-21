from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..config import ASSUMED_INPUT_FS, DENOISER_FS, ExperimentConfig, FileRecord, LabelMapping, WindowRecord


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_sample_plots(
    exp_dir: Path,
    trainable_denoiser,
    frozen_reference,
    dataset,
    device,
    class_order: list[str],
    n_examples: int = 8,
) -> None:
    import matplotlib.pyplot as plt

    plot_dir = exp_dir / "sample_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    picks = np.linspace(0, max(0, len(dataset) - 1), num=min(n_examples, len(dataset)), dtype=int)
    trainable_denoiser.eval()
    frozen_reference.eval()
    for idx in picks:
        raw, label = dataset[idx]
        raw_b = raw.unsqueeze(0).to(device)
        with torch.no_grad():
            ref = frozen_reference(raw_b)[0].cpu().numpy()
            tuned = trainable_denoiser(raw_b)[0].cpu().numpy()
        raw_np = raw.numpy()
        time_raw = np.arange(len(raw_np)) / ASSUMED_INPUT_FS
        time_den = np.arange(len(ref)) / DENOISER_FS
        plt.figure(figsize=(12, 5))
        plt.plot(time_raw, raw_np, color="0.7", linewidth=1, label="raw")
        plt.plot(time_den, ref, color="#1f77b4", linewidth=1, label="frozen_denoiser")
        plt.plot(time_den, tuned, color="#d62728", linewidth=1, alpha=0.9, label="task_aware_denoiser")
        plt.title(f"window_{idx:04d} | label={class_order[int(label)]}")
        plt.xlabel("seconds")
        plt.ylabel("normalized amplitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"window_{idx:04d}.png", dpi=150)
        plt.close()


def write_report(
    exp_dir: Path,
    config: ExperimentConfig,
    mapping: LabelMapping,
    file_records: list[FileRecord],
    window_records: list[WindowRecord],
    stage1_metrics: dict,
    stage2_metrics: dict | None,
    stage1_test_metrics: dict,
    stage2_test_metrics: dict | None,
    train_log_path: Path,
) -> None:
    class_counts = Counter(rec.mapped_label for rec in file_records)
    split_counts = Counter(rec.split for rec in file_records)
    raw_type_counts = Counter(rec.raw_type for rec in file_records)
    stage1_class_accuracy = {
        item["class_name"]: item["accuracy"]
        for item in stage1_metrics.get("per_class", [])
    }
    lines = [
        "# Task-Aware Denoiser Fine-Tune Report",
        "",
        "## Design",
        "",
        f"- Stage 1: train a PyTorch rhythm classifier using `{config.classifier_input}` input.",
        "- Stage 2: optional task-aware denoiser fine-tune with downstream classification loss.",
        f"- Progress is written to `{train_log_path.name}` and echoed to terminal during training.",
        f"- Label mapping preset: `{mapping.name}`",
        f"- Mapping description: {mapping.description}",
        "",
        "## Dataset Subset",
        "",
        f"- Days: {', '.join(config.days)}",
        f"- Files used: {len(file_records)}",
        f"- Windows used: {len(window_records)}",
        f"- Train files: {split_counts.get('train', 0)}",
        f"- Val files: {split_counts.get('val', 0)}",
        f"- Test files: {split_counts.get('test', 0)}",
        f"- Max files per mapped class: {config.max_files_per_class}",
        f"- Val fraction: {config.val_fraction}",
        f"- Test fraction: {config.test_fraction}",
        "",
        "### File Counts by Mapped Class",
        "",
    ]
    for name in mapping.class_order:
        lines.append(f"- {name}: {class_counts.get(name, 0)} files")
    lines.extend(["", "### Raw Folder Types Retained", ""])
    for raw_type, count in sorted(raw_type_counts.items()):
        lines.append(f"- {raw_type} -> {mapping.class_map.get(raw_type)}: {count} files")
    lines.extend(
        [
            "",
            "## Results",
            "",
            f"### Stage 1 Classifier on `{config.classifier_input}` Input",
            "",
            f"- Val accuracy: {stage1_metrics['accuracy']:.4f}",
            f"- Val macro F1: {stage1_metrics['macro_f1']:.4f}",
            "- Per-class val accuracy: "
            + ", ".join(f"{name}={stage1_class_accuracy.get(name, 0.0):.4f}" for name in mapping.class_order),
            "",
            "### Held-Out Test Set",
            "",
            f"- Stage 1 test accuracy: {stage1_test_metrics['accuracy']:.4f}",
            f"- Stage 1 test macro F1: {stage1_test_metrics['macro_f1']:.4f}",
            "",
            "## Notes",
            "",
            f"- Classifier strategy: `{config.classifier_strategy}`.",
            f"- Classifier input mode: `{config.classifier_input}`.",
            f"- Classifier-only mode: `{config.classifier_only}`.",
            "- This run uses a mapping-driven label set so we can either keep all raw folder types or align them to a pretrained taxonomy.",
        ]
    )
    if not config.classifier_only and stage2_metrics is not None and stage2_test_metrics is not None:
        stage2_class_accuracy = {
            item["class_name"]: item["accuracy"]
            for item in stage2_metrics.get("per_class", [])
        }
        lines[9:9] = [
            "- Preservation losses for the denoiser stage are:",
            f"  - anchor MSE to frozen denoiser output (`lambda_anchor={config.lambda_anchor}`)",
            f"  - slope-consistency L1 to frozen denoiser output (`lambda_slope={config.lambda_slope}`)",
            "",
        ]
        lines.extend(
            [
                "",
                "### Stage 2 Task-Aware Denoiser Fine-Tune",
                "",
                f"- Val accuracy: {stage2_metrics['accuracy']:.4f}",
                f"- Val macro F1: {stage2_metrics['macro_f1']:.4f}",
                "- Per-class val accuracy: "
                + ", ".join(f"{name}={stage2_class_accuracy.get(name, 0.0):.4f}" for name in mapping.class_order),
                f"- Val classification loss: {stage2_metrics['cls_loss']:.4f}",
                f"- Val anchor loss: {stage2_metrics['anchor_loss']:.6f}",
                f"- Val slope loss: {stage2_metrics['slope_loss']:.6f}",
            ]
        )
        lines.insert(lines.index("## Notes"), f"- Stage 2 test macro F1: {stage2_test_metrics['macro_f1']:.4f}")
        lines.insert(lines.index("## Notes"), f"- Stage 2 test accuracy: {stage2_test_metrics['accuracy']:.4f}")
    else:
        lines.extend(["", "### Stage 2 Task-Aware Denoiser Fine-Tune", "", "- Skipped in this classifier-only run."])
    (exp_dir / "REPORT.md").write_text("\n".join(lines))
