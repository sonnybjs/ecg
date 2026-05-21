from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
sys.path.insert(0, str(SRC_DIR))

from train_mil_classifier import (  # noqa: E402
    CsvBagDataset,
    balanced_file_subset,
    build_detection_model,
    collate_bags,
    move_bags_to_device,
)
from wecfgclassifier.config import ExperimentConfig, OUTPUT_ROOT  # noqa: E402
from wecfgclassifier.data.pipeline import build_split_file_records, load_mapping_preset  # noqa: E402
from wecfgclassifier.models.detection_mil import multilabel_candidate_metrics  # noqa: E402
from wecfgclassifier.training.metrics import confusion_and_metrics  # noqa: E402
from wecfgclassifier.utils.runtime import TeeLogger, get_device, save_json, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate global detection MIL thresholds on val/test split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--bag-batch-size", type=int, default=4)
    parser.add_argument("--max-bags", type=int, default=760)
    parser.add_argument("--noise-threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--log-every-batches", type=int, default=50)
    return parser.parse_args()


def next_calibration_dir() -> Path:
    root = OUTPUT_ROOT / "mil_evaluations"
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in root.glob("detection_threshold_calibration_*") if p.is_dir())
    next_idx = max((int(p.name.split("_")[-1]) for p in existing), default=0) + 1
    path = root / f"detection_threshold_calibration_{next_idx:03d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def make_args_for_model(config: ExperimentConfig, ckpt: dict, checkpoint: Path, threshold: float, noise_threshold: float) -> SimpleNamespace:
    model_config = ckpt.get("model_config", {})
    detection_config = model_config.get("detection_mil", {})
    return SimpleNamespace(
        model_type=model_config.get("model_type", "priority_detection_mil"),
        classifier_strategy=config.classifier_strategy,
        init_classifier_path=str(checkpoint),
        detection_head_type=detection_config.get("evidence_head_type", "linear"),
        detection_head_hidden_dim=detection_config.get("evidence_head_hidden_dim", 64),
        detection_head_dropout=detection_config.get("evidence_head_dropout", 0.3),
        init_detection_head_from_classifier=False,
        event_pooling=detection_config.get("event_pooling", "logsumexp"),
        event_lse_tau=detection_config.get("event_lse_tau", 1.0),
        rhythm_pooling=detection_config.get("rhythm_pooling", "topk_mean"),
        conduction_pooling=detection_config.get("conduction_pooling", "topk_attention_mix"),
        attention_mix_alpha=detection_config.get("attention_mix_alpha", 0.5),
        noise_pooling=detection_config.get("noise_pooling", "topk_mean"),
        ce_weight=detection_config.get("ce_weight", 1.0),
        pos_bce_weight=detection_config.get("pos_bce_weight", 0.3),
        group_ce_weight=detection_config.get("group_ce_weight", 0.2),
        noise_weight=detection_config.get("noise_weight", 0.2),
        sparsity_weight=detection_config.get("sparsity_weight", 0.0),
        non_noise_as_weak_negative=detection_config.get("non_noise_as_weak_negative", False),
        noise_weak_negative_weight=detection_config.get("noise_weak_negative_weight", 0.1),
        global_diagnosis_threshold=threshold,
        noise_threshold=noise_threshold,
        top_m_evidence_windows=detection_config.get("top_m_evidence_windows", 3),
        topk_fraction=detection_config.get("default_topk_fraction", 0.25),
        min_topk=detection_config.get("default_min_topk", 1),
    )


def candidate_rows_for_threshold(
    diagnosis_probs: torch.Tensor,
    noise_probs: torch.Tensor,
    softmax_probs: torch.Tensor,
    threshold: float,
    noise_threshold: float,
    class_order: list[str],
) -> tuple[list[list[str]], list[str]]:
    priority_order = [
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
    priority_rank = {name: idx for idx, name in enumerate(priority_order)}
    diagnosis_classes = [name for name in class_order if name != "Noise"]
    candidates_all: list[list[str]] = []
    primary_all: list[str] = []
    for row_idx in range(diagnosis_probs.shape[0]):
        candidates = [
            class_name
            for diag_idx, class_name in enumerate(diagnosis_classes)
            if float(diagnosis_probs[row_idx, diag_idx]) >= threshold
        ]
        noise_flag = float(noise_probs[row_idx]) >= noise_threshold
        if noise_flag:
            candidates.append("Noise")
        diagnosis_candidates = [name for name in candidates if name != "Noise"]
        if diagnosis_candidates:
            primary = min(diagnosis_candidates, key=lambda name: priority_rank.get(name, len(priority_rank)))
        elif noise_flag:
            primary = "Noise"
        else:
            primary = class_order[int(softmax_probs[row_idx].argmax().item())]
            candidates = [primary]
        candidates_all.append(candidates)
        primary_all.append(primary)
    return candidates_all, primary_all


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ExperimentConfig(**ckpt["config"])
    mapping = load_mapping_preset(config.mapping_preset)
    set_seed(config.seed)
    device = get_device(args.device)
    out_dir = next_calibration_dir()
    logger = TeeLogger(out_dir / "calibration.log")
    logger.log(json.dumps({"checkpoint": str(checkpoint), "device": str(device), "split": args.split}, indent=2))

    model_args = make_args_for_model(config, ckpt, checkpoint, threshold=0.3, noise_threshold=args.noise_threshold)
    model = build_detection_model(model_args, mapping, logger).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    logger.log(f"[load] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    model.eval()

    file_records, _ = build_split_file_records(config, mapping, logger)
    split_records_full = [rec for rec in file_records if rec.split == args.split]
    split_records = balanced_file_subset(split_records_full, args.max_bags, mapping.class_order, config.seed + 11)
    class_to_index = {name: idx for idx, name in enumerate(mapping.class_order)}
    dataset = CsvBagDataset(split_records, class_to_index, max_windows_per_file=0, seed=config.seed + 1000)
    loader = DataLoader(dataset, batch_size=args.bag_batch_size, shuffle=False, collate_fn=collate_bags)

    y_true: list[int] = []
    y_softmax: list[int] = []
    diagnosis_prob_rows: list[torch.Tensor] = []
    noise_prob_rows: list[torch.Tensor] = []
    softmax_prob_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_idx, (bags, labels, _) in enumerate(loader, start=1):
            output = model(move_bags_to_device(bags, device), return_evidence=False)
            y_true.extend(labels.tolist())
            y_softmax.extend(output["bag_logits_19"].argmax(dim=1).cpu().tolist())
            diagnosis_prob_rows.append(output["diagnosis_probs"].cpu())
            noise_prob_rows.append(output["noise_prob"].cpu())
            softmax_prob_rows.append(output["primary_softmax_probs"].cpu())
            if batch_idx == 1 or batch_idx % args.log_every_batches == 0 or batch_idx == len(loader):
                logger.log(f"[calibrate][batch {batch_idx}/{len(loader)}] bags={len(y_true)}")

    diagnosis_probs = torch.cat(diagnosis_prob_rows, dim=0)
    noise_probs = torch.cat(noise_prob_rows, dim=0)
    softmax_probs = torch.cat(softmax_prob_rows, dim=0)
    thresholds = args.thresholds or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

    rows = []
    for threshold in thresholds:
        candidate_labels, primary_labels = candidate_rows_for_threshold(
            diagnosis_probs,
            noise_probs,
            softmax_probs,
            threshold,
            args.noise_threshold,
            mapping.class_order,
        )
        metrics = multilabel_candidate_metrics(y_true, candidate_labels, primary_labels, model.mappings)
        rows.append(
            {
                "threshold": threshold,
                "noise_threshold": args.noise_threshold,
                "contains_true_rate": metrics["contains_true_rate"],
                "sample_precision": metrics["sample_precision"],
                "sample_recall": metrics["sample_recall"],
                "sample_f1": metrics["sample_f1"],
                "micro_precision": metrics["micro_precision"],
                "micro_recall": metrics["micro_recall"],
                "micro_f1": metrics["micro_f1"],
                "macro_accuracy": metrics["macro_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "priority_primary_accuracy": metrics["priority_primary_accuracy"],
                "average_predicted_labels": metrics["average_predicted_labels"],
            }
        )

    softmax_metrics = confusion_and_metrics(y_true, y_softmax, mapping.class_order)
    best_macro_f1 = max(rows, key=lambda row: row["macro_f1"])
    best_sample_f1 = max(rows, key=lambda row: row["sample_f1"])
    best_balanced = max(rows, key=lambda row: row["sample_f1"] - 0.03 * abs(row["average_predicted_labels"] - 2.0))
    summary = {
        "checkpoint": str(checkpoint),
        "evaluated_bags": len(y_true),
        "softmax_top1_accuracy": softmax_metrics["accuracy"],
        "softmax_macro_f1": softmax_metrics["macro_f1"],
        "best_macro_f1": best_macro_f1,
        "best_sample_f1": best_sample_f1,
        "best_balanced_target_2_labels": best_balanced,
    }
    with (out_dir / "threshold_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_json(out_dir / "threshold_sweep_summary.json", summary)
    save_json(out_dir / "experiment_config.json", asdict(config))
    logger.log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
