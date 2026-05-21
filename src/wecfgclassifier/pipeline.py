from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from .config import ASSUMED_INPUT_FS, DENOISER_FS, OUTPUT_ROOT, WINDOW_SEC, ExperimentConfig
from .data.pipeline import FileWindowBatchSampler, WindowDataset, build_split_file_records, load_mapping_preset
from .models.backbones import build_classifier
from .models.denoiser import GRUDenoiserWrapper, maybe_load_denoiser_weights
from .reporting.artifacts import save_sample_plots, write_csv, write_report
from .training.loops import evaluate_classifier, evaluate_task_aware, finetune_denoiser_stage, train_classifier_stage
from .utils.runtime import TeeLogger, get_device, next_experiment_dir, resolve_days, save_json, set_seed


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    mapping = load_mapping_preset(args.mapping_preset)
    resolved_days = resolve_days(args.days)
    return ExperimentConfig(
        days=resolved_days,
        seed=args.seed,
        classifier_strategy=args.classifier_strategy,
        classifier_head=args.classifier_head,
        classifier_input=args.classifier_input,
        classifier_only=args.classifier_only,
        mapping_preset=mapping.name,
        mapping_description=mapping.description,
        class_order=mapping.class_order,
        max_files_per_class=args.max_files_per_class,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        classifier_epochs=args.classifier_epochs,
        denoiser_epochs=args.denoiser_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        file_batch_sampler=not args.no_file_batch_sampler,
        log_every_batches=args.log_every_batches,
        checkpoint_every_batches=args.checkpoint_every_batches,
        validation_max_batches=args.validation_max_batches,
        test_max_batches=args.test_max_batches,
        learning_rate_classifier=args.learning_rate_classifier,
        learning_rate_denoiser=args.learning_rate_denoiser,
        lambda_anchor=args.lambda_anchor,
        lambda_slope=args.lambda_slope,
        lambda_noise=args.lambda_noise,
        noise_class_name=args.noise_class_name,
        noise_threshold=args.noise_threshold,
        diagnosis_confidence_threshold=args.diagnosis_confidence_threshold,
        window_sec=WINDOW_SEC,
        input_fs=ASSUMED_INPUT_FS,
        denoiser_fs=DENOISER_FS,
        init_classifier_path=args.init_classifier_path,
        init_denoiser_path=args.init_denoiser_path,
    )


def _loader_kwargs(num_workers: int, device: torch.device) -> dict:
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def _per_class_accuracy(metrics: dict) -> dict[str, float]:
    return {
        item["class_name"]: item["accuracy"]
        for item in metrics.get("per_class", [])
    }


def _balanced_eval_records(records: list, max_records: int, seed: int) -> list:
    if max_records <= 0 or len(records) <= max_records:
        return records
    by_class = defaultdict(list)
    for rec in records:
        by_class[rec.class_index].append(rec)
    rng = random.Random(seed)
    for class_records in by_class.values():
        rng.shuffle(class_records)

    selected = []
    class_ids = sorted(by_class)
    while len(selected) < max_records and class_ids:
        next_class_ids = []
        for class_id in class_ids:
            class_records = by_class[class_id]
            if class_records and len(selected) < max_records:
                selected.append(class_records.pop())
            if class_records:
                next_class_ids.append(class_id)
        class_ids = next_class_ids
    rng.shuffle(selected)
    return selected


def _class_count_summary(records: list, class_order: list[str]) -> dict[str, int]:
    counts = Counter(rec.class_index for rec in records)
    return {name: counts.get(index, 0) for index, name in enumerate(class_order)}


def build_dataloaders(
    config: ExperimentConfig,
    logger: TeeLogger,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader, list, list]:
    mapping = load_mapping_preset(config.mapping_preset)
    logger.log(f"[manifest] resolving {len(config.days)} day folders")
    file_records, window_records = build_split_file_records(config, mapping, logger)
    train_records = [rec for rec in window_records if rec.split == "train"]
    val_records = [rec for rec in window_records if rec.split == "val"]
    test_records = [rec for rec in window_records if rec.split == "test"]
    full_val_count = len(val_records)
    full_test_count = len(test_records)
    val_records = _balanced_eval_records(
        val_records,
        config.validation_max_batches * config.batch_size if config.validation_max_batches > 0 else 0,
        config.seed + 101,
    )
    test_records = _balanced_eval_records(
        test_records,
        config.test_max_batches * config.batch_size if config.test_max_batches > 0 else 0,
        config.seed + 202,
    )
    if len(val_records) != full_val_count:
        logger.log(
            f"[validation] using class-balanced subset windows={len(val_records)}/{full_val_count} "
            f"max_batches={config.validation_max_batches}"
        )
        logger.log(f"[validation] subset_class_counts={json.dumps(_class_count_summary(val_records, mapping.class_order), sort_keys=True)}")
    if len(test_records) != full_test_count:
        logger.log(
            f"[test] using class-balanced subset windows={len(test_records)}/{full_test_count} "
            f"max_batches={config.test_max_batches}"
        )
        logger.log(f"[test] subset_class_counts={json.dumps(_class_count_summary(test_records, mapping.class_order), sort_keys=True)}")

    train_dataset = WindowDataset(train_records)
    val_dataset = WindowDataset(val_records)
    test_dataset = WindowDataset(test_records)
    kwargs = _loader_kwargs(config.num_workers, device)
    if config.file_batch_sampler:
        train_sampler = FileWindowBatchSampler(train_records, batch_size=config.batch_size, seed=config.seed)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **kwargs)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader, file_records, window_records


def run_experiment(args: argparse.Namespace) -> None:
    config = build_config(args)
    mapping = load_mapping_preset(config.mapping_preset)
    set_seed(config.seed)
    device = get_device(args.device)
    exp_dir = next_experiment_dir(OUTPUT_ROOT)
    logger = TeeLogger(exp_dir / "train.log")

    train_loader, val_loader, test_loader, file_records, window_records = build_dataloaders(config, logger, device)
    logger.log(
        json.dumps(
            {
                "exp_dir": str(exp_dir),
                "device": str(device),
                "train_windows": len(train_loader.dataset),
                "val_windows": len(val_loader.dataset),
                "test_windows": len(test_loader.dataset),
                "full_val_windows": sum(1 for rec in window_records if rec.split == "val"),
                "full_test_windows": sum(1 for rec in window_records if rec.split == "test"),
                "mapping_preset": mapping.name,
                "classifier_input": config.classifier_input,
                "classifier_head": config.classifier_head,
                "classifier_only": config.classifier_only,
                "batch_size": config.batch_size,
                "num_workers": config.num_workers,
                "file_batch_sampler": config.file_batch_sampler,
                "checkpoint_every_batches": config.checkpoint_every_batches,
                "validation_max_batches": config.validation_max_batches,
                "test_max_batches": config.test_max_batches,
                "class_order": mapping.class_order,
            },
            indent=2,
        )
    )

    classifier = build_classifier(
        strategy=config.classifier_strategy,
        num_classes=len(mapping.class_order),
        logger=logger,
        init_path=config.init_classifier_path,
        classifier_head=config.classifier_head,
        noise_class_name=config.noise_class_name,
        class_order=mapping.class_order,
    ).to(device)

    frozen_denoiser = None
    frozen_reference = None
    trainable_denoiser = None
    if config.classifier_input == "frozen_denoiser" or not config.classifier_only:
        frozen_denoiser = GRUDenoiserWrapper().to(device)
    if not config.classifier_only:
        frozen_reference = GRUDenoiserWrapper().to(device)
        trainable_denoiser = GRUDenoiserWrapper().to(device)
        maybe_load_denoiser_weights(trainable_denoiser, config.init_denoiser_path, logger)

    stage1_metrics, stage1_history = train_classifier_stage(
        frozen_denoiser=frozen_denoiser,
        classifier=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=config.classifier_epochs,
        learning_rate=config.learning_rate_classifier,
        log_every_batches=config.log_every_batches,
        checkpoint_every_batches=config.checkpoint_every_batches,
        logger=logger,
        class_order=mapping.class_order,
        exp_dir=exp_dir,
        config=config,
        write_csv=write_csv,
    )

    stage2_metrics = None
    stage2_history: list[dict] = []
    if not config.classifier_only:
        if config.classifier_head != "single":
            raise NotImplementedError("Task-aware denoiser fine-tune currently expects --classifier-head single.")
        assert trainable_denoiser is not None and frozen_reference is not None
        stage2_metrics, stage2_history = finetune_denoiser_stage(
            trainable_denoiser=trainable_denoiser,
            frozen_reference=frozen_reference,
            classifier=classifier,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=config.denoiser_epochs,
            learning_rate=config.learning_rate_denoiser,
            lambda_anchor=config.lambda_anchor,
            lambda_slope=config.lambda_slope,
            log_every_batches=config.log_every_batches,
            checkpoint_every_batches=config.checkpoint_every_batches,
            logger=logger,
            class_order=mapping.class_order,
            exp_dir=exp_dir,
            config=config,
            write_csv=write_csv,
        )

    criterion = torch.nn.CrossEntropyLoss()
    stage1_test_metrics = evaluate_classifier(
        frozen_denoiser,
        classifier,
        test_loader,
        device,
        criterion,
        mapping.class_order,
        config.classifier_input,
        config,
        logger=logger,
        log_prefix="stage1][test",
    )
    stage2_test_metrics = None
    if not config.classifier_only:
        assert trainable_denoiser is not None and frozen_reference is not None
        stage2_test_metrics = evaluate_task_aware(
            trainable_denoiser,
            frozen_reference,
            classifier,
            test_loader,
            device,
            criterion,
            config.lambda_anchor,
            config.lambda_slope,
            mapping.class_order,
            logger=logger,
            log_prefix="stage2][test",
        )

    write_csv(exp_dir / "file_manifest.csv", [asdict(rec) for rec in file_records])
    write_csv(exp_dir / "window_manifest.csv", [asdict(rec) for rec in window_records])
    write_csv(exp_dir / "stage1_history.csv", stage1_history)
    if stage2_history:
        write_csv(exp_dir / "stage2_history.csv", stage2_history)
    save_json(exp_dir / "experiment_config.json", asdict(config))
    save_json(exp_dir / "stage1_best_metrics.json", stage1_metrics)
    if stage2_metrics is not None:
        save_json(exp_dir / "stage2_best_metrics.json", stage2_metrics)
    save_json(exp_dir / "stage1_test_metrics.json", stage1_test_metrics)
    if stage2_test_metrics is not None:
        save_json(exp_dir / "stage2_test_metrics.json", stage2_test_metrics)
    torch.save(classifier.state_dict(), exp_dir / "classifier_stage1_best.pt")
    if trainable_denoiser is not None:
        torch.save(trainable_denoiser.state_dict(), exp_dir / "denoiser_stage2_best.pt")

    write_report(
        exp_dir,
        config,
        mapping,
        file_records,
        window_records,
        stage1_metrics,
        stage2_metrics,
        stage1_test_metrics,
        stage2_test_metrics,
        exp_dir / "train.log",
    )
    if not args.skip_plots and not config.classifier_only:
        try:
            assert trainable_denoiser is not None and frozen_reference is not None
            save_sample_plots(exp_dir, trainable_denoiser, frozen_reference, val_loader.dataset, device, mapping.class_order)
        except Exception as exc:
            (exp_dir / "plot_error.txt").write_text(str(exc))

    summary = {
        "exp_dir": str(exp_dir),
        "device": str(device),
        "train_windows": len(train_loader.dataset),
        "val_windows": len(val_loader.dataset),
        "test_windows": len(test_loader.dataset),
        "stage1_val_accuracy": stage1_metrics["accuracy"],
        "stage1_val_macro_f1": stage1_metrics["macro_f1"],
        "stage1_val_per_class_accuracy": _per_class_accuracy(stage1_metrics),
        "stage1_test_accuracy": stage1_test_metrics["accuracy"],
        "stage1_test_macro_f1": stage1_test_metrics["macro_f1"],
    }
    if stage2_metrics is not None and stage2_test_metrics is not None:
        summary["stage2_val_accuracy"] = stage2_metrics["accuracy"]
        summary["stage2_val_macro_f1"] = stage2_metrics["macro_f1"]
        summary["stage2_val_per_class_accuracy"] = _per_class_accuracy(stage2_metrics)
        summary["stage2_test_accuracy"] = stage2_test_metrics["accuracy"]
        summary["stage2_test_macro_f1"] = stage2_test_metrics["macro_f1"]
    logger.log(json.dumps(summary, indent=2))
