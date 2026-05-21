from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..models.denoiser import compute_classifier_input
from ..utils.runtime import TeeLogger
from .metrics import confusion_and_metrics


def per_class_accuracy_columns(metrics: dict, prefix: str) -> dict:
    columns = {}
    for item in metrics.get("per_class", []):
        class_name = item["class_name"]
        columns[f"{prefix}_{class_name}_accuracy"] = item["accuracy"]
        columns[f"{prefix}_{class_name}_support"] = item["support"]
    return columns


def format_per_class_accuracy(metrics: dict) -> str:
    values = {
        item["class_name"]: round(float(item["accuracy"]), 4)
        for item in metrics.get("per_class", [])
    }
    return json.dumps(values, ensure_ascii=True, sort_keys=True)


def merge_running_and_val_metrics(running_metrics: dict, val_metrics: dict) -> dict:
    merged = dict(val_metrics)
    for key, value in running_metrics.items():
        merged[f"running_{key}" if key in val_metrics else key] = value
    return merged


def _diagnosis_noise_maps(class_order: list[str], noise_class_name: str) -> tuple[int, list[int], dict[int, int]]:
    noise_idx = class_order.index(noise_class_name)
    diagnosis_indices = [idx for idx, _ in enumerate(class_order) if idx != noise_idx]
    original_to_diagnosis = {original_idx: diag_idx for diag_idx, original_idx in enumerate(diagnosis_indices)}
    return noise_idx, diagnosis_indices, original_to_diagnosis


def _is_dual_output(output) -> bool:
    return isinstance(output, dict) and "diagnosis_logits" in output and "noise_logit" in output


def classifier_loss(
    output,
    labels: torch.Tensor,
    criterion: nn.Module,
    class_order: list[str],
    config: ExperimentConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not _is_dual_output(output):
        loss = criterion(output, labels)
        return loss, {"cls_loss": float(loss.detach().item())}

    noise_idx, _, original_to_diagnosis = _diagnosis_noise_maps(class_order, config.noise_class_name)
    diagnosis_logits = output["diagnosis_logits"]
    noise_logit = output["noise_logit"]
    non_noise_mask = labels != noise_idx
    if non_noise_mask.any():
        diagnosis_targets = torch.tensor(
            [original_to_diagnosis[int(label)] for label in labels[non_noise_mask].detach().cpu().tolist()],
            dtype=torch.long,
            device=labels.device,
        )
        diagnosis_loss = F.cross_entropy(diagnosis_logits[non_noise_mask], diagnosis_targets)
    else:
        diagnosis_loss = diagnosis_logits.sum() * 0.0
    noise_targets = (labels == noise_idx).float()
    noise_loss = F.binary_cross_entropy_with_logits(noise_logit, noise_targets)
    loss = diagnosis_loss + config.lambda_noise * noise_loss
    return loss, {
        "diagnosis_loss": float(diagnosis_loss.detach().item()),
        "noise_loss": float(noise_loss.detach().item()),
        "loss": float(loss.detach().item()),
    }


def classifier_predictions(output, class_order: list[str], config: ExperimentConfig) -> torch.Tensor:
    if not _is_dual_output(output):
        return output.argmax(dim=1)

    noise_idx, diagnosis_indices, _ = _diagnosis_noise_maps(class_order, config.noise_class_name)
    diagnosis_logits = output["diagnosis_logits"]
    diagnosis_probs = torch.softmax(diagnosis_logits, dim=1)
    diagnosis_conf, diagnosis_pred = diagnosis_probs.max(dim=1)
    mapped_indices = torch.tensor(diagnosis_indices, dtype=torch.long, device=diagnosis_logits.device)
    pred = mapped_indices[diagnosis_pred]
    noise_prob = torch.sigmoid(output["noise_logit"])
    noise_gate = (noise_prob >= config.noise_threshold) & (diagnosis_conf < config.diagnosis_confidence_threshold)
    pred = pred.clone()
    pred[noise_gate] = noise_idx
    return pred


def add_dual_head_metrics(metrics: dict, y_true: list[int], y_pred: list[int], class_order: list[str], config: ExperimentConfig) -> None:
    if config.classifier_head != "diagnosis_noise":
        return
    noise_idx, diagnosis_indices, _ = _diagnosis_noise_maps(class_order, config.noise_class_name)
    diagnosis_order = [class_order[idx] for idx in diagnosis_indices]
    diagnosis_true = []
    diagnosis_pred = []
    original_to_diag = {original_idx: diag_idx for diag_idx, original_idx in enumerate(diagnosis_indices)}
    for truth, pred in zip(y_true, y_pred):
        if truth == noise_idx:
            continue
        diagnosis_true.append(original_to_diag[truth])
        if pred == noise_idx:
            diagnosis_pred.append(-1)
        else:
            diagnosis_pred.append(original_to_diag.get(pred, -1))
    valid_true = []
    valid_pred = []
    for truth, pred in zip(diagnosis_true, diagnosis_pred):
        valid_true.append(truth)
        valid_pred.append(pred if pred >= 0 else len(diagnosis_order))
    extended_order = diagnosis_order + [config.noise_class_name]
    diagnosis_metrics = confusion_and_metrics(valid_true, valid_pred, extended_order)
    metrics["diagnosis_only_accuracy"] = diagnosis_metrics["accuracy"]
    metrics["diagnosis_only_macro_f1"] = float(
        sum(item["f1"] for item in diagnosis_metrics["per_class"][:-1]) / max(1, len(diagnosis_order))
    )
    metrics["diagnosis_only_per_class"] = diagnosis_metrics["per_class"][:-1]


def save_model_bundle(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict,
    config: ExperimentConfig,
    extra: dict | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metrics": metrics,
        "config": asdict(config),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def save_interval_checkpoint(
    ckpt_dir: Path,
    stage: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    batch_idx: int,
    global_step: int,
    metrics: dict,
    config: ExperimentConfig,
    logger: TeeLogger,
) -> None:
    extra = {
        "stage": stage,
        "batch_idx": batch_idx,
        "global_step": global_step,
        "checkpoint_kind": "batch_interval",
    }
    step_path = ckpt_dir / f"step_{global_step:08d}.ckpt"
    latest_path = ckpt_dir / "interval_latest.ckpt"
    save_model_bundle(step_path, model, optimizer, epoch, metrics, config, extra=extra)
    save_model_bundle(latest_path, model, optimizer, epoch, metrics, config, extra=extra)
    logger.log(f"[{stage}] saved interval checkpoint step={global_step} epoch={epoch} batch={batch_idx}")


def evaluate_classifier(
    denoiser: nn.Module | None,
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    class_order: list[str],
    classifier_input: str,
    config: ExperimentConfig,
    logger: TeeLogger | None = None,
    log_prefix: str = "validation",
    log_every_batches: int = 50,
) -> dict:
    if denoiser is not None:
        denoiser.eval()
    classifier.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    num_batches = len(loader)
    with torch.no_grad():
        for batch_idx, (raw_batch, labels) in enumerate(loader, start=1):
            raw_batch = raw_batch.to(device)
            labels = labels.to(device)
            features = compute_classifier_input(raw_batch, classifier_input, denoiser)
            output = classifier(features)
            loss, _ = classifier_loss(output, labels, criterion, class_order, config)
            total_loss += float(loss.item()) * labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(classifier_predictions(output, class_order, config).cpu().tolist())
            if logger is not None and (
                batch_idx == 1 or batch_idx % log_every_batches == 0 or batch_idx == num_batches
            ):
                logger.log(f"[{log_prefix}][batch {batch_idx}/{num_batches}] samples={len(y_true)}")
    metrics = confusion_and_metrics(y_true, y_pred, class_order)
    metrics["loss"] = total_loss / max(1, len(y_true))
    metrics["evaluated_batches"] = num_batches
    metrics["evaluated_samples"] = len(y_true)
    add_dual_head_metrics(metrics, y_true, y_pred, class_order, config)
    return metrics


def train_classifier_stage(
    frozen_denoiser: nn.Module | None,
    classifier: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    log_every_batches: int,
    checkpoint_every_batches: int,
    logger: TeeLogger,
    class_order: list[str],
    exp_dir: Path,
    config: ExperimentConfig,
    write_csv,
) -> tuple[dict, list[dict]]:
    if frozen_denoiser is not None:
        frozen_denoiser.eval()
        for param in frozen_denoiser.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_metrics = None
    best_score = -1.0
    history: list[dict] = []
    interval_history: list[dict] = []
    ckpt_dir = exp_dir / "checkpoints" / "stage1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(1, epochs + 1):
        classifier.train()
        train_loss = 0.0
        train_samples = 0
        y_true: list[int] = []
        y_pred: list[int] = []
        num_batches = len(train_loader)
        for batch_idx, (raw_batch, labels) in enumerate(train_loader, start=1):
            raw_batch = raw_batch.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                features = compute_classifier_input(raw_batch, config.classifier_input, frozen_denoiser)
            output = classifier(features)
            loss, loss_parts = classifier_loss(output, labels, criterion, class_order, config)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            train_samples += labels.size(0)
            train_loss += float(loss.item()) * labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(classifier_predictions(output, class_order, config).detach().cpu().tolist())
            if batch_idx == 1 or batch_idx % log_every_batches == 0 or batch_idx == num_batches:
                if config.classifier_head == "diagnosis_noise":
                    logger.log(
                        f"[stage1][epoch {epoch}/{epochs}][batch {batch_idx}/{num_batches}] "
                        f"loss={loss.item():.4f} diag={loss_parts['diagnosis_loss']:.4f} noise={loss_parts['noise_loss']:.4f}"
                    )
                else:
                    logger.log(f"[stage1][epoch {epoch}/{epochs}][batch {batch_idx}/{num_batches}] loss={loss.item():.4f}")
            if checkpoint_every_batches > 0 and global_step % checkpoint_every_batches == 0:
                running_metrics = {
                    "running_loss": train_loss / max(1, train_samples),
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "global_step": global_step,
                }
                save_interval_checkpoint(
                    ckpt_dir,
                    "stage1",
                    classifier,
                    optimizer,
                    epoch,
                    batch_idx,
                    global_step,
                    running_metrics,
                    config,
                    logger,
                )
                logger.log(f"[stage1][step {global_step}] running validation")
                val_metrics = evaluate_classifier(
                    frozen_denoiser,
                    classifier,
                    val_loader,
                    device,
                    criterion,
                    class_order,
                    config.classifier_input,
                    config,
                    logger=logger,
                    log_prefix=f"stage1][step {global_step}][validation",
                )
                interval_row = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "running_loss": running_metrics["running_loss"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                }
                if "diagnosis_only_accuracy" in val_metrics:
                    interval_row["val_diagnosis_only_accuracy"] = val_metrics["diagnosis_only_accuracy"]
                    interval_row["val_diagnosis_only_macro_f1"] = val_metrics["diagnosis_only_macro_f1"]
                interval_row.update(per_class_accuracy_columns(val_metrics, "val"))
                interval_history.append(interval_row)
                write_csv(exp_dir / "stage1_interval_history.csv", interval_history)
                save_model_bundle(
                    ckpt_dir / "interval_latest.ckpt",
                    classifier,
                    optimizer,
                    epoch,
                    merge_running_and_val_metrics(running_metrics, val_metrics),
                    config,
                    extra={
                        "stage": "stage1",
                        "batch_idx": batch_idx,
                        "global_step": global_step,
                        "checkpoint_kind": "batch_interval_validation",
                    },
                )
                logger.log(
                    f"[stage1][step {global_step}] val_loss={val_metrics['loss']:.4f} "
                    f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
                )
                logger.log(f"[stage1][step {global_step}] val_class_accuracy={format_per_class_accuracy(val_metrics)}")
                if "diagnosis_only_macro_f1" in val_metrics:
                    logger.log(
                        f"[stage1][step {global_step}] val_diagnosis_only_acc={val_metrics['diagnosis_only_accuracy']:.4f} "
                        f"val_diagnosis_only_f1={val_metrics['diagnosis_only_macro_f1']:.4f}"
                    )
                classifier.train()

        train_metrics = confusion_and_metrics(y_true, y_pred, class_order)
        train_metrics["loss"] = train_loss / max(1, len(train_loader.dataset))
        add_dual_head_metrics(train_metrics, y_true, y_pred, class_order, config)
        val_metrics = evaluate_classifier(
            frozen_denoiser,
            classifier,
            val_loader,
            device,
            criterion,
            class_order,
            config.classifier_input,
            config,
            logger=logger,
            log_prefix=f"stage1][epoch {epoch}][validation",
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        if "diagnosis_only_accuracy" in train_metrics:
            row["train_diagnosis_only_accuracy"] = train_metrics["diagnosis_only_accuracy"]
            row["train_diagnosis_only_macro_f1"] = train_metrics["diagnosis_only_macro_f1"]
        if "diagnosis_only_accuracy" in val_metrics:
            row["val_diagnosis_only_accuracy"] = val_metrics["diagnosis_only_accuracy"]
            row["val_diagnosis_only_macro_f1"] = val_metrics["diagnosis_only_macro_f1"]
        row.update(per_class_accuracy_columns(val_metrics, "val"))
        history.append(row)
        write_csv(exp_dir / "stage1_history.csv", history)
        logger.log(
            f"[stage1][epoch {epoch}/{epochs}] train_loss={row['train_loss']:.4f} "
            f"train_f1={row['train_macro_f1']:.4f} val_loss={row['val_loss']:.4f} val_f1={row['val_macro_f1']:.4f}"
        )
        logger.log(f"[stage1][epoch {epoch}/{epochs}] val_class_accuracy={format_per_class_accuracy(val_metrics)}")
        if "diagnosis_only_macro_f1" in val_metrics:
            logger.log(
                f"[stage1][epoch {epoch}/{epochs}] val_diagnosis_only_acc={val_metrics['diagnosis_only_accuracy']:.4f} "
                f"val_diagnosis_only_f1={val_metrics['diagnosis_only_macro_f1']:.4f}"
            )
        save_model_bundle(
            ckpt_dir / "last.ckpt",
            classifier,
            optimizer,
            epoch,
            val_metrics,
            config,
            extra={"history_rows": len(history), "class_order": class_order},
        )
        torch.save(classifier.state_dict(), ckpt_dir / "last_state_dict.pt")
        score = val_metrics.get("diagnosis_only_macro_f1", val_metrics["macro_f1"])
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in classifier.state_dict().items()}
            best_metrics = val_metrics
            save_model_bundle(
                ckpt_dir / "best.ckpt",
                classifier,
                optimizer,
                epoch,
                val_metrics,
                config,
                extra={"history_rows": len(history), "class_order": class_order},
            )
            torch.save(classifier.state_dict(), ckpt_dir / "best_state_dict.pt")

    assert best_state is not None and best_metrics is not None
    classifier.load_state_dict(best_state)
    return best_metrics, history


def evaluate_task_aware(
    trainable_denoiser: nn.Module,
    frozen_reference: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    lambda_anchor: float,
    lambda_slope: float,
    class_order: list[str],
    logger: TeeLogger | None = None,
    log_prefix: str = "validation",
    log_every_batches: int = 50,
) -> dict:
    trainable_denoiser.eval()
    classifier.eval()
    total_loss = 0.0
    total_cls = 0.0
    total_anchor = 0.0
    total_slope = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    num_batches = len(loader)
    with torch.no_grad():
        for batch_idx, (raw_batch, labels) in enumerate(loader, start=1):
            raw_batch = raw_batch.to(device)
            labels = labels.to(device)
            den = trainable_denoiser(raw_batch)
            ref = frozen_reference(raw_batch)
            logits = classifier(den)
            cls_loss = criterion(logits, labels)
            anchor_loss = F.mse_loss(den, ref)
            slope_loss = F.l1_loss(den[:, 1:] - den[:, :-1], ref[:, 1:] - ref[:, :-1])
            loss = cls_loss + lambda_anchor * anchor_loss + lambda_slope * slope_loss
            bs = labels.size(0)
            total_loss += float(loss.item()) * bs
            total_cls += float(cls_loss.item()) * bs
            total_anchor += float(anchor_loss.item()) * bs
            total_slope += float(slope_loss.item()) * bs
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            if logger is not None and (
                batch_idx == 1 or batch_idx % log_every_batches == 0 or batch_idx == num_batches
            ):
                logger.log(f"[{log_prefix}][batch {batch_idx}/{num_batches}] samples={len(y_true)}")

    metrics = confusion_and_metrics(y_true, y_pred, class_order)
    denom = max(1, len(y_true))
    metrics.update(
        {
            "loss": total_loss / denom,
            "cls_loss": total_cls / denom,
            "anchor_loss": total_anchor / denom,
            "slope_loss": total_slope / denom,
            "evaluated_batches": num_batches,
            "evaluated_samples": len(y_true),
        }
    )
    return metrics


def finetune_denoiser_stage(
    trainable_denoiser: nn.Module,
    frozen_reference: nn.Module,
    classifier: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    lambda_anchor: float,
    lambda_slope: float,
    log_every_batches: int,
    checkpoint_every_batches: int,
    logger: TeeLogger,
    class_order: list[str],
    exp_dir: Path,
    config: ExperimentConfig,
    write_csv,
) -> tuple[dict, list[dict]]:
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    frozen_reference.eval()
    for param in frozen_reference.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(trainable_denoiser.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_metrics = None
    best_score = -1.0
    history: list[dict] = []
    interval_history: list[dict] = []
    ckpt_dir = exp_dir / "checkpoints" / "stage2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(1, epochs + 1):
        trainable_denoiser.train()
        total_loss = 0.0
        total_cls = 0.0
        total_anchor = 0.0
        total_slope = 0.0
        train_samples = 0
        y_true: list[int] = []
        y_pred: list[int] = []
        num_batches = len(train_loader)
        for batch_idx, (raw_batch, labels) in enumerate(train_loader, start=1):
            raw_batch = raw_batch.to(device)
            labels = labels.to(device)
            den = trainable_denoiser(raw_batch)
            with torch.no_grad():
                ref = frozen_reference(raw_batch)
            logits = classifier(den)
            cls_loss = criterion(logits, labels)
            anchor_loss = F.mse_loss(den, ref)
            slope_loss = F.l1_loss(den[:, 1:] - den[:, :-1], ref[:, 1:] - ref[:, :-1])
            loss = cls_loss + lambda_anchor * anchor_loss + lambda_slope * slope_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            bs = labels.size(0)
            train_samples += bs
            total_loss += float(loss.item()) * bs
            total_cls += float(cls_loss.item()) * bs
            total_anchor += float(anchor_loss.item()) * bs
            total_slope += float(slope_loss.item()) * bs
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
            if batch_idx == 1 or batch_idx % log_every_batches == 0 or batch_idx == num_batches:
                logger.log(
                    f"[stage2][epoch {epoch}/{epochs}][batch {batch_idx}/{num_batches}] "
                    f"loss={loss.item():.4f} cls={cls_loss.item():.4f} "
                    f"anchor={anchor_loss.item():.6f} slope={slope_loss.item():.6f}"
                )
            if checkpoint_every_batches > 0 and global_step % checkpoint_every_batches == 0:
                running_metrics = {
                    "running_loss": total_loss / max(1, train_samples),
                    "running_cls_loss": total_cls / max(1, train_samples),
                    "running_anchor_loss": total_anchor / max(1, train_samples),
                    "running_slope_loss": total_slope / max(1, train_samples),
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "global_step": global_step,
                }
                save_interval_checkpoint(
                    ckpt_dir,
                    "stage2",
                    trainable_denoiser,
                    optimizer,
                    epoch,
                    batch_idx,
                    global_step,
                    running_metrics,
                    config,
                    logger,
                )
                logger.log(f"[stage2][step {global_step}] running validation")
                val_metrics = evaluate_task_aware(
                    trainable_denoiser,
                    frozen_reference,
                    classifier,
                    val_loader,
                    device,
                    criterion,
                    lambda_anchor,
                    lambda_slope,
                    class_order,
                    logger=logger,
                    log_prefix=f"stage2][step {global_step}][validation",
                )
                interval_row = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "running_loss": running_metrics["running_loss"],
                    "running_cls_loss": running_metrics["running_cls_loss"],
                    "running_anchor_loss": running_metrics["running_anchor_loss"],
                    "running_slope_loss": running_metrics["running_slope_loss"],
                    "val_loss": val_metrics["loss"],
                    "val_cls_loss": val_metrics["cls_loss"],
                    "val_anchor_loss": val_metrics["anchor_loss"],
                    "val_slope_loss": val_metrics["slope_loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                }
                interval_row.update(per_class_accuracy_columns(val_metrics, "val"))
                interval_history.append(interval_row)
                write_csv(exp_dir / "stage2_interval_history.csv", interval_history)
                save_model_bundle(
                    ckpt_dir / "interval_latest.ckpt",
                    trainable_denoiser,
                    optimizer,
                    epoch,
                    merge_running_and_val_metrics(running_metrics, val_metrics),
                    config,
                    extra={
                        "stage": "stage2",
                        "batch_idx": batch_idx,
                        "global_step": global_step,
                        "checkpoint_kind": "batch_interval_validation",
                    },
                )
                logger.log(
                    f"[stage2][step {global_step}] val_loss={val_metrics['loss']:.4f} "
                    f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
                )
                logger.log(f"[stage2][step {global_step}] val_class_accuracy={format_per_class_accuracy(val_metrics)}")
                trainable_denoiser.train()

        train_metrics = confusion_and_metrics(y_true, y_pred, class_order)
        denom = max(1, len(train_loader.dataset))
        train_metrics.update(
            {
                "loss": total_loss / denom,
                "cls_loss": total_cls / denom,
                "anchor_loss": total_anchor / denom,
                "slope_loss": total_slope / denom,
            }
        )
        val_metrics = evaluate_task_aware(
            trainable_denoiser, frozen_reference, classifier, val_loader, device, criterion,
            lambda_anchor, lambda_slope, class_order,
            logger=logger,
            log_prefix=f"stage2][epoch {epoch}][validation",
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_cls_loss": train_metrics["cls_loss"],
            "train_anchor_loss": train_metrics["anchor_loss"],
            "train_slope_loss": train_metrics["slope_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_cls_loss": val_metrics["cls_loss"],
            "val_anchor_loss": val_metrics["anchor_loss"],
            "val_slope_loss": val_metrics["slope_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        row.update(per_class_accuracy_columns(val_metrics, "val"))
        history.append(row)
        write_csv(exp_dir / "stage2_history.csv", history)
        logger.log(
            f"[stage2][epoch {epoch}/{epochs}] train_loss={row['train_loss']:.4f} "
            f"train_f1={row['train_macro_f1']:.4f} val_loss={row['val_loss']:.4f} val_f1={row['val_macro_f1']:.4f}"
        )
        logger.log(f"[stage2][epoch {epoch}/{epochs}] val_class_accuracy={format_per_class_accuracy(val_metrics)}")
        save_model_bundle(
            ckpt_dir / "last.ckpt",
            trainable_denoiser,
            optimizer,
            epoch,
            val_metrics,
            config,
            extra={"history_rows": len(history), "class_order": class_order},
        )
        torch.save(trainable_denoiser.state_dict(), ckpt_dir / "last_state_dict.pt")
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu() for k, v in trainable_denoiser.state_dict().items()}
            best_metrics = val_metrics
            save_model_bundle(
                ckpt_dir / "best.ckpt",
                trainable_denoiser,
                optimizer,
                epoch,
                val_metrics,
                config,
                extra={"history_rows": len(history), "class_order": class_order},
            )
            torch.save(trainable_denoiser.state_dict(), ckpt_dir / "best_state_dict.pt")

    assert best_state is not None and best_metrics is not None
    trainable_denoiser.load_state_dict(best_state)
    return best_metrics, history
