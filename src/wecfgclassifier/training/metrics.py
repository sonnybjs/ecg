from __future__ import annotations

import numpy as np


def confusion_and_metrics(y_true: list[int], y_pred: list[int], class_order: list[str]) -> dict:
    num_classes = len(class_order)
    conf = np.zeros((num_classes, num_classes), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        conf[truth, pred] += 1
    per_class = []
    f1s = []
    for idx in range(num_classes):
        tp = conf[idx, idx]
        fp = conf[:, idx].sum() - tp
        fn = conf[idx, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = int(conf[idx, :].sum())
        predicted = int(conf[:, idx].sum())
        tn = conf.sum() - tp - fp - fn
        class_accuracy = (tp + tn) / conf.sum() if conf.sum() else 0.0
        per_class.append(
            {
                "class_name": class_order[idx],
                "accuracy": class_accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "predicted_count": predicted,
            }
        )
        f1s.append(f1)
    return {
        "accuracy": float(np.trace(conf) / max(1, conf.sum())),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": conf.tolist(),
        "per_class": per_class,
    }
