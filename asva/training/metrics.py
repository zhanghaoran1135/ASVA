import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, precision_recall_fscore_support

from asva.data.label_utils import TASK_COLUMNS


def compute_task_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compute_all_metrics(targets, predictions):
    metrics = {task: compute_task_metrics(targets[task], predictions[task]) for task in TASK_COLUMNS}
    averaged = {
        key: float(np.mean([metrics[task][key] for task in TASK_COLUMNS]))
        for key in ["accuracy", "precision", "recall", "f1", "mcc"]
    }
    metrics["overall_average"] = averaged
    return metrics


def save_metrics(metrics, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / f"{prefix}_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "accuracy", "precision", "recall", "f1", "mcc"])
        for task in TASK_COLUMNS:
            row = metrics[task]
            writer.writerow([task, row["accuracy"], row["precision"], row["recall"], row["f1"], row["mcc"]])
        avg = metrics["overall_average"]
        writer.writerow(["overall_average", avg["accuracy"], avg["precision"], avg["recall"], avg["f1"], avg["mcc"]])
