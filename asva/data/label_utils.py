import logging
import math
from pathlib import Path

from .cache_utils import dump_json, load_json

LOGGER = logging.getLogger(__name__)

TASK_COLUMNS = [
    "cvss2_AV",
    "cvss2_AC",
    "cvss2_AU",
    "cvss2_C",
    "cvss2_I",
    "cvss2_A",
    "cvss2_severity",
]

EXPECTED_LABELS = {
    "cvss2_AV": ["A", "L", "N"],
    "cvss2_AC": ["H", "L", "M"],
    "cvss2_AU": ["M", "N", "S"],
    "cvss2_C": ["C", "N", "P"],
    "cvss2_I": ["C", "N", "P"],
    "cvss2_A": ["C", "N", "P"],
    "cvss2_severity": ["HIGH", "LOW", "MEDIUM"],
}


class LabelEncoderBundle:
    def __init__(self, label_to_id, id_to_label):
        self.label_to_id = label_to_id
        self.id_to_label = id_to_label

    def encode_row(self, row):
        return {task: self.label_to_id[task][str(row[task]).strip()] for task in TASK_COLUMNS}

    def decode_task(self, task, values):
        mapping = self.id_to_label[task]
        return [mapping[int(value)] for value in values]

    def to_dict(self):
        return {
            "label_to_id": self.label_to_id,
            "id_to_label": {task: {str(k): v for k, v in mapping.items()} for task, mapping in self.id_to_label.items()},
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            label_to_id={task: {str(k): int(v) for k, v in mapping.items()} for task, mapping in data["label_to_id"].items()},
            id_to_label={task: {int(k): str(v) for k, v in mapping.items()} for task, mapping in data["id_to_label"].items()},
        )


def build_label_encoders(records):
    label_to_id = {}
    id_to_label = {}
    for task in TASK_COLUMNS:
        values = sorted({str(record[task]).strip() for record in records if str(record.get(task, "")).strip()})
        expected = EXPECTED_LABELS.get(task, [])
        unexpected = sorted(set(values) - set(expected))
        if unexpected:
            LOGGER.warning("Task %s contains unexpected labels: %s", task, unexpected)
        if expected and set(values).issubset(set(expected)):
            ordered = [label for label in expected if label in values]
        else:
            ordered = values
        mapping = {label: idx for idx, label in enumerate(ordered)}
        label_to_id[task] = mapping
        id_to_label[task] = {idx: label for label, idx in mapping.items()}
    return LabelEncoderBundle(label_to_id=label_to_id, id_to_label=id_to_label)


def save_label_encoders(bundle, path):
    dump_json(bundle.to_dict(), path)


def load_label_encoders(path):
    return LabelEncoderBundle.from_dict(load_json(path))


def compute_balanced_class_weights(
    records,
    task_dims,
    power=0.5,
    min_weight=0.5,
    max_weight=4.0,
):
    weights = {}
    for task in TASK_COLUMNS:
        counts = [0 for _ in range(task_dims[task])]
        for record in records:
            labels = record.get("labels", {})
            if task not in labels:
                continue
            label_id = int(labels[task])
            if 0 <= label_id < len(counts):
                counts[label_id] += 1
        total = sum(counts)
        raw = []
        for count in counts:
            adjusted = max(count, 1)
            raw.append((total / adjusted) ** power if total > 0 else 1.0)
        mean_weight = sum(raw) / max(len(raw), 1)
        normalized = [min(max(weight / max(mean_weight, 1e-8), min_weight), max_weight) for weight in raw]
        weights[task] = [float(weight) for weight in normalized]
        LOGGER.info("Task %s class weights=%s counts=%s", task, normalized, counts)
    return weights


def compute_multitask_sample_weights(
    records,
    class_weights,
    tasks=None,
    power=1.0,
    min_weight=0.2,
    max_weight=5.0,
):
    selected_tasks = tasks or TASK_COLUMNS
    sample_weights = []
    for record in records:
        labels = record.get("labels", {})
        weights = []
        for task in selected_tasks:
            if task not in labels or task not in class_weights:
                continue
            label_id = int(labels[task])
            task_weights = class_weights[task]
            if 0 <= label_id < len(task_weights):
                weights.append(float(task_weights[label_id]))
        if not weights:
            sample_weights.append(1.0)
            continue
        aggregate = sum(weights) / float(len(weights))
        sample_weights.append(float(min(max(aggregate**power, min_weight), max_weight)))
    return sample_weights
