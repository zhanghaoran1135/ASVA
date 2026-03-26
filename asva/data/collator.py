import torch

from .label_utils import TASK_COLUMNS


class ASVACollator:
    def __init__(self, max_attack_lines=32, aux_feature_dim=0):
        self.max_attack_lines = max_attack_lines
        self.aux_feature_dim = aux_feature_dim

    def __call__(self, batch):
        labels = {
            task: torch.tensor([item["labels"][task] for item in batch], dtype=torch.long)
            for task in TASK_COLUMNS
        }
        aux_dim = max(self.aux_feature_dim, max((len(item["aux_features"]) for item in batch), default=0))
        aux_features = torch.zeros(len(batch), aux_dim, dtype=torch.float32)
        for idx, item in enumerate(batch):
            if item["aux_features"]:
                aux_features[idx, : len(item["aux_features"])] = torch.tensor(item["aux_features"], dtype=torch.float32)
        line_numbers = torch.zeros(len(batch), self.max_attack_lines, dtype=torch.long)
        line_mask = torch.zeros(len(batch), self.max_attack_lines, dtype=torch.bool)
        cfp_targets = torch.zeros(len(batch), self.max_attack_lines, self.max_attack_lines, dtype=torch.float32)
        attack_line_texts = []
        for batch_idx, item in enumerate(batch):
            lines = item["attack_line_texts"][: self.max_attack_lines]
            numbers = item["line_numbers"][: self.max_attack_lines]
            attack_line_texts.append(lines)
            if lines:
                line_mask[batch_idx, : len(lines)] = True
            if numbers:
                line_numbers[batch_idx, : len(numbers)] = torch.tensor(numbers, dtype=torch.long)
            for src, dst in item["cfp_positive_pairs"]:
                if src < self.max_attack_lines and dst < self.max_attack_lines:
                    cfp_targets[batch_idx, src, dst] = 1.0
        return {
            "ids": [item["id"] for item in batch],
            "file_names": [item["file_name"] for item in batch],
            "full_pair_texts": [item["full_pair_text"] for item in batch],
            "ces_pair_texts": [item["ces_pair_text"] for item in batch],
            "attack_line_texts": attack_line_texts,
            "changed_line_numbers": [item.get("changed_line_numbers", []) for item in batch],
            "line_numbers": line_numbers,
            "line_mask": line_mask,
            "cfp_targets": cfp_targets,
            "graphs": [item["graph"] for item in batch],
            "aux_features": aux_features,
            "labels": labels,
            "meta": [item["meta"] for item in batch],
        }
