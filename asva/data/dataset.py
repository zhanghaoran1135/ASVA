import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .cache_utils import load_json, load_jsonl
from .graph_loader import parse_joern_graph
from .graph_builder import build_heuristic_graph
from asva.training.utils import resolve_project_path

LOGGER = logging.getLogger(__name__)


def _parse_line_numbers(text):
    values = []
    for token in str(text or "").replace(",", " ").split():
        try:
            number = int(token)
        except ValueError:
            continue
        if number > 0:
            values.append(number)
    return values


class ASVADataset(Dataset):
    def __init__(
        self,
        prepared_dir,
        split,
        feature_mode="raw_plus_precomputed",
        max_graph_nodes=48,
    ):
        self.prepared_dir = Path(prepared_dir)
        self.split = split
        self.feature_mode = feature_mode
        self.max_graph_nodes = max_graph_nodes
        self.records = load_jsonl(self.prepared_dir / f"{split}.jsonl")
        self.metadata = load_json(self.prepared_dir / "metadata.json")
        self.graph_cache = {}

    def __len__(self):
        return len(self.records)

    def _load_graph(self, record):
        sample_id = int(record["id"])
        if sample_id in self.graph_cache:
            return self.graph_cache[sample_id]
        graph_info = record.get("graph", {})
        if graph_info.get("node_path") and graph_info.get("edge_path"):
            try:
                graph = parse_joern_graph(
                    resolve_project_path(graph_info["node_path"]),
                    resolve_project_path(graph_info["edge_path"]),
                    max_nodes=self.max_graph_nodes,
                    prioritize_line_numbers=_parse_line_numbers(record.get("delete_lines", "")),
                )
            except Exception as exc:
                LOGGER.warning("Graph parse failed for %s: %s", sample_id, exc)
                graph = build_heuristic_graph(
                    code=record.get("func_before", ""),
                    changed_lines=record.get("delete_lines", ""),
                    max_nodes=self.max_graph_nodes,
                )
        else:
            graph = build_heuristic_graph(
                code=record.get("func_before", ""),
                changed_lines=record.get("delete_lines", ""),
                max_nodes=self.max_graph_nodes,
            )
        graph_dict = {
            "node_texts": graph.node_texts,
            "edge_index": graph.edge_index,
            "edge_types": graph.edge_types,
            "line_numbers": graph.line_numbers,
            "graph_source": graph.graph_source,
            "metadata": graph.metadata,
        }
        self.graph_cache[sample_id] = graph_dict
        return graph_dict

    def __getitem__(self, index):
        record = self.records[index]
        graph = self._load_graph(record)
        use_raw = self.feature_mode != "precomputed_only"
        use_aux = self.feature_mode != "raw_code_only"
        return {
            "id": int(record["id"]),
            "file_name": record.get("file_name", ""),
            "func_before": record.get("func_before", ""),
            "func_after": record.get("func_after", ""),
            "full_pair_text": record.get("full_pair_text", "") if use_raw else "",
            "ces_pair_text": record.get("ces_pair_text", "") if use_raw else "",
            "attack_line_texts": record.get("attack_line_texts", []) if use_raw else [],
            "line_numbers": record.get("line_numbers", []) if use_raw else [],
            "cfp_positive_pairs": record.get("cfp_positive_pairs", []) if use_raw else [],
            "aux_features": record.get("aux_features", []) if use_aux else [],
            "graph": graph if use_raw else {
                "node_texts": [],
                "edge_index": [],
                "edge_types": [],
                "line_numbers": [],
                "graph_source": "disabled_precomputed_only",
                "metadata": {},
            },
            "labels": record["labels"],
            "changed_line_numbers": _parse_line_numbers(record.get("delete_lines", "")),
            "meta": {
                "split": record.get("split", self.split),
                "graph_source": graph["graph_source"],
                "graph_match_strategy": record.get("graph", {}).get("match_strategy", "unknown"),
                "cfp_source": record.get("cfp_source", "unknown"),
                "ces_method_before": record.get("ces_method_before", ""),
                "ces_method_after": record.get("ces_method_after", ""),
            },
        }


def tensorize_features(features):
    if not features:
        return torch.zeros(0, dtype=torch.float32)
    return torch.tensor(features, dtype=torch.float32)
