import argparse
import json
from pathlib import Path
import warnings

from torch.utils.data import DataLoader

from asva.data.collator import ASVACollator
from asva.data.ces_extractor import CESExtractor
from asva.data.graph_builder import build_heuristic_graph
from asva.data.line_selector import select_attack_line_candidates
from asva.data.label_utils import TASK_COLUMNS, load_label_encoders
from asva.data.pseudo_labels import build_fallback_cfp_pairs_for_selected_lines
from asva.data.text_windowing import TokenBudgetWindowBuilder, WindowingConfig
from asva.models.asva_model import ASVAModel
from asva.training.optim import build_optimizer
from asva.training.trainer import Trainer
from asva.training.utils import configure_logging, ensure_dir, get_device, load_yaml


class InferenceDataset:
    def __init__(self, samples, feature_mode, max_graph_nodes):
        self.samples = samples
        self.feature_mode = feature_mode
        self.max_graph_nodes = max_graph_nodes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        use_raw = self.feature_mode != "precomputed_only"
        return {
            "id": sample["id"],
            "file_name": sample.get("file_name", ""),
            "full_pair_text": sample["full_pair_text"] if use_raw else "",
            "ces_pair_text": sample["ces_pair_text"] if use_raw else "",
            "attack_line_texts": sample["attack_line_texts"] if use_raw else [],
            "line_numbers": sample["line_numbers"] if use_raw else [],
            "cfp_positive_pairs": sample["cfp_positive_pairs"] if use_raw else [],
            "aux_features": sample.get("aux_features", []) if self.feature_mode != "raw_code_only" else [],
            "graph": sample["graph"] if use_raw else {"node_texts": [], "edge_index": [], "edge_types": [], "line_numbers": [], "graph_source": "disabled", "metadata": {}},
            "labels": {task: 0 for task in TASK_COLUMNS},
            "meta": {"split": "infer", "graph_source": sample["graph"]["graph_source"], "graph_match_strategy": "heuristic_inference", "cfp_source": sample.get("cfp_source", "fallback")},
        }


def _build_sample_from_args(args, config):
    if args.input_json:
        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            if len(payload) != 1:
                raise ValueError("Single-sample mode expects one JSON object")
            payload = payload[0]
        func_before = str(payload["func_before"])
        func_after = str(payload["func_after"])
        delete_lines = str(payload.get("delete_lines", ""))
        add_lines = str(payload.get("add_lines", ""))
        file_name = str(payload.get("file_name", "inference_sample.c"))
        sample_id = int(payload.get("id", 0))
    else:
        func_before = args.func_before
        func_after = args.func_after
        delete_lines = args.delete_lines
        add_lines = args.add_lines
        file_name = args.file_name
        sample_id = 0
    ces = CESExtractor()
    ces_before = ces.extract(func_before, delete_lines)
    ces_after = ces.extract(func_after, add_lines)
    window_builder = TokenBudgetWindowBuilder(
        codebert_path=config["paths"]["codebert_path"],
        config=WindowingConfig(
            full_text_mode=str(config["data"].get("full_text_mode", "full")),
            full_window_context_lines=int(config["data"].get("full_window_context_lines", 20)),
            full_window_max_lines=int(config["data"].get("full_window_max_lines", 80)),
            full_window_token_budget=int(config["data"].get("full_window_token_budget", 0)),
            ces_token_budget=int(config["data"].get("ces_token_budget", config["data"]["max_seq_length_ces"])),
        ),
    )
    attack_line_texts, line_numbers = select_attack_line_candidates(
        func_before,
        delete_lines,
        max_lines=args.max_attack_lines,
    )
    cfp_fallback = build_fallback_cfp_pairs_for_selected_lines(attack_line_texts, line_numbers)
    graph = build_heuristic_graph(func_before, delete_lines, max_nodes=args.max_graph_nodes)
    return {
        "id": sample_id,
        "file_name": file_name,
        "full_pair_text": window_builder.build_full_pair_text(
            func_before=func_before,
            func_after=func_after,
            delete_lines=delete_lines,
            add_lines=add_lines,
            max_seq_length_full=int(config["data"]["max_seq_length_full"]),
        ),
        "ces_pair_text": window_builder.clip_text_to_budget(
            ces_before.text,
            token_budget=int(config["data"].get("ces_token_budget", config["data"]["max_seq_length_ces"])),
        )
        + "\n</s>\n"
        + window_builder.clip_text_to_budget(
            ces_after.text,
            token_budget=int(config["data"].get("ces_token_budget", config["data"]["max_seq_length_ces"])),
        ),
        "attack_line_texts": attack_line_texts,
        "line_numbers": line_numbers,
        "cfp_positive_pairs": cfp_fallback.cfp_positive_pairs,
        "cfp_source": cfp_fallback.cfp_source,
        "aux_features": [],
        "graph": {
            "node_texts": graph.node_texts,
            "edge_index": graph.edge_index,
            "edge_types": graph.edge_types,
            "line_numbers": graph.line_numbers,
            "graph_source": graph.graph_source,
            "metadata": graph.metadata,
        },
    }


def main():
    warnings.filterwarnings(
        "ignore",
        message=r".*torch\.cpu\.amp\.autocast.*deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*nested tensors is in prototype stage.*",
        category=UserWarning,
    )
    parser = argparse.ArgumentParser(description="Infer with ASVA")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--func-before", default="")
    parser.add_argument("--func-after", default="")
    parser.add_argument("--delete-lines", default="")
    parser.add_argument("--add-lines", default="")
    parser.add_argument("--file-name", default="inference_sample.c")
    parser.add_argument("--max-attack-lines", type=int, default=None)
    parser.add_argument("--max-graph-nodes", type=int, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.max_attack_lines is None:
        args.max_attack_lines = int(config["data"]["max_attack_lines"])
    if args.max_graph_nodes is None:
        args.max_graph_nodes = int(config["data"]["max_graph_nodes"])
    prepared_dir = Path(args.prepared_dir or config["paths"]["processed_dir"])
    metadata = json.loads((prepared_dir / "metadata.json").read_text(encoding="utf-8"))
    label_bundle = load_label_encoders(prepared_dir / "label_mappings.json")
    task_dims = {task: len(label_bundle.label_to_id[task]) for task in label_bundle.label_to_id}
    sample = _build_sample_from_args(args, config)
    dataset = InferenceDataset([sample], config["data"]["feature_mode"], args.max_graph_nodes)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=ASVACollator(max_attack_lines=args.max_attack_lines, aux_feature_dim=int(metadata.get("aux_feature_dim", 0))),
    )
    device = get_device(config["runtime"]["device"])
    model = ASVAModel(
        config,
        task_dims=task_dims,
        aux_feature_dim=int(metadata.get("aux_feature_dim", 0)),
    ).to(device)
    optimizer = build_optimizer(model, config)
    logger = configure_logging(config["paths"]["log_dir"], name="infer")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        config=config,
        device=device,
        logger=logger,
        checkpoint_dir=ensure_dir(config["paths"]["checkpoint_dir"]),
        log_dir=ensure_dir(config["paths"]["log_dir"]),
    )
    trainer.load_checkpoint(args.checkpoint, load_optimizer=False, load_scheduler=False)
    model.eval()
    batch = next(iter(loader))
    trainer._move_labels(batch)
    with __import__("torch").no_grad():
        outputs = model(batch)
    predictions = {}
    for task, logits in outputs["logits"].items():
        pred_id = int(logits.argmax(dim=-1).item())
        predictions[task] = {
            "label_id": pred_id,
            "label": label_bundle.id_to_label[task][pred_id],
        }
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
