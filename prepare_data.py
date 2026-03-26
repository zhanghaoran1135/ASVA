import argparse
import ast
import logging
import re
from pathlib import Path

import pandas as pd

from asva.data.cache_utils import dump_json, dump_jsonl
from asva.data.ces_extractor import CESExtractor
from asva.data.graph_loader import CFPLabelResult, GraphRepository, extract_cfp_line_pairs
from asva.data.joern_runner import JoernRunner, precompute_missing_joern_graphs
from asva.data.line_selector import select_attack_line_candidates
from asva.data.label_utils import TASK_COLUMNS, build_label_encoders, save_label_encoders
from asva.data.pseudo_labels import build_fallback_cfp_pairs_for_selected_lines
from asva.data.text_windowing import TokenBudgetWindowBuilder, WindowingConfig
from asva.training.utils import configure_logging, ensure_dir, load_yaml, to_project_relative

LOGGER = logging.getLogger(__name__)


def _clean_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.replace("\r\n", "\n").strip()


def _parse_precomputed_feature(value):
    text = _clean_text(value)
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [float(item) for item in parsed]
    except Exception:
        pass
    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", text, flags=re.IGNORECASE)
    return [float(number) for number in numbers]


def _build_aux_features(row, feature_columns):
    features = []
    for column in feature_columns:
        features.extend(_parse_precomputed_feature(row.get(column, "")))
    return features


def _split_dataframe(df, use_partition_column, seed):
    if use_partition_column and "partition" in df.columns:
        normalized = df["partition"].astype(str).str.strip().str.lower()
        if {"train", "valid", "test"}.issubset(set(normalized.unique())):
            return {
                "train": df[normalized == "train"].copy(),
                "valid": df[normalized == "valid"].copy(),
                "test": df[normalized == "test"].copy(),
            }
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)
    return {
        "train": shuffled.iloc[:train_end].copy(),
        "valid": shuffled.iloc[train_end:valid_end].copy(),
        "test": shuffled.iloc[valid_end:].copy(),
    }


def prepare_records(config, limit=None):
    data_cfg = config["data"]
    dataset_path = Path(config["paths"]["dataset_csv"])
    df = pd.read_csv(dataset_path)
    if limit:
        df = df.iloc[:limit].copy()
    required = ["id", "func_before", "func_after", *TASK_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].fillna("").map(_clean_text)
    if not data_cfg["keep_invalid_rows"]:
        mask = df["func_before"].astype(str).str.strip().ne("") & df["func_after"].astype(str).str.strip().ne("")
        for task in TASK_COLUMNS:
            mask &= df[task].astype(str).str.strip().ne("")
        df = df[mask].copy()
    split_frames = _split_dataframe(df, bool(data_cfg["use_partition_column"]), int(config["runtime"]["seed"]))
    ces_extractor = CESExtractor()
    window_builder = TokenBudgetWindowBuilder(
        codebert_path=config["paths"]["codebert_path"],
        config=WindowingConfig(
            full_text_mode=str(data_cfg.get("full_text_mode", "full")),
            full_window_context_lines=int(data_cfg.get("full_window_context_lines", 20)),
            full_window_max_lines=int(data_cfg.get("full_window_max_lines", 80)),
            full_window_token_budget=int(data_cfg.get("full_window_token_budget", 0)),
            ces_token_budget=int(data_cfg.get("ces_token_budget", data_cfg.get("max_seq_length_ces", 128))),
        ),
    )
    graph_repo = GraphRepository(
        config["paths"]["graph_dir"],
        content_match_limit=int(data_cfg["graph_content_match_limit"]),
        content_match_chars=int(data_cfg["graph_content_match_chars"]),
    )
    joern_runner = None
    joern_prefetch_summary = None
    failed_joern_ids = set()
    if data_cfg.get("generate_missing_joern", False):
        joern_runner = JoernRunner(
            joern_cli_dir=config["paths"]["joern_cli_dir"],
            script_path=config["paths"]["joern_script_path"],
            timeout_sec=int(data_cfg.get("joern_timeout_sec", 180)),
            verbose=int(data_cfg.get("joern_verbose", 0)),
        )
        joern_prefetch_samples = []
        for split_df in split_frames.values():
            for _, row in split_df.iterrows():
                func_before = _clean_text(row.get("func_before"))
                if not func_before and data_cfg["use_blaming_fallback"]:
                    func_before = _clean_text(row.get("blaming_func_before"))
                joern_prefetch_samples.append(
                    {
                        "id": row.get("id"),
                        "file_name": row.get("file_name", ""),
                        "commit_id": row.get("commit_id", ""),
                        "func_before": func_before,
                    }
                )
        joern_prefetch_summary = precompute_missing_joern_graphs(
            joern_prefetch_samples,
            graph_repo,
            joern_runner,
            workers=int(data_cfg.get("joern_workers", 8)),
        )
        failed_joern_ids = set(joern_prefetch_summary["failed_ids"])
        LOGGER.info(
            "Joern precompute finished: existing=%s queued=%s generated=%s failed=%s",
            joern_prefetch_summary["existing"],
            joern_prefetch_summary["queued"],
            joern_prefetch_summary["generated"],
            joern_prefetch_summary["failed"],
        )
    all_records = []
    per_split_records = {"train": [], "valid": [], "test": []}
    aux_dims = []
    graph_source_counter = {"joern_json": 0, "joern_generated": 0, "heuristic_fallback": 0}
    cfp_source_counter = {"joern": 0, "fallback": 0}
    graph_match_rows = []
    sample_index = 0
    for split_name, split_df in split_frames.items():
        for _, row in split_df.iterrows():
            func_before = _clean_text(row.get("func_before"))
            func_after = _clean_text(row.get("func_after"))
            if not func_before and data_cfg["use_blaming_fallback"]:
                func_before = _clean_text(row.get("blaming_func_before"))
            if not func_after and data_cfg["use_blaming_fallback"]:
                func_after = _clean_text(row.get("blaming_func_after"))
            delete_lines = _clean_text(row.get("delete_lines"))
            add_lines = _clean_text(row.get("add_lines"))
            ces_before = ces_extractor.extract(func_before, delete_lines)
            ces_after = ces_extractor.extract(func_after, add_lines)
            ces_before_text = window_builder.clip_text_to_budget(
                ces_before.text,
                token_budget=int(data_cfg.get("ces_token_budget", data_cfg["max_seq_length_ces"])),
            )
            ces_after_text = window_builder.clip_text_to_budget(
                ces_after.text,
                token_budget=int(data_cfg.get("ces_token_budget", data_cfg["max_seq_length_ces"])),
            )
            attack_line_texts, line_numbers = select_attack_line_candidates(
                func_before,
                delete_lines,
                max_lines=int(data_cfg["max_attack_lines"]),
            )
            graph_match, graph_backend = graph_repo.ensure_match(
                {
                    "id": row.get("id"),
                    "file_name": row.get("file_name", ""),
                    "commit_id": row.get("commit_id", ""),
                    "func_before": func_before,
                },
                joern_runner=joern_runner if str(row.get("id")) not in failed_joern_ids else None,
            )
            if graph_match.node_path and graph_match.edge_path:
                cfp_result = extract_cfp_line_pairs(
                    graph_match.node_path,
                    graph_match.edge_path,
                    allowed_line_numbers=line_numbers,
                )
                if not cfp_result.positive_pairs:
                    fallback = build_fallback_cfp_pairs_for_selected_lines(attack_line_texts, line_numbers)
                    cfp_result = CFPLabelResult(
                        positive_pairs=fallback.cfp_positive_pairs,
                        cfp_source=fallback.cfp_source,
                    )
            else:
                fallback = build_fallback_cfp_pairs_for_selected_lines(attack_line_texts, line_numbers)
                cfp_result = CFPLabelResult(positive_pairs=fallback.cfp_positive_pairs, cfp_source=fallback.cfp_source)
            aux_features = _build_aux_features(row, data_cfg["precomputed_feature_columns"])
            aux_dims.append(len(aux_features))
            graph_source_counter[graph_backend] = graph_source_counter.get(graph_backend, 0) + 1
            cfp_source_counter[cfp_result.cfp_source] = cfp_source_counter.get(cfp_result.cfp_source, 0) + 1
            labels = {task: str(row[task]).strip() for task in TASK_COLUMNS}
            record = {
                "id": int(row["id"]),
                "file_name": _clean_text(row.get("file_name")),
                "commit_id": _clean_text(row.get("commit_id")),
                "func_before": func_before,
                "func_after": func_after,
                "delete_lines": delete_lines,
                "add_lines": add_lines,
                "split": split_name,
                "full_pair_text": window_builder.build_full_pair_text(
                    func_before=func_before,
                    func_after=func_after,
                    delete_lines=delete_lines,
                    add_lines=add_lines,
                    max_seq_length_full=int(data_cfg["max_seq_length_full"]),
                ),
                "ces_pair_text": ces_before_text + "\n</s>\n" + ces_after_text,
                "ces_method_before": ces_before.method,
                "ces_method_after": ces_after.method,
                "attack_line_texts": attack_line_texts,
                "line_numbers": line_numbers,
                "cfp_positive_pairs": cfp_result.positive_pairs,
                "cfp_source": cfp_result.cfp_source,
                "graph": {
                    "graph_key": graph_match.graph_key,
                    "node_path": to_project_relative(graph_match.node_path),
                    "edge_path": to_project_relative(graph_match.edge_path),
                    "source_path": to_project_relative(graph_match.source_path),
                    "match_strategy": graph_match.match_strategy,
                    "graph_backend": graph_backend,
                },
                "aux_features": aux_features,
                "labels": labels,
            }
            graph_match_rows.append(
                {
                    "sample_index": sample_index,
                    "id": int(row["id"]),
                    "file_name": _clean_text(row.get("file_name")),
                    "matched_source_file": to_project_relative(graph_match.source_path) or "",
                    "matched_nodes_json": to_project_relative(graph_match.node_path) or "",
                    "matched_edges_json": to_project_relative(graph_match.edge_path) or "",
                    "graph_backend": graph_backend,
                    "cfp_source": cfp_result.cfp_source,
                }
            )
            sample_index += 1
            per_split_records[split_name].append(record)
            all_records.append(record)
    label_bundle = build_label_encoders([{task: record["labels"][task] for task in TASK_COLUMNS} for record in all_records])
    for records in per_split_records.values():
        for record in records:
            record["labels"] = label_bundle.encode_row(record["labels"])
    return {
        "records_by_split": per_split_records,
        "label_bundle": label_bundle,
        "metadata": {
            "dataset_csv": to_project_relative(dataset_path),
            "codebert_path": to_project_relative(config["paths"]["codebert_path"]),
            "graph_dir": to_project_relative(config["paths"]["graph_dir"]),
            "num_records": len(all_records),
            "split_sizes": {split: len(records) for split, records in per_split_records.items()},
            "aux_feature_dim": max(aux_dims) if aux_dims else 0,
            "graph_match_summary": graph_source_counter,
            "cfp_source_summary": cfp_source_counter,
            "feature_mode": data_cfg["feature_mode"],
            "ces_hierarchy": [
                "parser_if_available",
                "scope_heuristic",
                "context_window_fallback",
            ],
            "joern_prefetch_summary": joern_prefetch_summary or {},
        },
        "graph_match_report": graph_match_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare ASVA dataset")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    processed_dir = ensure_dir(config["paths"]["processed_dir"])
    logger = configure_logging(config["paths"]["log_dir"], name="prepare_data")
    logger.info("Loading dataset from %s", config["paths"]["dataset_csv"])
    prepared = prepare_records(config, limit=args.limit)
    for split, records in prepared["records_by_split"].items():
        dump_jsonl(records, processed_dir / f"{split}.jsonl")
    save_label_encoders(prepared["label_bundle"], processed_dir / "label_mappings.json")
    dump_json(prepared["metadata"], processed_dir / "metadata.json")
    pd.DataFrame(prepared["graph_match_report"]).to_csv(processed_dir / "graph_match_report.csv", index=False)
    logger.info("Prepared %s records", prepared["metadata"]["num_records"])


if __name__ == "__main__":
    main()
