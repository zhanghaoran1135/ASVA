import json
import logging
import re
from pathlib import Path

from .cache_utils import ensure_dir
from .graph_builder import GraphRecord, build_heuristic_graph

LOGGER = logging.getLogger(__name__)

EDGE_SOURCE_KEYS = ["outNode", "src", "source", "outnode", "start", "from"]
EDGE_TARGET_KEYS = ["inNode", "dst", "target", "innode", "end", "to"]
EDGE_TYPE_KEYS = ["label", "_label", "etype", "type"]
NODE_TEXT_KEYS = ["code", "name", "fullName", "value", "canonicalName"]
NODE_LINE_KEYS = ["lineNumber", "line", "lineNo"]
NODE_LABEL_KEYS = ["_label", "label", "type"]
CFP_EDGE_TYPES = {"CFG", "CDG"}
CONTROL_RELATED_EDGE_TYPES = {"CFG", "CDG", "DOMINATE", "POST_DOMINATE"}
CONTROL_DEP_EDGE_TYPES = {"CFG", "CDG", "DOMINATE", "POST_DOMINATE"}
DATA_DEP_EDGE_TYPES = {"REACHING_DEF"}
DEPENDENCY_EDGE_TYPES = CONTROL_DEP_EDGE_TYPES | DATA_DEP_EDGE_TYPES


def _normalize_name(value):
    text = str(value or "").strip().lower()
    text = text.replace("\\", "/")
    return re.sub(r"[^a-z0-9]+", "", text)


class GraphMatch:
    def __init__(self, graph_key, node_path, edge_path, source_path, match_strategy):
        self.graph_key = graph_key
        self.node_path = node_path
        self.edge_path = edge_path
        self.source_path = source_path
        self.match_strategy = match_strategy


class CFPLabelResult:
    def __init__(self, positive_pairs, cfp_source):
        self.positive_pairs = positive_pairs
        self.cfp_source = cfp_source


class GraphRepository:
    def __init__(self, graph_dir, content_match_limit=128, content_match_chars=2048):
        self.graph_dir = Path(graph_dir)
        self.content_match_limit = content_match_limit
        self.content_match_chars = content_match_chars
        self.by_key = {}
        self.by_idstem = {}
        self.by_basename = {}
        self.by_stem = {}
        self._build_index()

    def _build_index(self):
        ensure_dir(self.graph_dir)
        for node_path in self.graph_dir.glob("*.nodes.json"):
            self._register_key(node_path.name[: -len(".nodes.json")])

    def _register_key(self, key):
        edge_path = self.graph_dir / f"{key}.edges.json"
        source_path = self.graph_dir / key
        node_path = self.graph_dir / f"{key}.nodes.json"
        if not node_path.exists():
            return
        stem = Path(key).stem
        basename = Path(key).name
        self.by_key[key] = {
            "node_path": node_path,
            "edge_path": edge_path if edge_path.exists() else None,
            "source_path": source_path if source_path.exists() else None,
        }
        self.by_idstem[stem] = key
        self.by_basename.setdefault(_normalize_name(basename), [])
        if key not in self.by_basename[_normalize_name(basename)]:
            self.by_basename[_normalize_name(basename)].append(key)
        self.by_stem.setdefault(_normalize_name(stem), [])
        if key not in self.by_stem[_normalize_name(stem)]:
            self.by_stem[_normalize_name(stem)].append(key)

    def match(self, sample):
        file_name = str(sample.get("file_name", "") or "")
        sample_id = str(sample.get("id", "") or "")
        commit_id = str(sample.get("commit_id", "") or "")
        candidates = []
        basename_key = _normalize_name(Path(file_name).name)
        stem_key = _normalize_name(Path(file_name).stem)
        if basename_key in self.by_basename:
            candidates.extend([(key, "file_name_exact_basename") for key in self.by_basename[basename_key]])
        if stem_key in self.by_stem:
            candidates.extend([(key, "file_name_normalized_stem") for key in self.by_stem[stem_key]])
        if sample_id in self.by_key:
            candidates.append((sample_id, "id_exact"))
        elif sample_id in self.by_idstem:
            candidates.append((self.by_idstem[sample_id], "id_stem_exact"))
        if commit_id:
            for key in self.by_key:
                if commit_id in key:
                    candidates.append((key, "commit_id_substring"))
        if not candidates:
            content_match = self._content_match(sample)
            if content_match:
                candidates.append((content_match, "content_match"))
        if not candidates:
            return GraphMatch(None, None, None, None, "graph_missing")
        graph_key, strategy = candidates[0]
        paths = self.by_key[graph_key]
        return GraphMatch(
            graph_key=graph_key,
            node_path=str(paths["node_path"]) if paths["node_path"] else None,
            edge_path=str(paths["edge_path"]) if paths["edge_path"] else None,
            source_path=str(paths["source_path"]) if paths["source_path"] else None,
            match_strategy=strategy,
        )

    def _content_match(self, sample):
        source = str(sample.get("func_before", "") or "").strip()
        if not source:
            return None
        source_norm = re.sub(r"\s+", "", source[: self.content_match_chars])
        checked = 0
        for key, paths in self.by_key.items():
            if checked >= self.content_match_limit:
                break
            checked += 1
            src_path = paths["source_path"]
            if not src_path or not Path(src_path).exists():
                continue
            try:
                text = Path(src_path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if source_norm and source_norm[:512] in re.sub(r"\s+", "", text):
                return key
        return None

    def ensure_match(self, sample, joern_runner=None):
        match = self.match(sample)
        if match.node_path and match.edge_path:
            return match, "joern_json"
        if joern_runner is None:
            return match, "heuristic_fallback"
        try:
            source_path = joern_runner.materialize_source(sample, self.graph_dir)
            generated = joern_runner.generate(source_path)
        except Exception as exc:
            LOGGER.warning("Joern generation failed for sample %s: %s", sample.get("id"), exc)
            return match, "heuristic_fallback"
        if generated:
            self._register_key(source_path.name)
            regenerated_match = self.match(sample)
            if regenerated_match.node_path and regenerated_match.edge_path:
                return regenerated_match, "joern_generated"
        return match, "heuristic_fallback"


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_raw_nodes_edges(node_path, edge_path):
    raw_nodes = _read_json(node_path)
    raw_edges = _read_json(edge_path)
    if not isinstance(raw_nodes, list):
        if isinstance(raw_nodes, dict):
            for value in raw_nodes.values():
                if isinstance(value, list):
                    raw_nodes = value
                    break
    nodes = [node for node in raw_nodes if isinstance(node, dict)]
    return nodes, raw_edges


def _extract_node_text(node):
    for key in NODE_TEXT_KEYS:
        value = node.get(key)
        if value not in (None, "", "<empty>"):
            return str(value)
    label = next((node.get(key) for key in NODE_LABEL_KEYS if node.get(key)), "")
    return str(label or "NODE")


def _extract_line_number(node):
    for key in NODE_LINE_KEYS:
        value = node.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except Exception:
                continue
    return -1


def _extract_node_id(node):
    return str(node.get("id", node.get("_id", node.get("nodeId", ""))))


def _normalize_edges(raw_edges):
    edges = []
    if isinstance(raw_edges, list):
        for entry in raw_edges:
            if isinstance(entry, list):
                if len(entry) >= 3:
                    src, dst, edge_type = entry[0], entry[1], entry[2]
                    edges.append((str(src), str(dst), str(edge_type)))
            elif isinstance(entry, dict):
                src = next((entry.get(key) for key in EDGE_SOURCE_KEYS if entry.get(key) is not None), None)
                dst = next((entry.get(key) for key in EDGE_TARGET_KEYS if entry.get(key) is not None), None)
                edge_type = next((entry.get(key) for key in EDGE_TYPE_KEYS if entry.get(key)), "EDGE")
                if src is not None and dst is not None:
                    edges.append((str(src), str(dst), str(edge_type)))
    elif isinstance(raw_edges, dict):
        for value in raw_edges.values():
            edges.extend(_normalize_edges(value))
    return edges


def _sort_candidate_lines(candidate_lines, prioritize_line_numbers):
    if not prioritize_line_numbers:
        return sorted(candidate_lines)
    valid_priorities = sorted({int(line) for line in prioritize_line_numbers if int(line) > 0})
    if not valid_priorities:
        return sorted(candidate_lines)

    def _key(line_number):
        min_distance = min(abs(int(line_number) - priority) for priority in valid_priorities)
        exact_priority = 0 if min_distance == 0 else 1
        return (exact_priority, min_distance, int(line_number))

    return sorted(candidate_lines, key=_key)


def parse_joern_graph(
    node_path,
    edge_path,
    max_nodes=48,
    prioritize_line_numbers=None,
):
    nodes, raw_edges = _load_raw_nodes_edges(node_path, edge_path)
    nodes = [node for node in nodes if str(node.get("_label", node.get("label", ""))).upper() not in {"COMMENT", "FILE", "META_DATA"}]
    node_by_id = {_extract_node_id(node): node for node in nodes}
    node_text_by_line = {}
    for node in nodes:
        line_number = _extract_line_number(node)
        if line_number <= 0:
            continue
        text = _extract_node_text(node).strip()
        if not text or text == "<empty>":
            continue
        node_text_by_line.setdefault(line_number, [])
        if text not in node_text_by_line[line_number]:
            node_text_by_line[line_number].append(text)
    dependency_edges = []
    for src, dst, edge_type in _normalize_edges(raw_edges):
        if edge_type not in DEPENDENCY_EDGE_TYPES:
            continue
        src_node = node_by_id.get(src)
        dst_node = node_by_id.get(dst)
        if src_node is None or dst_node is None:
            continue
        src_line = _extract_line_number(src_node)
        dst_line = _extract_line_number(dst_node)
        if src_line <= 0 or dst_line <= 0 or src_line == dst_line:
            continue
        coarse_type = "DATA" if edge_type in DATA_DEP_EDGE_TYPES else "CONTROL"
        dependency_edges.append((src_line, dst_line, coarse_type))
    candidate_lines = sorted({line for line, _, _ in dependency_edges} | {line for _, line, _ in dependency_edges})
    if not candidate_lines:
        candidate_lines = sorted(node_text_by_line.keys())
    candidate_lines = _sort_candidate_lines(candidate_lines, prioritize_line_numbers)[:max_nodes]
    id_map = {line_number: idx for idx, line_number in enumerate(candidate_lines)}
    node_texts = [" ".join(node_text_by_line.get(line_number, [f"line_{line_number}"])) for line_number in candidate_lines]
    line_numbers = list(candidate_lines)
    edge_index = []
    edge_types = []
    seen_edges = set()
    for src_line, dst_line, coarse_type in dependency_edges:
        if src_line in id_map and dst_line in id_map:
            edge_tuple = (id_map[src_line], id_map[dst_line], coarse_type)
            if edge_tuple in seen_edges:
                continue
            seen_edges.add(edge_tuple)
            edge_index.append([edge_tuple[0], edge_tuple[1]])
            edge_types.append(coarse_type)
    if not edge_index and len(nodes) > 1:
        for idx in range(len(candidate_lines) - 1):
            edge_index.append([idx, idx + 1])
            edge_types.append("CONTROL")
    return GraphRecord(
        node_texts=node_texts,
        edge_index=edge_index,
        edge_types=edge_types,
        line_numbers=line_numbers,
        graph_source="joern_json",
        metadata={"node_path": str(node_path), "edge_path": str(edge_path), "graph_granularity": "dependency_line_regions"},
    )


def extract_cfp_line_pairs(
    node_path,
    edge_path,
    max_lines=None,
    allowed_line_numbers=None,
    preferred_edge_types=None,
):
    preferred_edge_types = preferred_edge_types or CFP_EDGE_TYPES
    allowed_lines = set(int(line) for line in (allowed_line_numbers or []) if int(line) > 0)
    selected_index = {int(line): idx for idx, line in enumerate(allowed_line_numbers or [])}
    nodes, raw_edges = _load_raw_nodes_edges(node_path, edge_path)
    node_line_map = {
        _extract_node_id(node): _extract_line_number(node)
        for node in nodes
    }
    pairs = set()
    related_pairs = set()
    for src, dst, edge_type in _normalize_edges(raw_edges):
        src_line = node_line_map.get(src, -1)
        dst_line = node_line_map.get(dst, -1)
        if src_line <= 0 or dst_line <= 0:
            continue
        if src_line == dst_line:
            continue
        if allowed_lines:
            if src_line not in allowed_lines or dst_line not in allowed_lines:
                continue
            pair = (selected_index[src_line], selected_index[dst_line])
        elif max_lines is not None and (src_line > max_lines or dst_line > max_lines):
            continue
        else:
            pair = (src_line - 1, dst_line - 1)
        if edge_type in preferred_edge_types:
            pairs.add(pair)
        elif edge_type in CONTROL_RELATED_EDGE_TYPES:
            related_pairs.add(pair)
    if pairs:
        return CFPLabelResult(positive_pairs=[list(pair) for pair in sorted(pairs)], cfp_source="joern")
    if related_pairs:
        return CFPLabelResult(positive_pairs=[list(pair) for pair in sorted(related_pairs)], cfp_source="joern")
    return CFPLabelResult(positive_pairs=[], cfp_source="fallback")


def load_graph_for_sample(sample, repository, max_nodes=48):
    match = repository.match(sample)
    if match.node_path and match.edge_path:
        try:
            graph = parse_joern_graph(match.node_path, match.edge_path, max_nodes=max_nodes)
            graph.metadata["match_strategy"] = match.match_strategy
            graph.metadata["graph_key"] = match.graph_key or ""
            return graph
        except Exception as exc:
            LOGGER.warning("Failed to parse Joern graph for sample %s: %s", sample.get("id"), exc)
    graph = build_heuristic_graph(
        code=str(sample.get("func_before", "") or ""),
        changed_lines=sample.get("delete_lines"),
        max_nodes=max_nodes,
    )
    graph.metadata["match_strategy"] = match.match_strategy
    if match.graph_key:
        graph.metadata["graph_key"] = match.graph_key
    return graph
