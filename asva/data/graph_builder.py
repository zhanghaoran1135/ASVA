import re

from .ces_extractor import parse_line_numbers

IDENT_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


class GraphRecord:
    def __init__(self, node_texts, edge_index, edge_types, line_numbers, graph_source, metadata):
        self.node_texts = node_texts
        self.edge_index = edge_index
        self.edge_types = edge_types
        self.line_numbers = line_numbers
        self.graph_source = graph_source
        self.metadata = metadata


def build_heuristic_graph(code, changed_lines, max_nodes=48):
    raw_lines = code.splitlines()
    indexed_lines = [(idx + 1, line.strip()) for idx, line in enumerate(raw_lines) if line.strip()]
    indexed_lines = indexed_lines[:max_nodes]
    node_texts = [line for _, line in indexed_lines]
    line_numbers = [line_no for line_no, _ in indexed_lines]
    node_count = len(node_texts)
    changed = set(parse_line_numbers(changed_lines))
    edge_index = []
    edge_types = []
    identifiers = [{token for token in IDENT_PATTERN.findall(line)} for line in node_texts]
    assignments = {}
    for idx, line in enumerate(node_texts):
        if "=" in line and "==" not in line:
            left = line.split("=", 1)[0]
            for token in IDENT_PATTERN.findall(left):
                assignments.setdefault(token, []).append(idx)
    for idx in range(node_count - 1):
        edge_index.append([idx, idx + 1])
        edge_types.append("LINE_ADJ")
        edge_index.append([idx + 1, idx])
        edge_types.append("LINE_ADJ")
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if identifiers[i] & identifiers[j]:
                edge_index.extend([[i, j], [j, i]])
                edge_types.extend(["IDENT_SHARE", "IDENT_SHARE"])
            if line_numbers[i] in changed and line_numbers[j] in changed:
                edge_index.extend([[i, j], [j, i]])
                edge_types.extend(["CHANGED", "CHANGED"])
    for token, defs in assignments.items():
        for def_idx in defs:
            for use_idx in range(def_idx + 1, node_count):
                if token in identifiers[use_idx]:
                    edge_index.append([def_idx, use_idx])
                    edge_types.append("APPROX_DEF_USE")
    if not edge_index and node_count > 1:
        edge_index = [[idx, idx + 1] for idx in range(node_count - 1)]
        edge_types = ["LINE_ADJ"] * len(edge_index)
    return GraphRecord(
        node_texts=node_texts,
        edge_index=edge_index,
        edge_types=edge_types,
        line_numbers=line_numbers,
        graph_source="heuristic_fallback",
        metadata={"reason": "no_matching_joern_json"},
    )
