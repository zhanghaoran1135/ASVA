import re


SENSITIVE_PATTERNS = [
    r"\b(if|else if|switch|case|while|for)\b",
    r"\b(malloc|calloc|realloc|free|memcpy|memmove|memset|strcpy|strncpy|strcat|strncat|sprintf|snprintf)\b",
    r"\b(read|write|recv|send|copy_from_user|copy_to_user)\b",
    r"\b(return|goto|break|continue)\b",
    r"\b(alloc|overflow|underflow|divide|index|array|pointer|null|size|length|count)\b",
    r"->|\*|/|%|\[|\]",
]


def _normalize_line(line):
    return re.sub(r"\s+", " ", line.strip())


def _changed_line_set(changed_lines):
    return {_normalize_line(line) for line in changed_lines.splitlines() if _normalize_line(line)}


def _line_score(line, index, changed_set, changed_hits):
    normalized = _normalize_line(line)
    score = 0.0
    if not normalized:
        return (-1.0, index)
    if normalized in changed_set:
        score += 8.0
        changed_hits.append(index)
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, line):
            score += 1.5
    if any(token in line for token in ("==", "!=", "<", ">", "<=", ">=")):
        score += 1.0
    if line.count("(") != line.count(")"):
        score += 0.25
    return (score, index)


def select_attack_line_candidates(code, changed_lines, max_lines):
    source_lines = code.splitlines()
    changed_set = _changed_line_set(changed_lines)
    changed_hits = []
    scored = [(_line_score(line, idx, changed_set, changed_hits), line) for idx, line in enumerate(source_lines)]
    selected = set()
    for hit in changed_hits:
        for neighbor in (hit - 1, hit, hit + 1):
            if 0 <= neighbor < len(source_lines):
                selected.add(neighbor)
    ranked = sorted(scored, key=lambda item: (-item[0][0], item[0][1]))
    for (score, idx), _line in ranked:
        if len(selected) >= max_lines:
            break
        if score <= 0 and selected:
            continue
        selected.add(idx)
    if len(selected) < max_lines:
        for idx, line in enumerate(source_lines):
            if len(selected) >= max_lines:
                break
            if _normalize_line(line):
                selected.add(idx)
    ordered = sorted(selected)[:max_lines]
    return [source_lines[idx] for idx in ordered], [idx + 1 for idx in ordered]
