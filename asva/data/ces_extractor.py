import logging
import re

LOGGER = logging.getLogger(__name__)


def _sanitize_line(line):
    line = re.sub(r"``.*``", "<STR>", line)
    line = re.sub(r"'.*?'", "<STR>", line)
    line = re.sub(r'".*?"', "<STR>", line)
    return re.sub(r"{.*}", "<STR>", line)


def parse_line_numbers(value):
    if isinstance(value, list):
        return sorted({int(v) for v in value if str(v).strip()})
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    parts = re.split(r"[\s,;]+", text)
    return sorted({int(part) for part in parts if part.isdigit()})


class CESResult:
    def __init__(self, text, line_numbers, method):
        self.text = text
        self.line_numbers = line_numbers
        self.method = method


class CESExtractor:
    def __init__(self, context_window=2):
        self.context_window = context_window
        self.parser_available = False
        try:
            import tree_sitter

            self.parser_available = True
        except Exception:
            self.parser_available = False

    def extract(self, code, changed_lines):
        lines = code.splitlines()
        if not lines:
            return CESResult(text="", line_numbers=[], method="empty")
        change_ids = parse_line_numbers(changed_lines)
        if not change_ids:
            limit = min(len(lines), max(1, self.context_window * 2 + 1))
            return CESResult(text="\n".join(lines[:limit]), line_numbers=list(range(1, limit + 1)), method="context_window_fallback")
        if self.parser_available:
            LOGGER.debug("Parser support detected but no grammar-specific extractor is configured; falling back.")
        scope_result = self._scope_heuristic(lines, change_ids)
        if scope_result.line_numbers:
            return scope_result
        return self._context_window(lines, change_ids)

    def _scope_heuristic(self, lines, changed_lines):
        selected = set()
        zero_based = [line - 1 for line in changed_lines if 1 <= line <= len(lines)]
        for diff in zero_based:
            open_balance = 0
            start_idx = -1
            for idx in range(diff, -1, -1):
                line = _sanitize_line(lines[idx])
                if "{" in line:
                    open_balance -= line.count("{")
                    if open_balance < 0:
                        start_idx = idx
                        break
                if "}" in line:
                    open_balance += line.count("}")
            if start_idx >= 0:
                begin_line = _sanitize_line(lines[start_idx])
                if "}" not in begin_line:
                    for idx in range(start_idx - 1, -1, -1):
                        line = _sanitize_line(lines[idx])
                        if ";" not in line and "}" not in line and "{" not in line:
                            selected.add(idx)
                        else:
                            break
                selected.add(start_idx)
            close_balance = 0
            for idx in range(diff, len(lines)):
                line = _sanitize_line(lines[idx])
                if "}" in line:
                    if "{" in line.split("}", 1)[0]:
                        close_balance += 1
                    close_balance -= line.count("}")
                    if close_balance < 0:
                        selected.add(idx)
                        break
                if "{" in line and "}" not in line:
                    close_balance += line.count("{")
            selected.add(diff)
        line_numbers = sorted(idx + 1 for idx in selected if 0 <= idx < len(lines))
        if not line_numbers:
            return CESResult(text="", line_numbers=[], method="scope_heuristic_failed")
        text = "\n".join(lines[idx - 1] for idx in line_numbers)
        return CESResult(text=text, line_numbers=line_numbers, method="scope_heuristic")

    def _context_window(self, lines, changed_lines):
        selected = set()
        for line in changed_lines:
            start = max(1, line - self.context_window)
            end = min(len(lines), line + self.context_window)
            selected.update(range(start, end + 1))
        ordered = sorted(selected)
        return CESResult(
            text="\n".join(lines[idx - 1] for idx in ordered),
            line_numbers=ordered,
            method="context_window_fallback",
        )
