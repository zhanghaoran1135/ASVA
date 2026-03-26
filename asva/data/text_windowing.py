from functools import lru_cache
from pathlib import Path

from transformers import AutoTokenizer

from .ces_extractor import parse_line_numbers

GAP_MARKER = "// ..."


class WindowingConfig:
    def __init__(
        self,
        full_text_mode="full",
        full_window_context_lines=20,
        full_window_max_lines=80,
        full_window_token_budget=0,
        ces_token_budget=0,
    ):
        self.full_text_mode = full_text_mode
        self.full_window_context_lines = full_window_context_lines
        self.full_window_max_lines = full_window_max_lines
        self.full_window_token_budget = full_window_token_budget
        self.ces_token_budget = ces_token_budget


class TokenBudgetWindowBuilder:
    def __init__(self, codebert_path, config):
        self.tokenizer = AutoTokenizer.from_pretrained(str(codebert_path), local_files_only=True)
        self.config = config
        self.gap_token_cost = max(1, len(self.tokenizer.encode(GAP_MARKER, add_special_tokens=False)))

    @lru_cache(maxsize=200000)
    def _line_token_cost(self, line):
        return max(1, len(self.tokenizer.encode(line, add_special_tokens=False)))

    def build_full_pair_text(
        self,
        func_before,
        func_after,
        delete_lines,
        add_lines,
        max_seq_length_full,
    ):
        if str(self.config.full_text_mode).lower() != "changed_window":
            return func_before + "\n</s>\n" + func_after
        sep_budget = 8
        total_budget = int(self.config.full_window_token_budget or max(max_seq_length_full - sep_budget, 32))
        side_budget = max(total_budget // 2, 16)
        before_text = self.build_changed_window_text(
            func_before,
            delete_lines,
            token_budget=side_budget,
            max_lines=self.config.full_window_max_lines,
            context_lines=self.config.full_window_context_lines,
        )
        after_text = self.build_changed_window_text(
            func_after,
            add_lines,
            token_budget=side_budget,
            max_lines=self.config.full_window_max_lines,
            context_lines=self.config.full_window_context_lines,
        )
        return before_text + "\n</s>\n" + after_text

    def clip_text_to_budget(self, text, token_budget):
        lines = text.splitlines()
        if not lines:
            return ""
        return self._build_window_from_lines(
            lines,
            anchor_lines=[max(1, (len(lines) + 1) // 2)],
            token_budget=max(token_budget, 16),
            max_lines=len(lines),
            context_lines=max(len(lines), 1),
        )

    def build_changed_window_text(
        self,
        code,
        changed_lines,
        token_budget,
        max_lines,
        context_lines,
    ):
        lines = code.splitlines()
        if not lines:
            return ""
        change_ids = [line for line in parse_line_numbers(changed_lines) if 1 <= line <= len(lines)]
        if not change_ids:
            return self._build_prefix_window(lines, token_budget=token_budget, max_lines=max_lines)
        anchor_lines = []
        for change_id in change_ids:
            start = max(1, change_id - context_lines)
            end = min(len(lines), change_id + context_lines)
            anchor_lines.extend(range(start, end + 1))
        anchor_lines = sorted(set(anchor_lines))
        return self._build_window_from_lines(
            lines,
            anchor_lines=anchor_lines,
            token_budget=token_budget,
            max_lines=max_lines,
            context_lines=context_lines,
        )

    def _build_prefix_window(self, lines, token_budget, max_lines):
        selected = []
        running = 0
        for idx, line in enumerate(lines, start=1):
            cost = self._line_token_cost(line)
            if selected and (len(selected) >= max_lines or running + cost > token_budget):
                break
            selected.append(idx)
            running += cost
            if len(selected) >= max_lines:
                break
        return self._render_lines(lines, selected or [1])

    def _build_window_from_lines(
        self,
        lines,
        anchor_lines,
        token_budget,
        max_lines,
        context_lines,
    ):
        unique_anchors = [line for line in sorted(set(anchor_lines)) if 1 <= line <= len(lines)]
        if not unique_anchors:
            return self._build_prefix_window(lines, token_budget=token_budget, max_lines=max_lines)
        selected = list(unique_anchors[:max_lines])
        current_cost = self._estimate_render_cost(lines, selected)
        if current_cost > token_budget:
            center = unique_anchors[len(unique_anchors) // 2]
            ordered = sorted(unique_anchors, key=lambda line: (abs(line - center), line))
            trimmed = []
            for line in ordered:
                candidate = sorted(trimmed + [line])
                if len(candidate) > max_lines:
                    continue
                candidate_cost = self._estimate_render_cost(lines, candidate)
                if trimmed and candidate_cost > token_budget:
                    continue
                trimmed = candidate
                if candidate_cost >= token_budget:
                    break
            return self._render_lines(lines, trimmed or [center])
        chosen = set(selected)
        remaining = [
            line
            for line in range(1, len(lines) + 1)
            if line not in chosen and min(abs(line - anchor) for anchor in unique_anchors) <= context_lines
        ]
        remaining.sort(key=lambda line: (min(abs(line - anchor) for anchor in unique_anchors), line))
        for line in remaining:
            if len(chosen) >= max_lines:
                break
            candidate = sorted(chosen | {line})
            candidate_cost = self._estimate_render_cost(lines, candidate)
            if candidate_cost > token_budget:
                continue
            chosen.add(line)
            current_cost = candidate_cost
        return self._render_lines(lines, sorted(chosen))

    def _estimate_render_cost(self, lines, selected_lines):
        if not selected_lines:
            return 0
        cost = 0
        prev = None
        for line_no in selected_lines:
            if prev is not None and line_no != prev + 1:
                cost += self.gap_token_cost
            cost += self._line_token_cost(lines[line_no - 1])
            prev = line_no
        return cost

    @staticmethod
    def _render_lines(lines, selected_lines):
        parts = []
        prev = None
        for line_no in selected_lines:
            if not (1 <= line_no <= len(lines)):
                continue
            if prev is not None and line_no != prev + 1:
                parts.append(GAP_MARKER)
            parts.append(lines[line_no - 1])
            prev = line_no
        return "\n".join(parts)
