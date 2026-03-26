class CFPFallbackOutput:
    def __init__(self, line_texts, cfp_positive_pairs, cfp_source, line_numbers):
        self.line_texts = line_texts
        self.cfp_positive_pairs = cfp_positive_pairs
        self.cfp_source = cfp_source
        self.line_numbers = line_numbers


def build_fallback_cfp_pairs_for_selected_lines(line_texts, line_numbers):
    pairs = []
    for idx in range(len(line_texts) - 1):
        pairs.append([idx, idx + 1])
        pairs.append([idx + 1, idx])
    return CFPFallbackOutput(
        line_texts=line_texts,
        cfp_positive_pairs=pairs,
        cfp_source="fallback",
        line_numbers=line_numbers,
    )


def build_fallback_cfp_pairs(code, max_lines):
    line_texts = code.splitlines()[:max_lines]
    line_numbers = list(range(1, len(line_texts) + 1))
    return build_fallback_cfp_pairs_for_selected_lines(line_texts, line_numbers)
