import torch
import torch.nn as nn

from .attack_site_selector import AttackSiteSelector


class AttackPathBranch(nn.Module):
    def __init__(
        self,
        hidden_size,
        max_attack_lines,
        attention_heads,
        transformer_layers,
        dropout,
        selector_config,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_attack_lines = max_attack_lines
        self.selector = AttackSiteSelector(
            hidden_size=hidden_size,
            theta=float(selector_config["theta"]),
            threshold_mode=str(selector_config["threshold_mode"]),
            empty_selection_policy=str(selector_config["empty_selection_policy"]),
            top_k_fallback=int(selector_config["top_k_fallback"]),
        )
        self.cfp_bilinear = nn.Bilinear(hidden_size, hidden_size, 1)
        self.position_embeddings = nn.Embedding(max_attack_lines, hidden_size)
        self.pre_attn = nn.MultiheadAttention(hidden_size, attention_heads, dropout=dropout)
        self.position_norm = nn.LayerNorm(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dropout=dropout,
            dim_feedforward=hidden_size * 4,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.output_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        line_embeddings,
        line_mask,
        line_texts=None,
        line_numbers=None,
        reencode_selected_fn=None,
    ):
        selector_outputs = self.selector(
            line_embeddings,
            line_mask,
            line_texts=line_texts,
            line_numbers=line_numbers,
        )
        pair_logits = self._pairwise_logits(line_embeddings, line_mask)
        selected_embeddings = selector_outputs["selected_embeddings"]
        selected_mask = selector_outputs["selected_mask"]
        if reencode_selected_fn is not None and selector_outputs["selected_texts"]:
            selected_embeddings = self._reencode_selected(
                selector_outputs["selected_texts"],
                selected_mask,
                hidden_size=line_embeddings.size(-1),
                device=line_embeddings.device,
                reencode_selected_fn=reencode_selected_fn,
            )
        h_t = selected_embeddings.transpose(0, 1)
        attn_out, _ = self.pre_attn(h_t, h_t, h_t, key_padding_mask=~selected_mask)
        attention_repr = attn_out.transpose(0, 1)
        positions = torch.arange(attention_repr.size(1), device=line_embeddings.device).unsqueeze(0).expand(attention_repr.size(0), -1)
        relative_repr = self.position_norm(attention_repr + self.position_embeddings(positions))
        transformed = self.transformer(relative_repr.transpose(0, 1), src_key_padding_mask=~selected_mask).transpose(0, 1)
        masked = transformed.masked_fill(~selected_mask.unsqueeze(-1), float("-inf"))
        attack_vector = torch.amax(masked, dim=1)
        attack_vector = torch.where(torch.isinf(attack_vector), torch.zeros_like(attack_vector), attack_vector)
        attack_vector = self.output_proj(attack_vector)
        return {
            "score_logits": selector_outputs["score_logits"],
            "score_probs": selector_outputs["score_probs"],
            "hard_mask": selector_outputs["hard_mask"],
            "self_targets": selector_outputs["self_targets"],
            "pair_logits": pair_logits,
            "attack_vector": attack_vector,
            "selected_mask": selected_mask,
            "selected_indices": selector_outputs["selected_indices"],
            "selected_texts": selector_outputs["selected_texts"],
            "selected_line_numbers": selector_outputs["selected_line_numbers"],
        }

    @staticmethod
    def _reencode_selected(
        selected_texts,
        selected_mask,
        hidden_size,
        device,
        reencode_selected_fn,
    ):
        flat_selected = [line for lines in selected_texts for line in lines]
        if not flat_selected:
            return torch.zeros(selected_mask.size(0), selected_mask.size(1), hidden_size, device=device)
        pooled = reencode_selected_fn(flat_selected)
        reencoded = torch.zeros(selected_mask.size(0), selected_mask.size(1), hidden_size, device=device)
        cursor = 0
        for batch_idx, lines in enumerate(selected_texts):
            count = len(lines)
            if count > 0:
                reencoded[batch_idx, :count] = pooled[cursor : cursor + count]
            cursor += count
        return reencoded

    def _pairwise_logits(self, line_embeddings, line_mask):
        batch_size, line_count, _ = line_embeddings.shape
        left = line_embeddings.unsqueeze(2).expand(batch_size, line_count, line_count, -1)
        right = line_embeddings.unsqueeze(1).expand(batch_size, line_count, line_count, -1)
        logits = self.cfp_bilinear(left.reshape(-1, self.hidden_size), right.reshape(-1, self.hidden_size))
        logits = logits.view(batch_size, line_count, line_count)
        pair_mask = line_mask.unsqueeze(1) & line_mask.unsqueeze(2)
        return logits.masked_fill(~pair_mask, 0.0)
