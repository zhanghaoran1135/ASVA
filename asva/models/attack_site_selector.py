import torch
import torch.nn as nn


class AttackSiteSelector(nn.Module):
    def __init__(
        self,
        hidden_size,
        theta,
        threshold_mode="fixed",
        empty_selection_policy="max_score",
        top_k_fallback=1,
    ):
        super().__init__()
        self.theta = float(theta)
        self.threshold_mode = threshold_mode
        self.empty_selection_policy = empty_selection_policy
        self.top_k_fallback = int(top_k_fallback)
        self.score_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        line_embeddings,
        line_mask,
        line_texts=None,
        line_numbers=None,
    ):
        score_logits = self.score_head(line_embeddings).squeeze(-1)
        score_probs = torch.sigmoid(score_logits) * line_mask.float()
        hard_mask = self._threshold_mask(score_probs, line_mask)
        selected = self._pack_selected(line_embeddings, hard_mask, line_texts=line_texts, line_numbers=line_numbers)
        self_targets = ((score_probs > self.theta) & line_mask).float()
        return {
            "score_logits": score_logits,
            "score_probs": score_probs,
            "hard_mask": hard_mask,
            "self_targets": self_targets,
            **selected,
        }

    def _threshold_mask(self, score_probs, line_mask):
        if self.threshold_mode != "fixed":
            raise ValueError(f"Unsupported threshold mode: {self.threshold_mode}")
        hard_mask = (score_probs > self.theta) & line_mask
        for batch_idx in range(hard_mask.size(0)):
            if not line_mask[batch_idx].any():
                continue
            if hard_mask[batch_idx].any():
                continue
            valid_scores = score_probs[batch_idx].masked_fill(~line_mask[batch_idx], -1.0)
            if self.empty_selection_policy == "max_score":
                selected_idx = int(torch.argmax(valid_scores).item())
                hard_mask[batch_idx, selected_idx] = True
            elif self.empty_selection_policy == "top_k":
                top_k = min(self.top_k_fallback, int(line_mask[batch_idx].sum().item()))
                top_indices = torch.topk(valid_scores, k=top_k, dim=0).indices.tolist()
                for index in top_indices:
                    hard_mask[batch_idx, int(index)] = True
            else:
                raise ValueError(f"Unsupported empty selection policy: {self.empty_selection_policy}")
        return hard_mask

    def _pack_selected(
        self,
        line_embeddings,
        hard_mask,
        line_texts=None,
        line_numbers=None,
    ):
        batch_size, _, hidden_size = line_embeddings.shape
        selected_counts = hard_mask.sum(dim=1)
        max_selected = int(selected_counts.max().item()) if int(selected_counts.max().item()) > 0 else 1
        selected_embeddings = line_embeddings.new_zeros((batch_size, max_selected, hidden_size))
        selected_mask = torch.zeros(batch_size, max_selected, dtype=torch.bool, device=line_embeddings.device)
        selected_indices = []
        selected_texts = []
        selected_line_numbers = []
        for batch_idx in range(batch_size):
            indices = torch.nonzero(hard_mask[batch_idx], as_tuple=False).flatten()
            index_list = indices.tolist()
            selected_indices.append(index_list)
            if indices.numel() > 0:
                count = indices.numel()
                selected_embeddings[batch_idx, :count] = line_embeddings[batch_idx, indices]
                selected_mask[batch_idx, :count] = True
            if line_texts is not None:
                selected_texts.append([line_texts[batch_idx][idx] for idx in index_list])
            else:
                selected_texts.append([])
            if line_numbers is not None:
                selected_line_numbers.append([int(line_numbers[batch_idx, idx].item()) for idx in index_list])
            else:
                selected_line_numbers.append([])
        return {
            "selected_embeddings": selected_embeddings,
            "selected_mask": selected_mask,
            "selected_indices": selected_indices,
            "selected_texts": selected_texts,
            "selected_line_numbers": selected_line_numbers,
        }
