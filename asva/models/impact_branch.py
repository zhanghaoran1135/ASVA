import torch
import torch.nn as nn


DEPENDENCY_TYPES = ("CONTROL", "DATA")


class DenseGATLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(input_dim, hidden_dim * heads, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, hidden_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x, adjacency):
        node_count = x.size(0)
        h = self.linear(x).view(node_count, self.heads, self.hidden_dim)
        src_scores = (h * self.attn_src.unsqueeze(0)).sum(dim=-1)
        dst_scores = (h * self.attn_dst.unsqueeze(0)).sum(dim=-1)
        logits = src_scores.unsqueeze(1) + dst_scores.unsqueeze(0)
        logits = self.activation(logits.permute(2, 0, 1))
        mask = adjacency.unsqueeze(0).expand(self.heads, -1, -1)
        logits = logits.masked_fill(~mask, -1e4)
        alpha = torch.softmax(logits, dim=-1)
        alpha = self.dropout(alpha)
        out = torch.einsum("hij,jhf->ihf", alpha, h)
        return out.mean(dim=1)


class TypedDenseGATLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, dropout):
        super().__init__()
        self.self_proj = nn.Linear(input_dim, hidden_dim)
        self.type_layers = nn.ModuleDict(
            {edge_type: DenseGATLayer(input_dim, hidden_dim, heads, dropout) for edge_type in DEPENDENCY_TYPES}
        )
        self.activation = nn.Tanh()

    def forward(self, x, adjacencies):
        outputs = [self.self_proj(x)]
        for edge_type, layer in self.type_layers.items():
            adjacency = adjacencies[edge_type]
            if adjacency.any():
                outputs.append(layer(x, adjacency))
        combined = torch.stack(outputs, dim=0).mean(dim=0)
        return self.activation(combined)


class ImpactBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        gat_layers = []
        current_dim = input_dim
        for _ in range(layers):
            gat_layers.append(TypedDenseGATLayer(current_dim, hidden_dim, heads, dropout))
            current_dim = hidden_dim
        self.gat_layers = nn.ModuleList(gat_layers)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _distance_weights(
        line_numbers,
        changed_line_numbers,
        device,
    ):
        if not line_numbers:
            return torch.zeros(0, device=device)
        if not changed_line_numbers:
            return torch.ones(len(line_numbers), device=device)
        node_lines = torch.tensor(line_numbers, dtype=torch.float32, device=device)
        changed = torch.tensor(sorted(set(changed_line_numbers)), dtype=torch.float32, device=device)
        distances = torch.abs(node_lines.unsqueeze(1) - changed.unsqueeze(0)).min(dim=1).values
        weights = torch.exp(-distances / 8.0)
        exact_match = (distances == 0).float()
        return (weights + exact_match).clamp(min=1e-3)

    @staticmethod
    def _adjacencies_from_edges(
        node_count,
        edge_index,
        edge_types,
        device,
    ):
        adjacencies = {
            edge_type: torch.eye(node_count, dtype=torch.bool, device=device)
            for edge_type in DEPENDENCY_TYPES
        }
        for (src, dst), edge_type in zip(edge_index, edge_types):
            coarse_type = edge_type if edge_type in DEPENDENCY_TYPES else "CONTROL"
            adjacencies[coarse_type][src, dst] = True
        return adjacencies

    def forward(
        self,
        graph_node_embeddings,
        edge_indices,
        edge_types,
        graph_line_numbers,
        changed_line_numbers,
    ):
        if not graph_node_embeddings:
            raise ValueError("graph_node_embeddings must not be empty")
        pooled_vectors = []
        for node_embeddings, edge_index, graph_edge_types, node_line_numbers, changed_lines in zip(
            graph_node_embeddings,
            edge_indices,
            edge_types,
            graph_line_numbers,
            changed_line_numbers,
        ):
            if node_embeddings.numel() == 0:
                pooled_vectors.append(torch.zeros(self.hidden_dim * 2, device=node_embeddings.device))
                continue
            adjacencies = self._adjacencies_from_edges(
                node_embeddings.size(0),
                edge_index,
                graph_edge_types,
                node_embeddings.device,
            )
            x = node_embeddings
            for layer in self.gat_layers:
                x = layer(x, adjacencies)
            weights = self._distance_weights(node_line_numbers[: x.size(0)], changed_lines, x.device)
            if weights.numel() == x.size(0):
                weighted_mean = (x * weights.unsqueeze(1)).sum(dim=0) / weights.sum().clamp(min=1e-6)
            else:
                weighted_mean = x.mean(dim=0)
            pooled_vectors.append(torch.cat([weighted_mean, x.max(dim=0).values], dim=0))
        stacked = torch.stack(pooled_vectors, dim=0)
        return self.out_proj(stacked)
