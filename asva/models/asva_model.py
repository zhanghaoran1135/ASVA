import torch
import torch.nn as nn

from asva.data.label_utils import TASK_COLUMNS

from .attack_path import AttackPathBranch
from .codebert_encoder import CodeBERTEncoder
from .exploitability_branch import ExploitabilityBranch
from .impact_branch import ImpactBranch
from .multitask_heads import MultiTaskHeads


class ASVAModel(nn.Module):
    def __init__(
        self,
        config,
        task_dims,
        aux_feature_dim=0,
    ):
        super().__init__()
        self.config = config
        self.feature_mode = config["data"]["feature_mode"]
        memory_cfg = config.get("memory", {})
        self.encoder = CodeBERTEncoder(
            model_path=config["paths"]["codebert_path"],
            cache_dir=config["runtime"]["embedding_cache_dir"],
            use_embedding_cache=bool(config["runtime"]["use_embedding_cache"]),
            enable_gradient_checkpointing=bool(memory_cfg.get("gradient_checkpointing", False)),
        )
        hidden_size = self.encoder.hidden_size
        self.hidden_size = hidden_size
        self.line_encode_chunk_size = int(memory_cfg.get("line_encode_chunk_size", 0) or 0)
        self.graph_encode_chunk_size = int(memory_cfg.get("graph_encode_chunk_size", 0) or 0)
        self.detach_line_graph_encoder = bool(memory_cfg.get("detach_line_graph_encoder", False))
        model_cfg = config["model"]
        data_cfg = config["data"]
        self.use_raw_code = self.feature_mode != "precomputed_only"
        self.attack_branch = AttackPathBranch(
            hidden_size=hidden_size,
            max_attack_lines=data_cfg["max_attack_lines"],
            attention_heads=model_cfg["attack_attention_heads"],
            transformer_layers=model_cfg["attack_transformer_layers"],
            dropout=model_cfg["dropout"],
            selector_config=config["attack_site_selector"],
        )
        self.exploit_branch = ExploitabilityBranch(
            input_dim=hidden_size,
            conv_channels=model_cfg["conv_channels"],
            gru_hidden_dim=model_cfg["gru_hidden_dim"],
            attention_dim=model_cfg["exploit_attention_dim"],
            attack_dim=hidden_size,
            dropout=model_cfg["dropout"],
        )
        self.impact_branch = ImpactBranch(
            input_dim=hidden_size,
            hidden_dim=model_cfg["gat_hidden_dim"],
            heads=model_cfg["gat_heads"],
            layers=model_cfg["gat_layers"],
            dropout=model_cfg["gat_dropout"],
        )
        fusion_dim = model_cfg["gru_hidden_dim"] + model_cfg["gat_hidden_dim"]
        self.shared_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, model_cfg["hidden_dim"]),
            nn.LayerNorm(model_cfg["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(model_cfg["dropout"]),
        )
        fusion_dim += model_cfg["hidden_dim"]
        self.use_aux_features = aux_feature_dim > 0 and self.feature_mode != "raw_code_only"
        if self.use_aux_features:
            self.aux_proj = nn.Sequential(
                nn.LayerNorm(aux_feature_dim),
                nn.Linear(aux_feature_dim, model_cfg["hidden_dim"]),
                nn.GELU(),
                nn.Dropout(model_cfg["dropout"]),
                nn.Linear(model_cfg["hidden_dim"], model_cfg["hidden_dim"]),
                nn.GELU(),
            )
            fusion_dim += model_cfg["hidden_dim"]
        else:
            self.aux_proj = None
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(model_cfg["dropout"]),
        )
        self.heads = MultiTaskHeads(
            fused_dim=fusion_dim,
            task_dims=task_dims,
            dropout=model_cfg["dropout"],
            multi_sample_dropout_num=int(model_cfg.get("multi_sample_dropout_num", 1)),
            exploit_dim=model_cfg["gru_hidden_dim"],
            impact_dim=model_cfg["gat_hidden_dim"],
        )

    def freeze_codebert(self):
        self.encoder.freeze()

    def unfreeze_codebert(self):
        self.encoder.unfreeze()

    def _encode_lines(self, attack_line_texts, device):
        flat_lines = [line for lines in attack_line_texts for line in lines]
        max_lines = self.config["data"]["max_attack_lines"]
        hidden_size = self.hidden_size
        if not flat_lines:
            return torch.zeros(len(attack_line_texts), max_lines, hidden_size, device=device)
        pooled = self.encoder.encode_pooled_texts(
            flat_lines,
            self.config["data"]["max_line_length"],
            device,
            chunk_size=self.line_encode_chunk_size,
            force_no_grad=self.detach_line_graph_encoder,
        )
        line_embeddings = torch.zeros(len(attack_line_texts), max_lines, hidden_size, device=device)
        cursor = 0
        for batch_idx, lines in enumerate(attack_line_texts):
            count = min(len(lines), max_lines)
            if count > 0:
                line_embeddings[batch_idx, :count] = pooled[cursor : cursor + count]
            cursor += len(lines)
        return line_embeddings

    def _encode_graphs(self, graphs, device):
        flat_nodes = [text for graph in graphs for text in graph["node_texts"]]
        if not flat_nodes:
            return [torch.zeros(0, self.hidden_size, device=device) for _ in graphs]
        pooled = self.encoder.encode_pooled_texts(
            flat_nodes,
            self.config["data"]["max_graph_node_length"],
            device,
            chunk_size=self.graph_encode_chunk_size,
            force_no_grad=self.detach_line_graph_encoder,
        )
        outputs = []
        cursor = 0
        for graph in graphs:
            count = len(graph["node_texts"])
            outputs.append(pooled[cursor : cursor + count] if count > 0 else torch.zeros(0, self.hidden_size, device=device))
            cursor += count
        return outputs

    def _reencode_selected_lines(self, texts, device):
        return self.encoder.encode_pooled_texts(
            texts,
            self.config["data"]["max_line_length"],
            device,
            chunk_size=self.line_encode_chunk_size,
            force_no_grad=self.detach_line_graph_encoder,
        )

    def forward(self, batch):
        device = next(self.parameters()).device
        aux_losses = {}
        batch_size = len(batch["ids"])
        shared_vector = torch.zeros(batch_size, self.config["model"]["hidden_dim"], device=device)
        aux_vector = torch.zeros(batch_size, self.config["model"]["hidden_dim"], device=device)
        if self.use_raw_code:
            full_encoded = self.encoder.encode_texts(batch["full_pair_texts"], self.config["data"]["max_seq_length_full"], device)
            ces_encoded = self.encoder.encode_texts(batch["ces_pair_texts"], self.config["data"]["max_seq_length_ces"], device)
            shared_sequence = torch.cat([full_encoded["hidden_states"], ces_encoded["hidden_states"]], dim=1)
            shared_mask = torch.cat([full_encoded["attention_mask"], ces_encoded["attention_mask"]], dim=1).bool()
            shared_vector = self.shared_proj(torch.cat([full_encoded["pooled"], ces_encoded["pooled"]], dim=-1))
            line_embeddings = self._encode_lines(batch["attack_line_texts"], device)
            attack_outputs = self.attack_branch(
                line_embeddings,
                batch["line_mask"].to(device),
                line_texts=batch["attack_line_texts"],
                line_numbers=batch["line_numbers"].to(device),
                reencode_selected_fn=lambda texts: self._reencode_selected_lines(texts, device),
            )
            graph_embeddings = self._encode_graphs(batch["graphs"], device)
            impact_vector = self.impact_branch(
                graph_embeddings,
                [graph["edge_index"] for graph in batch["graphs"]],
                [graph["edge_types"] for graph in batch["graphs"]],
                [graph["line_numbers"] for graph in batch["graphs"]],
                batch["changed_line_numbers"],
            )
            exploit_vector = self.exploit_branch(shared_sequence, shared_mask, attack_outputs["attack_vector"])
            if self.config["auxiliary"]["use_mlm_loss"]:
                aux_losses["mlm_loss"] = self.encoder.compute_mlm_loss(
                    full_encoded["input_ids"],
                    full_encoded["attention_mask"],
                    mlm_probability=self.config["auxiliary"]["mlm_probability"],
                )
            aux_outputs = attack_outputs
        else:
            exploit_vector = torch.zeros(batch_size, self.config["model"]["gru_hidden_dim"], device=device)
            impact_vector = torch.zeros(batch_size, self.config["model"]["gat_hidden_dim"], device=device)
            aux_outputs = {
                "score_logits": torch.zeros(batch_size, self.config["data"]["max_attack_lines"], device=device),
                "score_probs": torch.zeros(batch_size, self.config["data"]["max_attack_lines"], device=device),
                "hard_mask": torch.zeros(batch_size, self.config["data"]["max_attack_lines"], device=device, dtype=torch.bool),
                "self_targets": torch.zeros(batch_size, self.config["data"]["max_attack_lines"], device=device),
                "pair_logits": torch.zeros(batch_size, self.config["data"]["max_attack_lines"], self.config["data"]["max_attack_lines"], device=device),
                "selected_indices": [[] for _ in range(batch_size)],
                "selected_texts": [[] for _ in range(batch_size)],
                "selected_line_numbers": [[] for _ in range(batch_size)],
            }
        if self.use_aux_features and self.aux_proj is not None:
            aux_vector = self.aux_proj(batch["aux_features"].to(device))
        fused_parts = [exploit_vector, impact_vector, shared_vector]
        if self.use_aux_features:
            fused_parts.append(aux_vector)
        fused = self.fusion_proj(torch.cat(fused_parts, dim=-1))
        logits = self.heads(fused, exploit_vector, impact_vector)
        return {
            "logits": logits,
            "aux_losses": aux_losses,
            "key_line_logits": aux_outputs["score_logits"],
            "key_line_probs": aux_outputs["score_probs"],
            "key_line_hard_mask": aux_outputs["hard_mask"],
            "key_line_self_targets": aux_outputs["self_targets"],
            "cfp_logits": aux_outputs["pair_logits"],
            "selected_attack_line_indices": aux_outputs["selected_indices"],
            "selected_attack_line_texts": aux_outputs["selected_texts"],
            "selected_attack_line_numbers": aux_outputs["selected_line_numbers"],
        }
