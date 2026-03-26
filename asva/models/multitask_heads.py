import torch
import torch.nn as nn


class TaskHead(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout,
        multi_sample_dropout_num=1,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.multi_sample_dropout_num = max(int(multi_sample_dropout_num), 1)

    def forward(self, x):
        hidden = self.activation(self.proj(x))
        logits = [self.classifier(self.dropout(hidden)) for _ in range(self.multi_sample_dropout_num)]
        return torch.stack(logits, dim=0).mean(dim=0)


class MultiTaskHeads(nn.Module):
    def __init__(
        self,
        fused_dim,
        task_dims,
        dropout,
        multi_sample_dropout_num=1,
        exploit_dim=0,
        impact_dim=0,
    ):
        super().__init__()
        self.tasks = [
            "cvss2_AV",
            "cvss2_AC",
            "cvss2_AU",
            "cvss2_C",
            "cvss2_I",
            "cvss2_A",
            "cvss2_severity",
        ]
        self.exploit_tasks = {"cvss2_AV", "cvss2_AC", "cvss2_AU"}
        self.impact_tasks = {"cvss2_C", "cvss2_I", "cvss2_A"}
        self.task_input_dims = {
            "cvss2_AV": fused_dim + exploit_dim,
            "cvss2_AC": fused_dim + exploit_dim,
            "cvss2_AU": fused_dim + exploit_dim,
            "cvss2_C": fused_dim + impact_dim,
            "cvss2_I": fused_dim + impact_dim,
            "cvss2_A": fused_dim + impact_dim,
            "cvss2_severity": fused_dim + exploit_dim + impact_dim,
        }
        self.heads = nn.ModuleDict(
            {
                task: TaskHead(
                    self.task_input_dims[task],
                    self.task_input_dims[task],
                    task_dims[task],
                    dropout,
                    multi_sample_dropout_num=multi_sample_dropout_num,
                )
                for task in self.tasks
            }
        )

    def forward(
        self,
        fused_features,
        exploit_features,
        impact_features,
    ):
        logits = {}
        for task in self.tasks:
            if task in self.exploit_tasks:
                task_input = torch.cat([fused_features, exploit_features], dim=-1)
            elif task in self.impact_tasks:
                task_input = torch.cat([fused_features, impact_features], dim=-1)
            else:
                task_input = torch.cat([fused_features, exploit_features, impact_features], dim=-1)
            logits[task] = self.heads[task](task_input)
        return logits
