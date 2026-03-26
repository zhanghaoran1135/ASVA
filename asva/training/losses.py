import torch
import torch.nn as nn
import torch.nn.functional as F

from asva.data.label_utils import TASK_COLUMNS


class MultiTaskLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        loss_cfg = config["training"].get("loss", {})
        self.loss_type = str(loss_cfg.get("type", "cross_entropy"))
        self.focal_gamma = float(loss_cfg.get("focal_gamma", 1.5))
        self.label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
        self.task_weights = {task: float(weight) for task, weight in loss_cfg.get("task_weights", {}).items()}
        class_weight_cfg = loss_cfg.get("class_weights", {})
        self.class_weights = {
            task: torch.tensor(weights, dtype=torch.float32)
            for task, weights in class_weight_cfg.items()
        }
        self.logit_adjustment_tau = float(loss_cfg.get("logit_adjustment_tau", 0.0))
        self.class_priors = {}
        for task, weights in class_weight_cfg.items():
            inv = torch.tensor([1.0 / max(float(weight), 1e-8) for weight in weights], dtype=torch.float32)
            prior = inv / inv.sum().clamp(min=1e-8)
            self.class_priors[task] = prior

    def _aux_scale(self, name, epoch):
        aux_cfg = self.config["auxiliary"]
        total_epochs = int(self.config["training"]["epochs"])
        decay_cfg = aux_cfg.get("decay", {})
        if not decay_cfg.get("enabled", False):
            return float(aux_cfg[name])
        start_epoch = int(decay_cfg.get("start_epoch", max(1, total_epochs // 5)))
        final_scale = float(decay_cfg.get("final_scale", 0.2))
        if epoch <= start_epoch:
            return float(aux_cfg[name])
        progress = float(epoch - start_epoch) / float(max(total_epochs - start_epoch, 1))
        scale = 1.0 - progress * (1.0 - final_scale)
        return float(aux_cfg[name]) * max(final_scale, scale)

    def _classification_loss(self, task, logits, target):
        if self.logit_adjustment_tau > 0.0 and task in self.class_priors:
            priors = self.class_priors[task].to(logits.device).clamp(min=1e-8)
            logits = logits - self.logit_adjustment_tau * torch.log(priors).unsqueeze(0)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        class_weights = self.class_weights.get(task)
        if class_weights is not None:
            weight_tensor = class_weights.to(logits.device)
            sample_weights = weight_tensor.gather(0, target)
        else:
            sample_weights = torch.ones_like(target_probs)
        if self.label_smoothing > 0.0:
            smoothing = self.label_smoothing
            n_classes = logits.size(-1)
            smooth_loss = -log_probs.mean(dim=-1)
            nll_loss = -target_log_probs
            ce_loss = (1.0 - smoothing) * nll_loss + smoothing * smooth_loss
        else:
            ce_loss = -target_log_probs
        if self.loss_type == "weighted_focal":
            focal_factor = (1.0 - target_probs).clamp(min=1e-6).pow(self.focal_gamma)
            loss = ce_loss * focal_factor
        else:
            loss = ce_loss
        return (loss * sample_weights).mean()

    def forward(self, outputs, batch, epoch=1):
        device = next(iter(outputs["logits"].values())).device
        total_loss = torch.tensor(0.0, device=device)
        stats = {}
        for task in TASK_COLUMNS:
            logits = outputs["logits"][task]
            target = batch["labels"][task].to(device)
            loss = self._classification_loss(task, logits, target)
            task_weight = self.task_weights.get(task, 1.0)
            total_loss = total_loss + task_weight * loss
            stats[f"{task}_loss"] = float(loss.detach().cpu())
            stats[f"{task}_weight"] = float(task_weight)
        aux_cfg = self.config["auxiliary"]
        line_mask = batch["line_mask"].to(device)
        if aux_cfg["use_vuln_consistency_loss"]:
            eps = 1e-8
            probs = outputs["key_line_probs"].clamp(min=eps, max=1.0 - eps)
            targets = outputs["key_line_self_targets"].to(device)
            line_losses = -(targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs))
            denom = line_mask.float().sum().clamp(min=1.0)
            line_loss = (line_losses * line_mask.float()).sum() / denom
            vuln_weight = self._aux_scale("vuln_consistency_weight", epoch)
            total_loss = total_loss + vuln_weight * line_loss
            stats["l_vuln_loss"] = float(line_loss.detach().cpu())
            stats["l_vuln_weight"] = float(vuln_weight)
        if aux_cfg["use_cfp_loss"]:
            pair_mask = (line_mask.unsqueeze(1) & line_mask.unsqueeze(2)).float()
            cfp_losses = self.bce_loss(outputs["cfp_logits"], batch["cfp_targets"].to(device))
            denom = pair_mask.sum().clamp(min=1.0)
            cfp_loss = (cfp_losses * pair_mask).sum() / denom
            cfp_weight = self._aux_scale("cfp_weight", epoch)
            total_loss = total_loss + cfp_weight * cfp_loss
            stats["cfp_loss"] = float(cfp_loss.detach().cpu())
            stats["cfp_weight"] = float(cfp_weight)
        for name, value in outputs["aux_losses"].items():
            if name == "mlm_loss" and aux_cfg["use_mlm_loss"]:
                mlm_weight = self._aux_scale("mlm_weight", epoch)
                total_loss = total_loss + mlm_weight * value
                stats[name] = float(value.detach().cpu())
                stats["mlm_weight"] = float(mlm_weight)
        stats["total_loss"] = float(total_loss.detach().cpu())
        return total_loss, stats
