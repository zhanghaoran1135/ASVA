import math

import torch
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


def build_optimizer(model, config):
    train_cfg = config["training"]
    lr = float(train_cfg["learning_rate"])
    backbone_scale = float(train_cfg.get("backbone_lr_scale", 1.0))
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            backbone_params.append(param)
        else:
            head_params.append(param)
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": lr})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr * backbone_scale})
    return torch.optim.Adam(param_groups, lr=lr, weight_decay=float(train_cfg["weight_decay"]))


def build_scheduler(
    optimizer,
    config,
    steps_per_epoch,
):
    scheduler_cfg = config["training"].get("scheduler", {})
    if not scheduler_cfg.get("enabled", False):
        return None
    scheduler_type = str(scheduler_cfg.get("type", "cosine_with_warmup")).lower()
    if scheduler_type == "reduce_on_plateau":
        factor = float(scheduler_cfg.get("factor", 0.5))
        patience = int(scheduler_cfg.get("patience", 2))
        threshold = float(scheduler_cfg.get("threshold", 1e-3))
        cooldown = int(scheduler_cfg.get("cooldown", 0))
        min_lr_scale = float(scheduler_cfg.get("min_lr_scale", 0.1))
        min_lrs = [group["lr"] * min_lr_scale for group in optimizer.param_groups]
        return ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg.get("mode", "max")),
            factor=factor,
            patience=patience,
            threshold=threshold,
            cooldown=cooldown,
            min_lr=min_lrs,
            verbose=bool(scheduler_cfg.get("verbose", True)),
        )
    total_epochs = int(config["training"]["epochs"])
    total_steps = max(steps_per_epoch * total_epochs, 1)
    warmup_epochs = float(scheduler_cfg.get("warmup_epochs", 0))
    warmup_steps = int(round(warmup_epochs * steps_per_epoch))
    min_lr_scale = float(scheduler_cfg.get("min_lr_scale", 0.1))
    if scheduler_type != "cosine_with_warmup":
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return max(float(current_step + 1) / float(warmup_steps), 1e-6)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(current_step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return max(min_lr_scale, min_lr_scale + (1.0 - min_lr_scale) * cosine)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
