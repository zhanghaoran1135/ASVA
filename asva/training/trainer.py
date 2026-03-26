import json
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from asva.data.label_utils import TASK_COLUMNS

from .losses import MultiTaskLoss
from .metrics import compute_all_metrics, save_metrics


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        config,
        device,
        logger,
        checkpoint_dir,
        log_dir,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.logger = logger
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.criterion = MultiTaskLoss(config)
        self.scaler = GradScaler(enabled=bool(config["training"]["use_amp"] and device.type == "cuda"))
        self.scheduler_on_metric = isinstance(scheduler, ReduceLROnPlateau)
        ema_cfg = config["training"].get("ema", {})
        self.ema_enabled = bool(ema_cfg.get("enabled", False))
        self.ema_decay = float(ema_cfg.get("decay", 0.999))
        self.ema_state = None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = []

    def _move_labels(self, batch):
        batch["labels"] = {task: tensor.to(self.device) for task, tensor in batch["labels"].items()}
        batch["line_numbers"] = batch["line_numbers"].to(self.device)
        batch["line_mask"] = batch["line_mask"].to(self.device)
        batch["cfp_targets"] = batch["cfp_targets"].to(self.device)
        batch["aux_features"] = batch["aux_features"].to(self.device)

    def _get_monitor_tasks(self):
        exclude = {str(task) for task in self.config["training"].get("monitor_exclude_tasks", [])}
        tasks = [task for task in TASK_COLUMNS if task not in exclude]
        if not tasks:
            raise ValueError("All tasks were excluded by training.monitor_exclude_tasks.")
        return tasks

    def _compute_monitor_score(self, valid_metrics, monitor_metric):
        tasks = self._get_monitor_tasks()
        values = [float(valid_metrics[task][monitor_metric]) for task in tasks]
        return float(sum(values) / max(len(values), 1)), tasks

    def train(
        self,
        train_loader,
        valid_loader,
        start_epoch=1,
        best_score=float("-inf"),
        best_epoch=-1,
        patience=0,
    ):
        train_cfg = self.config["training"]
        monitor_metric = str(train_cfg.get("monitor_metric", "f1")).lower()
        if monitor_metric not in {"f1", "mcc"}:
            raise ValueError(f"Unsupported monitor metric: {monitor_metric}")
        monitor_tasks = self._get_monitor_tasks()
        if train_cfg["freeze_codebert"]:
            self.model.freeze_codebert()
        for epoch in range(start_epoch, train_cfg["epochs"] + 1):
            if train_cfg["freeze_codebert"] and epoch > train_cfg.get("freeze_codebert_epochs", 0):
                self.model.unfreeze_codebert()
            train_loss = self._run_epoch(train_loader, training=True, epoch=epoch)
            valid_metrics = self.evaluate(valid_loader, split_name="valid", save=False)
            valid_f1 = float(valid_metrics["overall_average"]["f1"])
            valid_mcc_all = float(valid_metrics["overall_average"]["mcc"])
            score, active_monitor_tasks = self._compute_monitor_score(valid_metrics, monitor_metric)
            scheduler_cfg = train_cfg.get("scheduler", {})
            scheduler_start_epoch = int(scheduler_cfg.get("start_epoch", 1))
            if self.scheduler is not None and self.scheduler_on_metric and epoch >= scheduler_start_epoch:
                self.scheduler.step(score)
            current_lrs = [group["lr"] for group in self.optimizer.param_groups]
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_metrics": valid_metrics,
                "learning_rates": current_lrs,
                "monitor_metric": monitor_metric,
                "monitor_tasks": active_monitor_tasks,
                "monitor_score": score,
            }
            self.history.append(record)
            self.logger.info(
                "Epoch %s train_loss=%.4f valid_f1=%.4f valid_mcc_all=%.4f monitor_%s=%.4f",
                epoch,
                train_loss,
                valid_f1,
                valid_mcc_all,
                monitor_metric,
                score,
            )
            self._save_history()
            if score > best_score:
                best_score = score
                best_epoch = epoch
                patience = 0
                self.save_checkpoint(self.checkpoint_dir / "best.pt", epoch=epoch, metrics=valid_metrics)
            else:
                patience += 1
            if train_cfg.get("save_last", True):
                self.save_checkpoint(self.checkpoint_dir / "last.pt", epoch=epoch, metrics=valid_metrics)
            if patience >= train_cfg["early_stopping_patience"]:
                self.logger.info("Early stopping at epoch %s; best epoch was %s", epoch, best_epoch)
                break
        return {
            "best_epoch": best_epoch,
            f"best_valid_{monitor_metric}": best_score,
            "monitor_tasks": monitor_tasks,
            "last_epoch": min(train_cfg["epochs"], max(start_epoch - 1, best_epoch if best_epoch > 0 else 0)),
        }

    def _run_epoch(self, data_loader, training, epoch):
        self.model.train(training)
        total_loss = 0.0
        total_batches = 0
        grad_steps = int(self.config["training"]["gradient_accumulation_steps"])
        max_steps = self.config["training"].get("max_train_steps")
        iterator = tqdm(data_loader, desc=f"{'train' if training else 'eval'}-{epoch}", leave=False)
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(iterator, start=1):
            self._move_labels(batch)
            with autocast(enabled=bool(self.config["training"]["use_amp"] and self.device.type == "cuda")):
                outputs = self.model(batch)
                loss, loss_stats = self.criterion(outputs, batch, epoch=epoch)
            if not torch.isfinite(loss):
                self.logger.warning(
                    "Skipping non-finite loss at epoch=%s step=%s: %s",
                    epoch,
                    step,
                    loss_stats,
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue
            scaled_loss = loss / grad_steps
            if training:
                self.scaler.scale(scaled_loss).backward()
                if step % grad_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["max_grad_norm"])
                    previous_scale = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    optimizer_step_ran = self.scaler.get_scale() >= previous_scale
                    if optimizer_step_ran and self.ema_enabled:
                        self._update_ema()
                    if self.scheduler is not None and optimizer_step_ran and not self.scheduler_on_metric:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu())
            total_batches += 1
            iterator.set_postfix({"loss": f"{loss_stats['total_loss']:.4f}"})
            if training and max_steps and step >= max_steps:
                break
        return total_loss / max(total_batches, 1)

    @torch.no_grad()
    def evaluate(self, data_loader, split_name="test", save=True):
        return self._evaluate_with_current_weights(data_loader, split_name=split_name, save=save)

    def save_checkpoint(self, path, epoch, metrics):
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": self.config,
        }
        if self.ema_state is not None:
            payload["ema_state_dict"] = self.ema_state
        if self.scheduler is not None:
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(payload, Path(path))

    @staticmethod
    def _upgrade_legacy_state_dict(state_dict):
        upgraded = {}
        for key, value in state_dict.items():
            new_key = key
            if ".heads." in key and ".net.0." in key:
                new_key = key.replace(".net.0.", ".proj.")
            elif ".heads." in key and ".net.3." in key:
                new_key = key.replace(".net.3.", ".classifier.")
            upgraded[new_key] = value
        return upgraded

    def load_checkpoint(
        self,
        path,
        load_optimizer=True,
        load_scheduler=True,
    ):
        payload = torch.load(Path(path), map_location=self.device)
        model_state_dict = self._upgrade_legacy_state_dict(payload["model_state_dict"])
        self.model.load_state_dict(model_state_dict)
        if load_optimizer and "optimizer_state_dict" in payload:
            try:
                self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            except Exception as exc:
                self.logger.warning("Skipping optimizer state load for %s: %s", path, exc)
        if "ema_state_dict" in payload:
            self.ema_state = {key: value.to(self.device) for key, value in payload["ema_state_dict"].items()}
        if load_scheduler and self.scheduler is not None and "scheduler_state_dict" in payload:
            try:
                self.scheduler.load_state_dict(payload["scheduler_state_dict"])
            except Exception as exc:
                self.logger.warning("Skipping scheduler state load for %s: %s", path, exc)
        return payload

    def _save_history(self):
        (self.log_dir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")

    @staticmethod
    def resolve_resume_state(
        payload,
        history,
        monitor_metric,
        monitor_tasks,
    ):
        start_epoch = int(payload.get("epoch", 0)) + 1
        checkpoint_metrics = payload.get("metrics", {})
        try:
            default_score = float(sum(float(checkpoint_metrics[task][monitor_metric]) for task in monitor_tasks) / max(len(monitor_tasks), 1))
        except Exception:
            default_score = float("-inf")
        if history:
            scored_history = [
                (
                    int(record.get("epoch", -1)),
                    float(record.get("monitor_score", default_score)),
                )
                for record in history
            ]
            best_epoch, best_score = max(scored_history, key=lambda item: item[1])
        else:
            best_epoch = int(payload.get("epoch", -1))
            best_score = default_score
        last_epoch = int(payload.get("epoch", 0))
        if history:
            history = [record for record in history if int(record.get("epoch", 0)) <= last_epoch]
        return {
            "start_epoch": start_epoch,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "patience": 0,
            "history": history or [],
        }

    def _update_ema(self):
        state_dict = self.model.state_dict()
        if self.ema_state is None:
            self.ema_state = {key: value.detach().clone() for key, value in state_dict.items()}
            return
        for key, value in state_dict.items():
            if not torch.is_floating_point(value):
                self.ema_state[key] = value.detach().clone()
                continue
            self.ema_state[key].mul_(self.ema_decay).add_(value.detach(), alpha=1.0 - self.ema_decay)

    @torch.no_grad()
    def _evaluate_with_current_weights(self, data_loader, split_name="test", save=True):
        active_state = None
        if self.ema_enabled and self.ema_state is not None:
            active_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
            self.model.load_state_dict(self.ema_state, strict=False)
        self.model.eval()
        targets = {task: [] for task in TASK_COLUMNS}
        predictions = {task: [] for task in TASK_COLUMNS}
        for batch in tqdm(data_loader, desc=f"eval-{split_name}", leave=False):
            self._move_labels(batch)
            outputs = self.model(batch)
            for task in TASK_COLUMNS:
                logits = outputs["logits"][task]
                preds = torch.argmax(logits, dim=-1)
                predictions[task].extend(preds.detach().cpu().tolist())
                targets[task].extend(batch["labels"][task].detach().cpu().tolist())
        metrics = compute_all_metrics(targets, predictions)
        if save:
            save_metrics(metrics, self.log_dir, prefix=split_name)
        if active_state is not None:
            self.model.load_state_dict(active_state, strict=False)
        return metrics
