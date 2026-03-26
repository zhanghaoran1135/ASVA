import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from asva.data.collator import ASVACollator
from asva.data.dataset import ASVADataset
from asva.data.label_utils import compute_balanced_class_weights, compute_multitask_sample_weights, load_label_encoders
from asva.models.asva_model import ASVAModel
from asva.training.optim import build_optimizer, build_scheduler
from asva.training.trainer import Trainer
from asva.training.utils import configure_logging, ensure_dir, get_device, load_yaml, save_json, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train ASVA")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.max_train_steps is not None:
        config["training"]["max_train_steps"] = args.max_train_steps
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    prepared_dir = Path(args.prepared_dir or config["paths"]["processed_dir"])
    set_seed(int(config["runtime"]["seed"]))
    logger = configure_logging(config["paths"]["log_dir"], name="train")
    label_bundle = load_label_encoders(prepared_dir / "label_mappings.json")
    metadata = json.loads((prepared_dir / "metadata.json").read_text(encoding="utf-8"))
    task_dims = {task: len(label_bundle.label_to_id[task]) for task in label_bundle.label_to_id}
    aux_dim = int(metadata.get("aux_feature_dim", 0))
    collator = ASVACollator(max_attack_lines=int(config["data"]["max_attack_lines"]), aux_feature_dim=aux_dim)
    train_dataset = ASVADataset(prepared_dir, split="train", feature_mode=config["data"]["feature_mode"], max_graph_nodes=int(config["data"]["max_graph_nodes"]))
    valid_dataset = ASVADataset(prepared_dir, split="valid", feature_mode=config["data"]["feature_mode"], max_graph_nodes=int(config["data"]["max_graph_nodes"]))
    loss_cfg = config["training"].setdefault("loss", {})
    if loss_cfg.get("use_class_weights", True):
        loss_cfg["class_weights"] = compute_balanced_class_weights(
            train_dataset.records,
            task_dims,
            power=float(loss_cfg.get("class_weight_power", 0.5)),
            min_weight=float(loss_cfg.get("class_weight_min", 0.5)),
            max_weight=float(loss_cfg.get("class_weight_max", 4.0)),
        )
    sampler = None
    shuffle = True
    sampler_cfg = config["training"].get("sampler", {})
    if sampler_cfg.get("enabled", False):
        sampler_weights = compute_multitask_sample_weights(
            train_dataset.records,
            loss_cfg.get("class_weights", {}),
            tasks=list(sampler_cfg.get("tasks", [])) or None,
            power=float(sampler_cfg.get("power", 1.0)),
            min_weight=float(sampler_cfg.get("min_weight", 0.2)),
            max_weight=float(sampler_cfg.get("max_weight", 5.0)),
        )
        sampler = WeightedRandomSampler(
            weights=sampler_weights,
            num_samples=len(sampler_weights),
            replacement=bool(sampler_cfg.get("replacement", True)),
        )
        shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collator,
    )
    device = get_device(config["runtime"]["device"])
    model = ASVAModel(config, task_dims=task_dims, aux_feature_dim=aux_dim).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=max(len(train_loader), 1))
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        logger=logger,
        checkpoint_dir=ensure_dir(config["paths"]["checkpoint_dir"]),
        log_dir=ensure_dir(config["paths"]["log_dir"]),
    )
    resume_state = {"start_epoch": 1, "best_epoch": -1, "best_score": float("-inf"), "patience": 0, "history": []}
    if args.resume_checkpoint:
        payload = trainer.load_checkpoint(args.resume_checkpoint)
        history_path = Path(config["paths"]["log_dir"]) / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        monitor_metric = str(config["training"].get("monitor_metric", "f1")).lower()
        monitor_tasks = trainer._get_monitor_tasks()
        resume_state = trainer.resolve_resume_state(payload, history, monitor_metric, monitor_tasks)
        trainer.history = resume_state["history"]
        logger.info(
            "Resuming from %s at epoch %s; best_epoch=%s best_valid_%s=%.4f monitor_tasks=%s target_epochs=%s",
            args.resume_checkpoint,
            resume_state["start_epoch"],
            resume_state["best_epoch"],
            monitor_metric,
            resume_state["best_score"],
            monitor_tasks,
            config["training"]["epochs"],
        )
        if resume_state["start_epoch"] > int(config["training"]["epochs"]):
            raise ValueError(
                f"Resume epoch {resume_state['start_epoch']} exceeds configured total epochs {config['training']['epochs']}. "
                "Increase --epochs or training.epochs to continue."
            )
    summary = trainer.train(
        train_loader,
        valid_loader,
        start_epoch=resume_state["start_epoch"],
        best_score=resume_state["best_score"],
        best_epoch=resume_state["best_epoch"],
        patience=resume_state["patience"],
    )
    save_json(summary, Path(config["paths"]["log_dir"]) / "train_summary.json")


if __name__ == "__main__":
    main()
