import argparse
import json
from pathlib import Path
import warnings

from torch.utils.data import DataLoader

from asva.data.collator import ASVACollator
from asva.data.dataset import ASVADataset
from asva.data.label_utils import load_label_encoders
from asva.models.asva_model import ASVAModel
from asva.training.optim import build_optimizer
from asva.training.trainer import Trainer
from asva.training.utils import configure_logging, ensure_dir, get_device, load_yaml, save_json


def main():
    warnings.filterwarnings(
        "ignore",
        message=r".*torch\.cpu\.amp\.autocast.*deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*nested tensors is in prototype stage.*",
        category=UserWarning,
    )
    parser = argparse.ArgumentParser(description="Evaluate ASVA")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    args = parser.parse_args()
    config = load_yaml(args.config)
    prepared_dir = Path(args.prepared_dir or config["paths"]["processed_dir"])
    metadata = json.loads((prepared_dir / "metadata.json").read_text(encoding="utf-8"))
    label_bundle = load_label_encoders(prepared_dir / "label_mappings.json")
    task_dims = {task: len(label_bundle.label_to_id[task]) for task in label_bundle.label_to_id}
    collator = ASVACollator(max_attack_lines=int(config["data"]["max_attack_lines"]), aux_feature_dim=int(metadata.get("aux_feature_dim", 0)))
    dataset = ASVADataset(prepared_dir, split=args.split, feature_mode=config["data"]["feature_mode"], max_graph_nodes=int(config["data"]["max_graph_nodes"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collator,
    )
    device = get_device(config["runtime"]["device"])
    model = ASVAModel(
        config,
        task_dims=task_dims,
        aux_feature_dim=int(metadata.get("aux_feature_dim", 0)),
    ).to(device)
    optimizer = build_optimizer(model, config)
    logger = configure_logging(config["paths"]["log_dir"], name="evaluate")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        config=config,
        device=device,
        logger=logger,
        checkpoint_dir=ensure_dir(config["paths"]["checkpoint_dir"]),
        log_dir=ensure_dir(config["paths"]["log_dir"]),
    )
    trainer.load_checkpoint(args.checkpoint, load_optimizer=False, load_scheduler=False)
    metrics = trainer.evaluate(loader, split_name=args.split, save=True)
    save_json(metrics, Path(config["paths"]["log_dir"]) / f"{args.split}_metrics_snapshot.json")


if __name__ == "__main__":
    main()
