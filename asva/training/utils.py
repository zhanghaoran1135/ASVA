import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config_base_dir(path):
    config_path = Path(path).resolve()
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return Path.cwd().resolve()


def _resolve_path_values(data, base_dir):
    if isinstance(data, dict):
        resolved = {}
        for key, value in data.items():
            if isinstance(value, str) and (key.endswith("_dir") or key.endswith("_path") or key.endswith("_csv")):
                path = Path(value)
                resolved[key] = str(path if path.is_absolute() else (base_dir / path).resolve())
            else:
                resolved[key] = _resolve_path_values(value, base_dir)
        return resolved
    if isinstance(data, list):
        return [_resolve_path_values(item, base_dir) for item in data]
    return data


def load_yaml(path):
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return _resolve_path_values(data, _config_base_dir(config_path))


def get_project_root():
    return PROJECT_ROOT


def resolve_project_path(path):
    if path in (None, ""):
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def to_project_relative(path):
    if path in (None, ""):
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return candidate.as_posix()


def relativize_path_values(data):
    if isinstance(data, dict):
        relativized = {}
        for key, value in data.items():
            if isinstance(value, str) and (key.endswith("_dir") or key.endswith("_path") or key.endswith("_csv")):
                relativized[key] = to_project_relative(value)
            else:
                relativized[key] = relativize_path_values(value)
        return relativized
    if isinstance(data, list):
        return [relativize_path_values(item) for item in data]
    return data


def dump_yaml(data, path):
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def configure_logging(log_dir, name="asva"):
    log_dir = ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def save_json(data, path):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
