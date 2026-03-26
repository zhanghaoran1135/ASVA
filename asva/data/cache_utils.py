import hashlib
import json
import pickle
from pathlib import Path


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_hash(parts):
    joined = "||".join("" if part is None else str(part) for part in parts)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def dump_json(data, path):
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_jsonl(rows, path):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_pickle(data, path):
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        pickle.dump(data, handle)


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)
