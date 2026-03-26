import csv
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_ROOT / "data" / "dataset" / "MSR_data_cleaned.csv"
METRIC_CSV = PROJECT_ROOT / "data" / "dataset" / "metric.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "dataset" / "Big-vul.csv"

KEEP_COLUMNS = [
    "id",
    "file_name",
    "commit_id",
    "func_before",
    "func_after",
    "delete_lines",
    "add_lines",
    "cvss2_AV",
    "cvss2_AC",
    "cvss2_AU",
    "cvss2_C",
    "cvss2_I",
    "cvss2_A",
    "cvss2_severity",
    "partition",
]

REQUIRED_NON_EMPTY_COLUMNS = [
    "func_before",
    "func_after",
    "cvss2_AV",
    "cvss2_AC",
    "cvss2_AU",
    "cvss2_C",
    "cvss2_I",
    "cvss2_A",
    "cvss2_severity",
]
LABEL_COLUMNS = [
    "cvss2_AV",
    "cvss2_AC",
    "cvss2_AU",
    "cvss2_C",
    "cvss2_I",
    "cvss2_A",
    "cvss2_severity",
]


def has_meaningful_code_change(row):
    before = row["func_before"].strip()
    after = row["func_after"].strip()
    return before != after


def has_valid_labels(row):
    return all(clean_text(row.get(column)) != "-1" for column in LABEL_COLUMNS)

def clean_text(value):
    text = "" if value is None else str(value)
    if text.lower() == "nan":
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def build_partition_key(row):
    pieces = [
        clean_text(row.get("") or row.get("Unnamed: 0")),
        clean_text(row.get("CVE ID")),
        clean_text(row.get("commit_id")),
        clean_text(row.get("file_name")),
    ]
    return "|".join(pieces)


def assign_partition():
    bucket = random.randint(0, 10)
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "valid"
    return "test"


def load_metric_labels(path):
    labels_by_id = {}
    for row in read_rows(path):
        source_id = clean_text(row.get("Unnamed: 0") or row.get("id"))
        if not source_id:
            continue
        labels_by_id[source_id] = {
            "cvss2_severity": clean_text(row.get("label")),
            "cvss2_AV": clean_text(row.get("AV")),
            "cvss2_AC": clean_text(row.get("AC")),
            "cvss2_AU": clean_text(row.get("AU")),
            "cvss2_C": clean_text(row.get("C")),
            "cvss2_I": clean_text(row.get("I")),
            "cvss2_A": clean_text(row.get("A")),
        }
    return labels_by_id


def build_output_row(row, row_index, labels_by_id):
    source_id = clean_text(row.get("") or row.get("Unnamed: 0"))
    row_id = source_id if source_id else str(row_index)
    metric_labels = labels_by_id.get(row_id, {})
    output = {
        "id": row_id,
        "file_name": clean_text(row.get("file_name")),
        "commit_id": clean_text(row.get("commit_id")),
        "func_before": clean_text(row.get("func_before")),
        "func_after": clean_text(row.get("func_after")),
        "delete_lines": clean_text(row.get("del_lines")),
        "add_lines": clean_text(row.get("add_lines")),
        "cvss2_AV": metric_labels.get("cvss2_AV", ""),
        "cvss2_AC": metric_labels.get("cvss2_AC", ""),
        "cvss2_AU": metric_labels.get("cvss2_AU", ""),
        "cvss2_C": metric_labels.get("cvss2_C", ""),
        "cvss2_I": metric_labels.get("cvss2_I", ""),
        "cvss2_A": metric_labels.get("cvss2_A", ""),
        "cvss2_severity": metric_labels.get("cvss2_severity", ""),
        "partition": assign_partition(),
    }
    return output


def read_rows(path):
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit = limit // 10
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    if not METRIC_CSV.exists():
        raise FileNotFoundError(f"Metric CSV not found: {METRIC_CSV}")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    labels_by_id = load_metric_labels(METRIC_CSV)
    total_rows = 0
    kept_rows = 0
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=KEEP_COLUMNS)
        writer.writeheader()
        for row_index, row in enumerate(read_rows(INPUT_CSV), start=1):
            output_row = build_output_row(row, row_index, labels_by_id)
            total_rows += 1
            if any(not output_row[column] for column in REQUIRED_NON_EMPTY_COLUMNS):
                continue
            if not has_valid_labels(output_row):
                continue
            if not has_meaningful_code_change(output_row):
                continue
            writer.writerow(output_row)
            kept_rows += 1
    print(f"Processed {total_rows} rows and wrote {kept_rows} valid rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
