import argparse
import json
import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = PROJECT_ROOT / "cache_prediction"
DEFAULT_DB_DIR = PROJECT_ROOT / "db"
DEFAULT_OUTPUT = PIPELINE_ROOT / "config" / "runs.json"

TXT_PATTERN = re.compile(
    r"l1d_(?P<l1d>[^_]+)_l1i_(?P<l1i>[^_]+)_ll_(?P<ll>[^_]+)_(?P<program>.+)_(?P<param>[^_]+)\.txt"
)
DB_PATTERN = re.compile(r"Printing cache stats database to (?P<db>cache_stats_\d+\.db)")


def parse_size_to_bytes(size_text):
    size_text = size_text.strip().lower()
    if size_text.endswith("k"):
        return int(size_text[:-1]) * 1024
    if size_text.endswith("m"):
        return int(size_text[:-1]) * 1024 * 1024
    return int(size_text)


def infer_split(index, total):
    if total == 1:
        return "train"
    if index == total - 1:
        return "test"
    if index == total - 2:
        return "val"
    return "train"


def discover_runs(db_dir, default_max_rows=None, default_cores=1):
    discovered = []

    txt_files = sorted(db_dir.glob("*.txt"))
    for txt_file in txt_files:
        match = TXT_PATTERN.match(txt_file.name)
        if not match:
            continue

        text = txt_file.read_text(encoding="utf-8", errors="replace")
        db_match = DB_PATTERN.search(text)
        if not db_match:
            continue

        db_name = db_match.group("db")
        db_path = db_dir / db_name
        if not db_path.exists():
            continue

        l1d_text = match.group("l1d")
        l1i_text = match.group("l1i")
        ll_text = match.group("ll")
        program = match.group("program")
        param = match.group("param")

        discovered.append(
            {
                "name": f"{program}_{param}_{db_path.stem}",
                "db_path": os.path.relpath(db_path, PIPELINE_ROOT),
                "program": program,
                "split": "train",
                "l1d_size": parse_size_to_bytes(l1d_text),
                "l1i_size": parse_size_to_bytes(l1i_text),
                "ll_size": parse_size_to_bytes(ll_text),
                "cores": default_cores,
                "max_rows": default_max_rows,
            }
        )

    discovered.sort(
        key=lambda row: (
            row["l1d_size"],
            row["l1i_size"],
            row["ll_size"],
            row["name"],
        )
    )

    total = len(discovered)
    for idx, row in enumerate(discovered):
        row["split"] = infer_split(idx, total)

    return discovered


def main():
    parser = argparse.ArgumentParser(
        description="Generate runs.json from DB metadata files under the db directory."
    )
    parser.add_argument(
        "--db-dir",
        default=str(DEFAULT_DB_DIR),
        help="Directory containing cache_stats DBs and companion .txt metadata files.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output runs manifest path.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional max_rows value to assign to every generated run.",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Default core count to assign to every generated run.",
    )
    args = parser.parse_args()

    db_dir = Path(args.db_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    runs = discover_runs(
        db_dir=db_dir,
        default_max_rows=args.max_rows,
        default_cores=args.cores,
    )
    if not runs:
        raise SystemExit(f"No runs discovered from {db_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
        f.write("\n")

    print(f"Generated {len(runs)} runs in {output_path}")
    split_counts = {}
    for run in runs:
        split_counts[run["split"]] = split_counts.get(run["split"], 0) + 1
    print("Split counts:", split_counts)


if __name__ == "__main__":
    main()
