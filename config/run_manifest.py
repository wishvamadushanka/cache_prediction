import json

from dataset.cache_dataset import CacheRunConfig


def load_run_specs(config_path, split=None, max_rows_override=None):
    with open(config_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    run_specs = []
    for entry in manifest:
        if split is not None and entry.get("split") != split:
            continue

        run_specs.append(
            CacheRunConfig(
                db_path=entry["db_path"],
                l1d_size=entry["l1d_size"],
                l1i_size=entry["l1i_size"],
                ll_size=entry["ll_size"],
                name=entry.get("name"),
                program=entry.get("program"),
                split=entry.get("split"),
                cores=entry.get("cores"),
                instruction_column=entry.get(
                    "instruction_column", "preprocessed_instruction"
                ),
                fallback_instruction_column=entry.get(
                    "fallback_instruction_column", "disassembly_string"
                ),
                max_rows=max_rows_override,
            )
        )

    if not run_specs:
        split_text = "all splits" if split is None else f"split='{split}'"
        raise ValueError(f"No run specs found for {split_text} in {config_path}")

    return run_specs
