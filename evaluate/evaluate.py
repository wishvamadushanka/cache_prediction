import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from config.run_manifest import load_run_specs
from config.settings_loader import load_settings
from dataset.cache_dataset import CacheTraceDataset
from model.combined_lstm import CombinedLSTMModel

# -----------------------
# Config
# -----------------------
RUN_CONFIG_PATH = "./config/runs.json"
SETTINGS_PATH = "./config/settings.json"
CACHE_LABELS = ("L1D", "L1I", "LL")


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    metrics = {}
    for idx, label in enumerate(CACHE_LABELS):
        actual = y_true[:, idx]
        predicted = y_pred[:, idx]

        mse = np.mean((actual - predicted) ** 2)
        rmse = math.sqrt(mse)

        total_actual = float(np.sum(actual))
        total_predicted = float(np.sum(predicted))

        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")

        if total_actual == 0:
            rep = float("nan")
        else:
            rep = abs(total_actual - total_predicted) / total_actual * 100.0

        metrics[label] = {
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "total_actual": total_actual,
            "total_predicted": total_predicted,
            "rep": float(rep),
        }

    return metrics


def print_metrics(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))
    for label in CACHE_LABELS:
        values = metrics[label]
        print(
            f"{label}: "
            f"MSE={values['mse']:.4f} "
            f"RMSE={values['rmse']:.4f} "
            f"R2={values['r2']:.4f} "
            f"REP={values['rep']:.2f}% "
            f"TotalActual={values['total_actual']:.2f} "
            f"TotalPred={values['total_predicted']:.2f}"
        )


def write_window_predictions_csv(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "run_name",
        "program",
        "split",
        "db_path",
        "cores",
        "l1d_size",
        "l1i_size",
        "ll_size",
        "window_idx",
        "actual_l1d",
        "pred_l1d",
        "actual_l1i",
        "pred_l1i",
        "actual_ll",
        "pred_ll",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            values = [str(row[column]) for column in header]
            f.write(",".join(values) + "\n")


def main():
    settings = load_settings(SETTINGS_PATH)
    paths = settings["paths"]
    model_settings = settings["model"]
    test_settings = settings["test"]

    tokenizer_path = paths["tokenizer_path"]
    model_path = paths["model_path"]
    output_dir = paths["evaluation_output_dir"]
    eval_split = test_settings["split"]
    device = settings["device"]

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    run_specs = load_run_specs(
        RUN_CONFIG_PATH,
        split=eval_split,
        max_rows_override=test_settings.get("max_rows_override"),
    )

    dataset = CacheTraceDataset(
        runs=run_specs,
        tokenizer=tokenizer,
        sequence_length=test_settings["sequence_length"],
        max_token_length=test_settings["max_token_length"],
    )

    loader = DataLoader(dataset, batch_size=test_settings["batch_size"], shuffle=False)

    print(
        f"Evaluating {len(run_specs)} {eval_split} runs | "
        f"{len(dataset)} windows | "
        f"{len(loader)} batches"
    )

    model = CombinedLSTMModel(
        token_vocab_size=tokenizer.vocab_size,
        token_embedding_dim=model_settings["token_embedding_dim"],
        access_feature_size=11,
        hidden_dim=model_settings["hidden_dim"],
        output_dim=model_settings["output_dim"],
        num_layers=model_settings["num_layers"],
        dropout=model_settings["dropout"],
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_targets = []
    all_predictions = []
    run_targets = {}
    run_predictions = {}
    prediction_rows = []

    sample_index = 0
    with torch.no_grad():
        for token_ids, access_feats, targets in loader:
            token_ids = token_ids.to(device)
            access_feats = access_feats.to(device)

            predictions = model(token_ids, access_feats).cpu().numpy()
            batch_targets = targets.cpu().numpy()

            all_predictions.append(predictions)
            all_targets.append(batch_targets)

            for row_idx in range(len(batch_targets)):
                metadata = dataset.get_sample_metadata(sample_index)
                run_name = metadata["run_name"]

                run_targets.setdefault(run_name, []).append(batch_targets[row_idx])
                run_predictions.setdefault(run_name, []).append(predictions[row_idx])

                prediction_rows.append(
                    {
                        "run_name": run_name,
                        "program": metadata["program"] or "",
                        "split": metadata["split"] or "",
                        "db_path": metadata["db_path"],
                        "cores": metadata["cores"] if metadata["cores"] is not None else "",
                        "l1d_size": metadata["l1d_size"],
                        "l1i_size": metadata["l1i_size"],
                        "ll_size": metadata["ll_size"],
                        "window_idx": metadata["window_idx"],
                        "actual_l1d": float(batch_targets[row_idx][0]),
                        "pred_l1d": float(predictions[row_idx][0]),
                        "actual_l1i": float(batch_targets[row_idx][1]),
                        "pred_l1i": float(predictions[row_idx][1]),
                        "actual_ll": float(batch_targets[row_idx][2]),
                        "pred_ll": float(predictions[row_idx][2]),
                    }
                )
                sample_index += 1

    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    overall_metrics = compute_metrics(all_targets, all_predictions)
    print_metrics("Overall Metrics", overall_metrics)

    for run_name in sorted(run_targets):
        metrics = compute_metrics(run_targets[run_name], run_predictions[run_name])
        print_metrics(f"Run Metrics: {run_name}", metrics)

    output_path = Path(output_dir) / f"window_predictions_{eval_split or 'all'}.csv"
    write_window_predictions_csv(output_path, prediction_rows)
    print(f"\nSaved per-window predictions to {output_path}")

    dataset.close()


if __name__ == "__main__":
    main()
