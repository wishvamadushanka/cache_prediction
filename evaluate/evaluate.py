import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from config.run_manifest import load_run_specs
from dataset.cache_dataset import CacheTraceDataset
from model.combined_lstm import CombinedLSTMModel

# -----------------------
# Config
# -----------------------
TOKENIZER_PATH = "../DBs_Randika/trained_assembly_tokenizer/fast_tokenizer"
RUN_CONFIG_PATH = "./config/runs.json"
MODEL_PATH = "./combined_lstm.pt"

EVAL_SPLIT = None
SEQ_LEN = 200
MAX_TOKEN_LEN = 15
BATCH_SIZE = 32

device = "cpu"
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


def main():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
    run_specs = load_run_specs(RUN_CONFIG_PATH, split=EVAL_SPLIT)

    dataset = CacheTraceDataset(
        runs=run_specs,
        tokenizer=tokenizer,
        sequence_length=SEQ_LEN,
        max_token_length=MAX_TOKEN_LEN,
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CombinedLSTMModel(
        token_vocab_size=tokenizer.vocab_size,
        token_embedding_dim=15,
        access_feature_size=11,
        hidden_dim=128,
        output_dim=3,
        num_layers=2,
        dropout=0.2,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_targets = []
    all_predictions = []
    run_targets = {}
    run_predictions = {}

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
                sample_index += 1

    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    overall_metrics = compute_metrics(all_targets, all_predictions)
    print_metrics("Overall Metrics", overall_metrics)

    for run_name in sorted(run_targets):
        metrics = compute_metrics(run_targets[run_name], run_predictions[run_name])
        print_metrics(f"Run Metrics: {run_name}", metrics)

    dataset.close()


if __name__ == "__main__":
    main()
