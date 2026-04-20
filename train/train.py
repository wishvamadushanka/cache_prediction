import torch
from torch.utils.data import DataLoader, Subset
from transformers import PreTrainedTokenizerFast

from config.run_manifest import load_run_specs
from config.settings_loader import load_settings
from dataset.cache_dataset import CacheTraceDataset
from model.combined_lstm import CombinedLSTMModel

# from dataset.cache_dataset import CacheTraceDataset
# from model.combined_lstm import CombinedLSTMModel

# -----------------------
# Config
# -----------------------
RUN_CONFIG_PATH = "./config/runs.json"
SETTINGS_PATH = "./config/settings.json"
# -----------------------
# Load tokenizer
# -----------------------
settings = load_settings(SETTINGS_PATH)
paths = settings["paths"]
model_settings = settings["model"]
train_settings = settings["train"]
val_settings = settings["val"]

TOKENIZER_PATH = paths["tokenizer_path"]
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
seen_run_specs = load_run_specs(RUN_CONFIG_PATH, split=train_settings["split"])

# -----------------------
# Dataset & Loader
# -----------------------
seen_dataset = CacheTraceDataset(
    runs=seen_run_specs,
    tokenizer=tokenizer,
    sequence_length=train_settings["sequence_length"],
    max_token_length=train_settings["max_token_length"],
)
if train_settings["sequence_length"] != val_settings["sequence_length"]:
    raise ValueError("Train and val sequence lengths must match for seen-pool splitting.")
if train_settings["max_token_length"] != val_settings["max_token_length"]:
    raise ValueError("Train and val max token lengths must match for seen-pool splitting.")

val_ratio = float(train_settings.get("val_ratio", 0.1))
split_seed = int(train_settings.get("split_seed", 42))

if not 0.0 < val_ratio < 1.0:
    raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

total_seen_windows = len(seen_dataset)
if total_seen_windows < 2:
    raise ValueError(
        "Seen dataset must contain at least 2 windows to create train/val splits."
    )

val_window_count = max(1, int(round(total_seen_windows * val_ratio)))
if val_window_count >= total_seen_windows:
    val_window_count = total_seen_windows - 1
train_window_count = total_seen_windows - val_window_count

generator = torch.Generator().manual_seed(split_seed)
permutation = torch.randperm(total_seen_windows, generator=generator).tolist()
train_indices = permutation[:train_window_count]
val_indices = permutation[train_window_count:]

train_dataset = Subset(seen_dataset, train_indices)
val_dataset = Subset(seen_dataset, val_indices)

# print("Dataset length:", len(dataset))
# print("Dataset :", dataset)
# exit()

train_loader = DataLoader(
    train_dataset, batch_size=train_settings["batch_size"], shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=val_settings["batch_size"], shuffle=False
)

print(
    f"Loaded {len(seen_run_specs)} seen runs for train/val splitting | "
    f"{len(train_dataset)} train windows / {len(val_dataset)} val windows | "
    f"{len(train_loader)} train batches / {len(val_loader)} val batches"
)

# -----------------------
# Model
# -----------------------
model = CombinedLSTMModel(
    token_vocab_size=tokenizer.vocab_size,
    token_embedding_dim=model_settings["token_embedding_dim"],
    access_feature_size=11,
    hidden_dim=model_settings["hidden_dim"],
    output_dim=model_settings["output_dim"],
    num_layers=model_settings["num_layers"],
    dropout=model_settings["dropout"],
)

device = settings["device"]
model.to(device)

# -----------------------
# Training setup
# -----------------------
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_settings["learning_rate"])
best_val_loss = float("inf")

# -----------------------
# Training loop
# -----------------------
for epoch in range(train_settings["epochs"]):
    model.train()
    total_train_loss = 0

    for token_ids, access_feats, targets in train_loader:
        token_ids = token_ids.to(device)
        access_feats = access_feats.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(token_ids, access_feats)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for token_ids, access_feats, targets in val_loader:
            token_ids = token_ids.to(device)
            access_feats = access_feats.to(device)
            targets = targets.to(device)

            preds = model(token_ids, access_feats)
            loss = criterion(preds, targets)
            total_val_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)

    print(
        f"Epoch {epoch+1}/{train_settings['epochs']} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), paths["model_path"])
        print(f"Saved new best model with val loss {best_val_loss:.4f}")

seen_dataset.close()
