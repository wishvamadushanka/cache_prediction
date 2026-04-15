import torch
from torch.utils.data import DataLoader
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
train_run_specs = load_run_specs(RUN_CONFIG_PATH, split=train_settings["split"])
val_run_specs = load_run_specs(RUN_CONFIG_PATH, split=val_settings["split"])

# -----------------------
# Dataset & Loader
# -----------------------
train_dataset = CacheTraceDataset(
    runs=train_run_specs,
    tokenizer=tokenizer,
    sequence_length=train_settings["sequence_length"],
    max_token_length=train_settings["max_token_length"],
)
val_dataset = CacheTraceDataset(
    runs=val_run_specs,
    tokenizer=tokenizer,
    sequence_length=val_settings["sequence_length"],
    max_token_length=val_settings["max_token_length"],
)

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
    f"Loaded {len(train_run_specs)} train runs and {len(val_run_specs)} val runs | "
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

train_dataset.close()
val_dataset.close()
