import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from config.run_manifest import load_run_specs
from dataset.cache_dataset import CacheTraceDataset
from model.combined_lstm import CombinedLSTMModel

# from dataset.cache_dataset import CacheTraceDataset
# from model.combined_lstm import CombinedLSTMModel

# -----------------------
# Config
# -----------------------
TOKENIZER_PATH = "../DBs_Randika/trained_assembly_tokenizer/fast_tokenizer"
RUN_CONFIG_PATH = "./config/runs.json"

SEQ_LEN = 200
MAX_TOKEN_LEN = 15
BATCH_SIZE = 60
EPOCHS = 10
LR = 1e-3
HIDDEN_DIM = 330
DROPOUT = 0.05
# -----------------------
# Load tokenizer
# -----------------------
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
train_run_specs = load_run_specs(RUN_CONFIG_PATH, split="train")
val_run_specs = load_run_specs(RUN_CONFIG_PATH, split="val")

# -----------------------
# Dataset & Loader
# -----------------------
train_dataset = CacheTraceDataset(
    runs=train_run_specs,
    tokenizer=tokenizer,
    sequence_length=SEQ_LEN,
    max_token_length=MAX_TOKEN_LEN,
)
val_dataset = CacheTraceDataset(
    runs=val_run_specs,
    tokenizer=tokenizer,
    sequence_length=SEQ_LEN,
    max_token_length=MAX_TOKEN_LEN,
)

# print("Dataset length:", len(dataset))
# print("Dataset :", dataset)
# exit()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
    token_embedding_dim=15,
    access_feature_size=11,
    hidden_dim=HIDDEN_DIM,
    output_dim=3,
    num_layers=2,
    dropout=DROPOUT,
)

device = "cpu"
model.to(device)

# -----------------------
# Training setup
# -----------------------
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_val_loss = float("inf")

# -----------------------
# Training loop
# -----------------------
for epoch in range(EPOCHS):
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
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "combined_lstm.pt")
        print(f"Saved new best model with val loss {best_val_loss:.4f}")

train_dataset.close()
val_dataset.close()
