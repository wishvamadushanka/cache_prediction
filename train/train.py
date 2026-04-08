import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from dataset.cache_dataset import CacheRunConfig, CacheTraceDataset
from model.combined_lstm import CombinedLSTMModel

# from dataset.cache_dataset import CacheTraceDataset
# from model.combined_lstm import CombinedLSTMModel

# -----------------------
# Config
# -----------------------
TOKENIZER_PATH = "../DBs_Randika/trained_assembly_tokenizer/fast_tokenizer"

RUN_SPECS = [
    CacheRunConfig(
        db_path="../DBs_Randika/cache_stats_1769606772.db",
        l1d_size=1024,
        l1i_size=1024,
        ll_size=1024,
    ),
    CacheRunConfig(
        db_path="../DBs_Randika/cache_stats_1769606774.db",
        l1d_size=1024,
        l1i_size=1024,
        ll_size=1024,
    ),
    CacheRunConfig(
        db_path="../DBs_Randika/cache_stats_1769606778.db",
        l1d_size=1024,
        l1i_size=1024,
        ll_size=1024,
    ),
]

SEQ_LEN = 200
MAX_TOKEN_LEN = 15
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

# -----------------------
# Load tokenizer
# -----------------------
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)

# -----------------------
# Dataset & Loader
# -----------------------
dataset = CacheTraceDataset(
    runs=RUN_SPECS,
    tokenizer=tokenizer,
    sequence_length=SEQ_LEN,
    max_token_length=MAX_TOKEN_LEN,
)

# print("Dataset length:", len(dataset))
# print("Dataset :", dataset)
# exit()

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------
# Model
# -----------------------
model = CombinedLSTMModel(
    token_vocab_size=tokenizer.vocab_size,
    token_embedding_dim=15,
    access_feature_size=11,
    hidden_dim=128,
    output_dim=3,
    num_layers=2,
    dropout=0.2,
)

device = "cpu"
model.to(device)

# -----------------------
# Training setup
# -----------------------
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------
# Training loop
# -----------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for token_ids, access_feats, targets in loader:
        token_ids = token_ids.to(device)
        access_feats = access_feats.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(token_ids, access_feats)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {total_loss / len(loader):.4f}"
    )

torch.save(model.state_dict(), "combined_lstm.pt")
dataset.close()
