import torch
from transformers import PreTrainedTokenizerFast
from dataset.cache_dataset import CacheTraceDataset
from model.combined_lstm import CombinedLSTMModel

# -----------------------
# Config
# -----------------------
DB_PATH = "./DBs_Randika/cache_stats_1769606772.db"
TOKENIZER_PATH = "./DBs_Randika/trained_assembly_tokenizer/fast_tokenizer"

SEQ_LEN = 16
MAX_TOKEN_LEN = 15

# -----------------------
# Device
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Load tokenizer
# -----------------------
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)

# -----------------------
# Create dataset (for inference you can set sequence_length=SEQ_LEN)
# -----------------------
dataset = CacheTraceDataset(
    db_path=DB_PATH,
    tokenizer=tokenizer,
    sequence_length=SEQ_LEN,
    max_token_length=MAX_TOKEN_LEN
)

# -----------------------
# Load model
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
model.load_state_dict(torch.load("combined_lstm.pt", map_location=device))
model.to(device)
model.eval()

# -----------------------
# Pick a sample from dataset
# -----------------------
sample_token_ids, sample_access_feats, actual_misses = dataset[0]

# Add batch dimension
sample_token_ids = sample_token_ids.unsqueeze(0).to(device)
sample_access_feats = sample_access_feats.unsqueeze(0).to(device)

# -----------------------
# Predict
# -----------------------
with torch.no_grad():
    predicted_misses = model(sample_token_ids, sample_access_feats)

print("Predicted cache misses:", predicted_misses.cpu().numpy())
print("Actual cache misses:   ", actual_misses.numpy())
