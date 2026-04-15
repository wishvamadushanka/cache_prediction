import torch
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

# -----------------------
# Device
# -----------------------
settings = load_settings(SETTINGS_PATH)
paths = settings["paths"]
model_settings = settings["model"]
test_settings = settings["test"]

TOKENIZER_PATH = paths["tokenizer_path"]
device = settings["device"]
# -----------------------
# Load tokenizer
# -----------------------
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
run_specs = load_run_specs(RUN_CONFIG_PATH, split=test_settings["split"])

# -----------------------
# Create dataset (for inference you can set sequence_length=SEQ_LEN)
# -----------------------
dataset = CacheTraceDataset(
    runs=run_specs,
    tokenizer=tokenizer,
    sequence_length=test_settings["sequence_length"],
    max_token_length=test_settings["max_token_length"],
)

# -----------------------
# Load model
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
model.load_state_dict(torch.load(paths["model_path"], map_location=device))
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
dataset.close()
