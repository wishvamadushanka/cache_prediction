#!/bin/bash
# run_tokenizer_training.sh
# Script to run the complete tokenizer training pipeline

set -e

echo "==============================================================="
echo "Assembly Tokenizer Training Script"
echo "==============================================================="

SETTINGS_PATH="${1:-./config/settings.json}"

echo "Using settings file: $SETTINGS_PATH"
echo "Tokenizer configuration will be read from the 'tokenizer' section."
echo ""

# Step 1: Build corpus from all SQLite files
echo "Step 1: Building corpus from SQLite files..."
python -m tokenizer.train_assembly_tokenizer \
    --settings-path "$SETTINGS_PATH"

echo ""
echo "Step 2: Tokenizer training complete!"
echo ""
echo "Next steps:"
echo " Check the tokenizer output path from config/settings.json"
echo ""
