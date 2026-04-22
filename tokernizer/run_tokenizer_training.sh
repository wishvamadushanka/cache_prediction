#!/bin/bash
# run_tokenizer_training.sh
# Script to run the complete tokenizer training pipeline

echo "==============================================================="
echo "Assembly Tokenizer Training Script"
echo "==============================================================="

# Configuration
DB_DIR="./"  # Change this to your DB directory
OUTPUT_DIR="trained_assembly_tokenizer"
VOCAB_SIZE=5000
MAX_INSTRUCTIONS=1000000  # Limit to 1M instructions for faster training

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Database directory: $DB_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Vocabulary size: $VOCAB_SIZE"
echo "Max instructions: $MAX_INSTRUCTIONS"
echo ""

# Step 1: Build corpus from all SQLite files
echo "Step 1: Building corpus from SQLite files..."
python train_assembly_tokenizer.py \
    --db-dir "$DB_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --vocab-size $VOCAB_SIZE \
    --max-instructions $MAX_INSTRUCTIONS \
    --corpus-file "$OUTPUT_DIR/assembly_corpus.txt"

if [ $? -ne 0 ]; then
    echo "Error: Corpus building failed!"
    exit 1
fi

echo ""
echo "Step 2: Tokenizer training complete!"
echo ""
echo "Next steps:"
echo " Check the tokenizer output in: $OUTPUT_DIR/"
echo ""

# List generated files
echo "Generated files:"
ls -la $OUTPUT_DIR/
