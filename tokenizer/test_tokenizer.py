#!/usr/bin/env python3
"""
Test the trained tokenizer using the shared pipeline settings.
"""

import argparse
import sys
from pathlib import Path

from transformers import PreTrainedTokenizerFast


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = PROJECT_ROOT / "cache_prediction"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from config.settings_loader import load_settings


DEFAULT_SETTINGS_PATH = PIPELINE_ROOT / "config" / "settings.json"


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (PIPELINE_ROOT / path).resolve()
    return path


def get_default_tokenizer_path(settings: dict) -> Path:
    tokenizer_settings = settings.get("tokenizer", {})
    output_dir = tokenizer_settings.get("output_dir")
    if output_dir:
        return resolve_path(output_dir) / "fast_tokenizer"
    return resolve_path(settings["paths"]["tokenizer_path"])


def test_tokenizer(tokenizer_path: Path, max_token_length: int):
    print("=" * 60)
    print("Testing Trained Tokenizer")
    print("=" * 60)

    if not tokenizer_path.exists():
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        print("Please train the tokenizer first using train_assembly_tokenizer.py")
        sys.exit(1)

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    print("✓ Tokenizer loaded")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Special tokens: {tokenizer.all_special_tokens}")
    print(f"  Model max length: {tokenizer.model_max_length}")

    test_instructions = [
        "call $0x00007b235c6a3030 %rsp -> %rsp 0xfffffff8(%rsp)[8byte]",
        "nop %edx",
        "push %r14 %rsp -> %rsp 0xfffffff8(%rsp)[8byte]",
        "cmp %rax $0x00000000000000022",
        "mov %rdi -> %rsi",
        "sub %rax %rsi -> %rsi",
        "and $0xdf <rel> 0x00007b235c6bce0e[1byte] -> <rel> 0x00007b235c6bce0e[1byte]",
    ]

    print("\n" + "=" * 60)
    print("Tokenizer Test Results")
    print("=" * 60)

    for i, instr in enumerate(test_instructions):
        print(f"\nTest {i + 1}: {instr}")
        print("-" * 40)

        encoding = tokenizer(
            instr,
            truncation=True,
            padding="max_length",
            max_length=max_token_length,
            return_tensors="pt",
            return_attention_mask=True,
        )

        token_ids = encoding["input_ids"][0].tolist()
        attention_mask = encoding["attention_mask"][0].tolist()
        tokens = tokenizer.tokenize(instr)

        print(f"  Original: {instr}")
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        print(f"  Attention mask: {attention_mask}")
        print(f"  Token count: {len(tokens)}")

        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"  Decoded: {decoded}")

        if decoded.strip() == instr.strip():
            print("  ✓ Round-trip successful")
        else:
            print("  ! Round-trip mismatch")
            print(f"    Original: '{instr}'")
            print(f"    Decoded:  '{decoded}'")

    print("\n" + "=" * 60)
    print("Batch Processing Test")
    print("=" * 60)

    batch = test_instructions[:3]
    print(f"Batch size: {len(batch)}")

    batch_encoding = tokenizer(
        batch,
        truncation=True,
        padding="max_length",
        max_length=max_token_length,
        return_tensors="pt",
        return_attention_mask=True,
    )

    print(f"Batch input_ids shape: {batch_encoding['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch_encoding['attention_mask'].shape}")

    print("\n" + "=" * 60)
    print("Tokenizer Test Complete!")
    print("=" * 60)

    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Test the trained tokenizer.")
    parser.add_argument(
        "--settings-path",
        default=str(DEFAULT_SETTINGS_PATH),
        help="Path to settings.json containing tokenizer configuration.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional explicit tokenizer path. Overrides settings-derived path.",
    )
    args = parser.parse_args()

    settings_path = resolve_path(args.settings_path)
    settings = load_settings(str(settings_path))

    tokenizer_path = (
        resolve_path(args.tokenizer_path)
        if args.tokenizer_path
        else get_default_tokenizer_path(settings)
    )
    max_token_length = settings.get("tokenizer", {}).get("max_token_length", 15)

    test_tokenizer(tokenizer_path, max_token_length)


if __name__ == "__main__":
    main()
