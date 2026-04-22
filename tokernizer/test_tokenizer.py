#!/usr/bin/env python3
"""
Test the trained tokenizer
"""

import os
import sys
from transformers import PreTrainedTokenizerFast
import torch

def test_tokenizer(tokenizer_path: str = "trained_assembly_tokenizer/fast_tokenizer"):
    """
    Test the trained tokenizer
    """
    print("=" * 60)
    print("Testing Trained Tokenizer")
    print("=" * 60)
    
    # Load tokenizer
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        print("Please train the tokenizer first using train_assembly_tokenizer.py")
        sys.exit(1)
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    print(f"✓ Tokenizer loaded")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Special tokens: {tokenizer.all_special_tokens}")
    print(f"  Model max length: {tokenizer.model_max_length}")
    
    # Test samples
    test_instructions = [
        "call $0x00007b235c6a3030 %rsp -> %rsp 0xfffffff8(%rsp)[8byte]",
        "nop %edx",
        "push %r14 %rsp -> %rsp 0xfffffff8(%rsp)[8byte]",
        "cmp %rax $0x00000000000000022",
        "mov %rdi -> %rsi",
        "sub %rax %rsi -> %rsi",
        "and $0xdf <rel> 0x00007b235c6bce0e[1byte] -> <rel> 0x00007b235c6bce0e[1byte]"
    ]
    
    print("\n" + "=" * 60)
    print("Tokenizer Test Results")
    print("=" * 60)
    
    for i, instr in enumerate(test_instructions):
        print(f"\nTest {i+1}: {instr}")
        print("-" * 40)
        
        # Tokenize with padding/truncation to max_length=15
        encoding = tokenizer(
            instr,
            truncation=True,
            padding='max_length',
            max_length=15,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        # Get token IDs and attention mask
        token_ids = encoding['input_ids'][0].tolist()
        attention_mask = encoding['attention_mask'][0].tolist()
        
        # Decode tokens
        tokens = tokenizer.tokenize(instr)
        
        print(f"  Original: {instr}")
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        print(f"  Attention mask: {attention_mask}")
        print(f"  Token count: {len(tokens)}")
        
        # Decode back
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"  Decoded: {decoded}")
        
        # Verify round-trip
        if decoded.strip() == instr.strip():
            print(f"  ✓ Round-trip successful")
        else:
            print(f"  ⚠ Round-trip mismatch")
            print(f"    Original: '{instr}'")
            print(f"    Decoded:  '{decoded}'")
    
    # Test batch processing
    print("\n" + "=" * 60)
    print("Batch Processing Test")
    print("=" * 60)
    
    batch = test_instructions[:3]
    print(f"Batch size: {len(batch)}")
    
    batch_encoding = tokenizer(
        batch,
        truncation=True,
        padding='max_length',
        max_length=15,
        return_tensors="pt",
        return_attention_mask=True
    )
    
    print(f"Batch input_ids shape: {batch_encoding['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch_encoding['attention_mask'].shape}")
    
    print("\n" + "=" * 60)
    print("Tokenizer Test Complete!")
    print("=" * 60)
    
    return tokenizer

if __name__ == "__main__":
    # You can specify a custom tokenizer path as argument
    tokenizer_path = "trained_assembly_tokenizer/fast_tokenizer"
    if len(sys.argv) > 1:
        tokenizer_path = sys.argv[1]
    
    tokenizer = test_tokenizer(tokenizer_path)