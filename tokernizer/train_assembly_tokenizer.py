#!/usr/bin/env python3

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import time
from typing import List, Set, Tuple
import logging
import re

# Tokenizer imports (HuggingFace tokenizers library)
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.processors import TemplateProcessing
from tokenizers import decoders

# Transformers integration
from transformers import PreTrainedTokenizerFast

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tokenizer_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AssemblyCorpusBuilder:
    
    
    def __init__(self, db_paths: List[str], max_instructions: int = None):
        """
        
        Args:
            db_paths: List of paths to SQLite database files
            max_instructions: Maximum number of instructions to process (None = all)
        """
        self.db_paths = db_paths
        self.max_instructions = max_instructions
        self.corpus_file = "assembly_corpus.txt"
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_instructions': 0,
            'unique_instructions': 0,
            'corpus_size_mb': 0
        }
    
    def _extract_clean_assembly(self, raw_string: str) -> str:
        """
        Extract clean assembly instruction from raw disassembly string
        Removes hex bytes at beginning and keeps assembly part only
        
        Args:
            raw_string: Raw disassembly string from SQLite
            
        Returns:
            Cleaned assembly instruction string
        """
        if raw_string is None:
            return ""
        match = re.search(r'([0-9a-f]{2} )+', raw_string.lower())
        
        if match:
            # Remove hex bytes and extra whitespace
            hex_end = match.end()
            assembly = raw_string[hex_end:].strip()
            return assembly
        else:
            # No hex bytes, return as is
            return raw_string.strip()
    
    def _validate_db_file(self, db_path: str) -> bool:
        """
        Validate SQLite database file has required structure
        
        Args:
            db_path: Path to SQLite file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if trace_entries table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cache_stats'")
            if not cursor.fetchone():
                logger.warning(f"No 'cache_stats' table in {db_path}")
                conn.close()
                return False
            
            # Check for required column
            cursor.execute("PRAGMA table_info(cache_stats)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'disassembly_string' not in columns:
                logger.warning(f"No 'disassembly_string' column in {db_path}")
                conn.close()
                return False
            
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error validating {db_path}: {str(e)}")
            return False
    
    def _process_db_file(self, db_path: str, corpus_writer) -> Tuple[int, int]:
        """
        Process single SQLite database file and extract instructions
        
        Args:
            db_path: Path to SQLite file
            corpus_writer: File writer object
            
        Returns:
            Tuple of (instructions_processed, unique_instructions)
        """
        instructions_processed = 0
        unique_instructions = set()
        
        try:
            logger.info(f"Processing database: {db_path}")
            
            conn = sqlite3.connect(db_path)
            
            # Use efficient chunking for large databases
            chunk_size = 100000
            offset = 0
            
            while True:
                # Query with limit and offset
                query = f"""
                SELECT disassembly_string 
                FROM cache_stats 
                LIMIT {chunk_size} OFFSET {offset}
                """
                
                df_chunk = pd.read_sql_query(query, conn)
                
                if df_chunk.empty:
                    break
                
                # Process each instruction
                for _, row in df_chunk.iterrows():
                    raw_instruction = row['disassembly_string']
                    
                    # Clean the assembly instruction
                    cleaned = self._extract_clean_assembly(raw_instruction)
                    
                    if cleaned and len(cleaned) > 0:
                        # Write to corpus file
                        corpus_writer.write(cleaned + '\n')
                        unique_instructions.add(cleaned)
                        instructions_processed += 1
                        
                        # Check if we've reached the limit
                        if self.max_instructions and instructions_processed >= self.max_instructions:
                            conn.close()
                            return instructions_processed, len(unique_instructions)
                
                offset += chunk_size
                
                # Log progress
                if offset % 500000 == 0:
                    logger.info(f"  Processed {offset:,} rows...")
            
            conn.close()
            
            return instructions_processed, len(unique_instructions)
            
        except Exception as e:
            logger.error(f"Error processing {db_path}: {str(e)}")
            return instructions_processed, len(unique_instructions)
    
    def build_corpus(self) -> dict:
        """
        Build corpus from all SQLite database files
        
        Returns:
            Dictionary with corpus statistics
        """
        logger.info("=" * 60)
        logger.info("Building Assembly Instruction Corpus")
        logger.info("=" * 60)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.corpus_file) if os.path.dirname(self.corpus_file) else '.', exist_ok=True)
        
        total_instructions = 0
        all_unique_instructions = set()
        
        # Open corpus file for writing
        with open(self.corpus_file, 'w', encoding='utf-8') as f:
            # Process each database file
            for db_path in self.db_paths:
                self.stats['total_files'] += 1
                
                # Validate file
                if not os.path.exists(db_path):
                    logger.error(f"Database file not found: {db_path}")
                    self.stats['failed_files'] += 1
                    continue
                
                if not self._validate_db_file(db_path):
                    logger.error(f"Invalid database structure: {db_path}")
                    self.stats['failed_files'] += 1
                    continue
                
                # Process the file
                processed, unique = self._process_db_file(db_path, f)
                total_instructions += processed
                all_unique_instructions.update([f"{ins}\n" for ins in all_unique_instructions])
                
                self.stats['processed_files'] += 1
                
                logger.info(f"  ✓ {db_path}: {processed:,} instructions ({unique:,} unique)")
                
                # Check if we've reached the global limit
                if self.max_instructions and total_instructions >= self.max_instructions:
                    logger.info(f"Reached maximum instruction limit: {self.max_instructions}")
                    break
        
        # Update statistics
        self.stats['total_instructions'] = total_instructions
        self.stats['unique_instructions'] = len(all_unique_instructions)
        
        # Calculate corpus file size
        if os.path.exists(self.corpus_file):
            self.stats['corpus_size_mb'] = os.path.getsize(self.corpus_file) / (1024 * 1024)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("CORPUS BUILDING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total database files: {self.stats['total_files']}")
        logger.info(f"Successfully processed: {self.stats['processed_files']}")
        logger.info(f"Failed files: {self.stats['failed_files']}")
        logger.info(f"Total instructions extracted: {self.stats['total_instructions']:,}")
        logger.info(f"Unique instructions: {self.stats['unique_instructions']:,}")
        logger.info(f"Corpus file: {self.corpus_file}")
        logger.info(f"Corpus size: {self.stats['corpus_size_mb']:.2f} MB")
        logger.info("=" * 60)
        
        return self.stats

class AssemblyTokenizerTrainer:
    """
    Trains Byte-Level BPE tokenizer with Whitespace preprocessor
  
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize tokenizer trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            'vocab_size': 5000,  # Adjust based on your needs
            'min_frequency': 2,
            'special_tokens': ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            'max_token_length': 15,  # From paper
            'save_path': "trained_tokenizer",
            'corpus_file': "assembly_corpus.txt"
        }
        
        # Ensure save directory exists
        os.makedirs(self.config['save_path'], exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = None
        self.fast_tokenizer = None
        
    def train_tokenizer(self, corpus_file: str = None) -> dict:
        """
        Train Byte-Level BPE tokenizer on assembly corpus
        
        Args:
            corpus_file: Path to corpus file (overrides config)
            
        Returns:
            Dictionary with training statistics
        """
        corpus_file = corpus_file or self.config['corpus_file']
        
        logger.info("=" * 60)
        logger.info("Training Byte-Level BPE Tokenizer")
        logger.info("=" * 60)
        logger.info(f"Corpus file: {corpus_file}")
        logger.info(f"Vocabulary size: {self.config['vocab_size']}")
        logger.info(f"Special tokens: {self.config['special_tokens']}")
        
        # Check if corpus exists
        if not os.path.exists(corpus_file):
            logger.error(f"Corpus file not found: {corpus_file}")
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        
        # Get corpus size
        corpus_size = os.path.getsize(corpus_file) / (1024 * 1024)  # MB
        logger.info(f"Corpus size: {corpus_size:.2f} MB")
        
        start_time = time.time()
        
        try:
            # Step 1: Initialize Byte-Level BPE tokenizer
            # Using byte-level BPE
            logger.info("Initializing Byte-Level BPE tokenizer...")
            self.tokenizer = Tokenizer(models.BPE(
                unk_token="[UNK]",
                fuse_unk=True
            ))
            
            # Step 2: Add normalizer (NFKC normalization)
            self.tokenizer.normalizer = normalizers.Sequence([
                normalizers.NFKC()  # Unicode normalization
            ])
            
            # Step 3: Add pre-tokenizer
            # Note: "Whitespace" preprocessor, which splits on whitespace
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            
            # Step 4: Create trainer with specified parameters
            trainer = BpeTrainer(
                vocab_size=self.config['vocab_size'],
                min_frequency=self.config['min_frequency'],
                special_tokens=self.config['special_tokens'],
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # Byte-level alphabet
                show_progress=True
            )
            
            # Step 5: Train tokenizer
            logger.info("Training tokenizer on assembly corpus...")
            self.tokenizer.train(
                files=[corpus_file],
                trainer=trainer
            )
            
            # Step 6: Add post-processor (for special tokens)
            self.tokenizer.post_processor = processors.TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", self.tokenizer.token_to_id("[CLS]")),
                    ("[SEP]", self.tokenizer.token_to_id("[SEP]"))
                ]
            )
            
            # Step 7: Add decoder
            self.tokenizer.decoder = decoders.ByteLevel()
            
            # Step 8: Save tokenizer
            tokenizer_json_path = os.path.join(self.config['save_path'], "assembly_tokenizer.json")
            self.tokenizer.save(tokenizer_json_path)
            logger.info(f"Tokenizer saved to: {tokenizer_json_path}")
            
            # Step 9: Create PreTrainedTokenizerFast for Transformers compatibility
            self.fast_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=self.tokenizer,
                pad_token="[PAD]",
                unk_token="[UNK]",
                cls_token="[CLS]",
                sep_token="[SEP]",
                mask_token="[MASK]",
                padding_side="right",
                truncation_side="right"
            )
            
            # Set model_max_length (maximum sequence length)
            self.fast_tokenizer.model_max_length = self.config['max_token_length']
            
            # Save the fast tokenizer
            fast_tokenizer_path = os.path.join(self.config['save_path'], "fast_tokenizer")
            self.fast_tokenizer.save_pretrained(fast_tokenizer_path)
            logger.info(f"Fast tokenizer saved to: {fast_tokenizer_path}")
            
            # Calculate training statistics
            training_time = time.time() - start_time
            vocab_size = self.tokenizer.get_vocab_size()
            
            stats = {
                'training_time_seconds': training_time,
                'vocab_size': vocab_size,
                'tokenizer_path': tokenizer_json_path,
                'fast_tokenizer_path': fast_tokenizer_path,
                'config': self.config
            }
            
            # Test the tokenizer
            self._test_tokenizer()
            
            logger.info("\n" + "=" * 60)
            logger.info("TOKENIZER TRAINING COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Training time: {training_time:.2f} seconds")
            logger.info(f"Vocabulary size: {vocab_size}")
            logger.info(f"Special tokens: {self.fast_tokenizer.all_special_tokens}")
            logger.info(f"Tokenizer saved to: {tokenizer_json_path}")
            logger.info(f"Fast tokenizer saved to: {fast_tokenizer_path}")
            logger.info("=" * 60)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error training tokenizer: {str(e)}")
            raise
    
    def _test_tokenizer(self):
        """Test the trained tokenizer on sample instructions"""
        logger.info("\nTesting tokenizer on sample instructions...")
        
        # Sample assembly instructions
        test_instructions = [
            "call $0x00007b235c6a3030 %rsp -> %rsp 0xfffffff8(%rsp)[8byte]",
            "nop %edx",
            "push %r14 %rsp -> %rsp 0xfffffff8(%rsp)[8byte]",
            "cmp %rax $0x00000000000000022",
            "mov %rdi -> %rsi",
            "sub %rax %rsi -> %rsi",
            "and $0xdf <rel> 0x00007b235c6bce0e[1byte] -> <rel> 0x00007b235c6bce0e[1byte]"
        ]
        
        for i, instr in enumerate(test_instructions):
            logger.info(f"\nTest {i+1}: {instr[:50]}...")
            
            # Tokenize
            encoding = self.fast_tokenizer(
                instr,
                truncation=True,
                padding='max_length',
                max_length=self.config['max_token_length'],
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False
            )
            
            # Get tokens
            tokens = self.fast_tokenizer.tokenize(instr)
            
            logger.info(f"  Tokens: {tokens}")
            logger.info(f"  Token IDs: {encoding['input_ids'].tolist()[0]}")
            logger.info(f"  Attention mask: {encoding['attention_mask'].tolist()[0]}")
            logger.info(f"  Token count: {len(tokens)}")
            
            # Decode back
            decoded = self.fast_tokenizer.decode(encoding['input_ids'][0], skip_special_tokens=True)
            logger.info(f"  Decoded: {decoded[:50]}...")
    
    def load_tokenizer(self, tokenizer_path: str = None) -> PreTrainedTokenizerFast:
        """
        Load trained tokenizer
        
        Args:
            tokenizer_path: Path to saved tokenizer
            
        Returns:
            PreTrainedTokenizerFast instance
        """
        tokenizer_path = tokenizer_path or os.path.join(self.config['save_path'], "fast_tokenizer")
        
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        logger.info(f"Loading tokenizer from {tokenizer_path}...")
        self.fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        
        # Ensure special tokens are set
        if not self.fast_tokenizer.pad_token:
            self.fast_tokenizer.pad_token = "[PAD]"
        if not self.fast_tokenizer.unk_token:
            self.fast_tokenizer.unk_token = "[UNK]"
        
        logger.info(f"✓ Tokenizer loaded (vocab size: {self.fast_tokenizer.vocab_size})")
        
        return self.fast_tokenizer

def find_sqlite_files(directory: str, pattern: str = "*.db") -> List[str]:
    """
    Find all SQLite database files in a directory
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        
    Returns:
        List of file paths
    """
    db_files = []
    
    if os.path.isfile(directory) and directory.endswith('.db'):
        return [directory]
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.db') or file.endswith('.sqlite') or file.endswith('.sqlite3'):
                db_files.append(os.path.join(root, file))
    
    return sorted(db_files)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Train Byte-Level BPE Tokenizer on Assembly Instructions")
    parser.add_argument("--db-dir", type=str, required=True,
                       help="Directory containing SQLite database files")
    parser.add_argument("--output-dir", type=str, default="trained_tokenizer",
                       help="Output directory for tokenizer")
    parser.add_argument("--vocab-size", type=int, default=5000,
                       help="Vocabulary size for tokenizer")
    parser.add_argument("--max-instructions", type=int, default=None,
                       help="Maximum number of instructions to process")
    parser.add_argument("--corpus-file", type=str, default="assembly_corpus.txt",
                       help="Output corpus file path")
    parser.add_argument("--skip-corpus", action="store_true",
                       help="Skip corpus building (use existing corpus)")
    parser.add_argument("--train-only", action="store_true",
                       help="Train tokenizer only (skip corpus building)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Find all SQLite files
    logger.info(f"Searching for SQLite files in: {args.db_dir}")
    db_files = find_sqlite_files(args.db_dir)
    
    if not db_files:
        logger.error(f"No SQLite database files found in {args.db_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(db_files)} database files")
    for i, db_file in enumerate(db_files[:10]):  # Show first 10
        logger.info(f"  {i+1}. {db_file}")
    if len(db_files) > 10:
        logger.info(f"  ... and {len(db_files) - 10} more")
    
    # Step 2: Build corpus (unless skipping)
    if not args.skip_corpus and not args.train_only:
        corpus_builder = AssemblyCorpusBuilder(
            db_paths=db_files,
            max_instructions=args.max_instructions
        )
        
        corpus_stats = corpus_builder.build_corpus()
        
        # Save corpus statistics
        stats_file = os.path.join(args.output_dir, "corpus_stats.json")
        import json
        with open(stats_file, 'w') as f:
            json.dump(corpus_stats, f, indent=2)
        logger.info(f"Corpus statistics saved to: {stats_file}")
        
        corpus_file = corpus_builder.corpus_file
    else:
        corpus_file = args.corpus_file
        if not os.path.exists(corpus_file):
            logger.error(f"Corpus file not found: {corpus_file}")
            sys.exit(1)
        logger.info(f"Using existing corpus file: {corpus_file}")
    
    # Step 3: Train tokenizer
    trainer_config = {
        'vocab_size': args.vocab_size,
        'min_frequency': 2,
        'special_tokens': ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        'max_token_length': 15,
        'save_path': args.output_dir,
        'corpus_file': corpus_file
    }
    
    trainer = AssemblyTokenizerTrainer(trainer_config)
    
    try:
        training_stats = trainer.train_tokenizer(corpus_file)
        
        # Save training statistics
        stats_file = os.path.join(args.output_dir, "training_stats.json")
        import json
        with open(stats_file, 'w') as f:
            json.dump(training_stats, f, indent=2)
        logger.info(f"Training statistics saved to: {stats_file}")
        
        # Generate a summary report
        report_file = os.path.join(args.output_dir, "tokenizer_report.txt")
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ASSEMBLY TOKENIZER TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Training Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Database Files: {len(db_files)}\n")
            f.write(f"Corpus File: {corpus_file}\n")
            f.write(f"Vocabulary Size: {training_stats['vocab_size']}\n")
            f.write(f"Training Time: {training_stats['training_time_seconds']:.2f} seconds\n")
            f.write(f"Output Directory: {args.output_dir}\n")
            f.write(f"Tokenizer JSON: {training_stats['tokenizer_path']}\n")
            f.write(f"Fast Tokenizer: {training_stats['fast_tokenizer_path']}\n")
            f.write("\n" + "=" * 60 + "\n")
        
        logger.info(f"Summary report saved to: {report_file}")
        logger.info("\n" + "=" * 60)
        logger.info("TOKENIZER TRAINING COMPLETE!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to train tokenizer: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
