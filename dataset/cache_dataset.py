import sqlite3
import torch
from torch.utils.data import Dataset
import numpy as np

from transformers import PreTrainedTokenizerFast

class CacheTraceDataset(Dataset):
    """
    Builds sliding windows of length N:
      X  = instructions + access features
      y  = cache misses in that window
    """

    def __init__(
        self,
        db_path,
        tokenizer,
        sequence_length=16,
        max_token_length=15
    ):
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                *
            FROM cache_stats
            ORDER BY instruction_number
            LIMIT 1000
        """)

        rows = cursor.fetchall()
        # print(rows)
        conn.close()

        self.sample_size = int(len(rows)/sequence_length)
        # print(self.sample_size)

        self.samples = []

        for i in range(0, self.sample_size):
            window = rows[i * sequence_length : (i * sequence_length) + sequence_length]
            # print(window)

            instructions = [r[-1] for r in window]
            # print(instructions)
            access_features = []
            misses = []
            l1d_miss = 0
            l1i_miss = 0
            ll_miss = 0

            for r in window:
                tmp = []
                for idx, val in enumerate(r):
                    # 0 Instruction Number
                    # 1 Access Address Delta
                    # 2 PC Address Delta
                    # 6 Instruction Type
                    # 7 Byte Count
                    # 10 Core
                    # 11 Thread Switch
                    # 12 Core Switch
                    if idx in [ 0, 1, 2, 6, 7, 10, 11, 12]:  # skip 2nd and 5th index
                        if val:
                            tmp.append(val)
                        else:
                            tmp.append(0)

                    if idx == 3:
                        if val ==1:
                            l1d_miss = l1d_miss + 1
                    if idx == 4:
                        if val ==1:
                            l1i_miss = l1i_miss + 1
                    if idx == 5:
                        if val ==1:
                            ll_miss = ll_miss + 1
                        
                
                # d cache size in bytes
                tmp.append(1024)
                # i cache size in bytes
                tmp.append(1024)
                # ll cache size in bytes
                tmp.append(1024)
                access_features.append(tmp)
            
            misses = [l1d_miss, l1i_miss, ll_miss]
            
            # print(access_features)

            encoding = tokenizer(
                instructions,
                padding="max_length",
                truncation=True,
                max_length=max_token_length,
                return_tensors="pt",
                return_attention_mask=True
            )
            # for inst in instructions:
            #     encoding_1 = tokenizer(
            #         inst,
            #         padding="max_length",
            #         truncation=True,
            #         max_length=max_token_length,
            #         return_tensors="pt",
            #         return_attention_mask=True
            #     )
            #     token_ids = encoding_1["input_ids"]
            #     print(token_ids)

            token_ids = encoding["input_ids"]
            # print(token_ids)

            access_tensor = torch.tensor(access_features, dtype=torch.float)

            self.samples.append((
                token_ids,
                access_tensor,
                torch.tensor(misses, dtype=torch.float)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == '__main__':
    DB_PATH = "./DBs_Randika/cache_stats_1769606772.db"
    # DB_PATH = "data/raw/traces.db"
    TOKENIZER_PATH = "./DBs_Randika/trained_assembly_tokenizer/fast_tokenizer"
    # TOKENIZER_PATH = "tokenizer/trained_tokenizer/fast_tokenizer"

    SEQ_LEN = 2
    MAX_TOKEN_LEN = 15
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1e-3

    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)

    dataset = CacheTraceDataset(
        db_path=DB_PATH,
        tokenizer=tokenizer,
        sequence_length=SEQ_LEN,
        max_token_length=MAX_TOKEN_LEN
    )

    print(dataset.samples)