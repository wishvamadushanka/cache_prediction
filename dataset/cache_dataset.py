import bisect
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CacheRunConfig:
    db_path: str
    l1d_size: int
    l1i_size: int
    ll_size: int
    instruction_column: str = "preprocessed_instruction"
    fallback_instruction_column: str = "disassembly_string"
    max_rows: int | None = None


class CacheTraceDataset(Dataset):
    """
    Lazy dataset of non-overlapping subsequences.

    Each sample is one window of `sequence_length` rows from a specific run.
    The model input is:
    - tokenized instruction strings
    - numeric access/cache features per row

    The regression target is the summed cache misses across the window:
    [L1D, L1I, LL].
    """

    FEATURE_COLUMNS = (
        "instruction_number",
        "access_address_delta",
        "pc_address_delta",
        "instr_type",
        "byte_count",
        "core",
        "thread_switch",
        "core_switch",
    )

    TARGET_COLUMNS = ("l1d_miss", "l1i_miss", "ll_miss")

    def __init__(
        self,
        runs,
        tokenizer,
        sequence_length=200,
        max_token_length=15,
    ):
        if not runs:
            raise ValueError("At least one CacheRunConfig is required.")

        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.runs = [self._coerce_run(run) for run in runs]

        self._connections = {}
        self._sample_offsets = []
        self._run_sample_counts = []

        total_samples = 0
        for run in self.runs:
            row_count = self._get_row_count(run)
            sample_count = row_count // self.sequence_length
            self._run_sample_counts.append(sample_count)
            total_samples += sample_count
            self._sample_offsets.append(total_samples)

        self.total_samples = total_samples

    def _coerce_run(self, run):
        if isinstance(run, CacheRunConfig):
            return run

        if isinstance(run, dict):
            return CacheRunConfig(**run)

        raise TypeError(f"Unsupported run spec: {type(run)!r}")

    def _get_row_count(self, run: CacheRunConfig) -> int:
        conn = sqlite3.connect(run.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cache_stats")
        row_count = cursor.fetchone()[0]
        conn.close()

        if run.max_rows is not None:
            row_count = min(row_count, run.max_rows)

        return row_count

    def _get_connection(self, run: CacheRunConfig):
        db_path = str(Path(run.db_path).resolve())
        connection = self._connections.get(db_path)
        if connection is None:
            connection = sqlite3.connect(db_path)
            self._connections[db_path] = connection
        return connection

    def _resolve_sample(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(idx)

        run_idx = bisect.bisect_right(self._sample_offsets, idx)
        previous_offset = 0 if run_idx == 0 else self._sample_offsets[run_idx - 1]
        window_idx = idx - previous_offset
        return self.runs[run_idx], window_idx

    def _fetch_window(self, run: CacheRunConfig, window_idx: int):
        conn = self._get_connection(run)
        cursor = conn.cursor()

        offset = window_idx * self.sequence_length
        instruction_column = run.instruction_column
        fallback_column = run.fallback_instruction_column

        query = f"""
            SELECT
                instruction_number,
                access_address_delta,
                pc_address_delta,
                l1d_miss,
                l1i_miss,
                ll_miss,
                instr_type,
                byte_count,
                core,
                thread_switch,
                core_switch,
                {instruction_column},
                {fallback_column}
            FROM cache_stats
            ORDER BY instruction_number
            LIMIT ? OFFSET ?
        """

        cursor.execute(query, (self.sequence_length, offset))
        rows = cursor.fetchall()

        if len(rows) != self.sequence_length:
            raise IndexError(
                f"Expected {self.sequence_length} rows, got {len(rows)} "
                f"for {run.db_path} window {window_idx}"
            )

        return rows

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        run, window_idx = self._resolve_sample(idx)
        rows = self._fetch_window(run, window_idx)

        instructions = []
        access_features = []
        l1d_miss_total = 0
        l1i_miss_total = 0
        ll_miss_total = 0

        for row in rows:
            (
                instruction_number,
                access_address_delta,
                pc_address_delta,
                l1d_miss,
                l1i_miss,
                ll_miss,
                instr_type,
                byte_count,
                core,
                thread_switch,
                core_switch,
                instruction_text,
                fallback_instruction_text,
            ) = row

            final_instruction = instruction_text or fallback_instruction_text or ""
            instructions.append(final_instruction)

            access_features.append(
                [
                    instruction_number or 0,
                    access_address_delta or 0,
                    pc_address_delta or 0,
                    instr_type or 0,
                    byte_count or 0,
                    core or 0,
                    thread_switch or 0,
                    core_switch or 0,
                    run.l1d_size,
                    run.l1i_size,
                    run.ll_size,
                ]
            )

            l1d_miss_total += int(l1d_miss or 0)
            l1i_miss_total += int(l1i_miss or 0)
            ll_miss_total += int(ll_miss or 0)

        encoding = self.tokenizer(
            instructions,
            padding="max_length",
            truncation=True,
            max_length=self.max_token_length,
            return_tensors="pt",
            return_attention_mask=False,
        )

        token_ids = encoding["input_ids"]
        access_tensor = torch.tensor(access_features, dtype=torch.float32)
        target_tensor = torch.tensor(
            [l1d_miss_total, l1i_miss_total, ll_miss_total],
            dtype=torch.float32,
        )

        return token_ids, access_tensor, target_tensor

    def close(self):
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()

    def __del__(self):
        self.close()
