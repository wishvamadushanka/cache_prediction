import sqlite3
import re
import sys
import glob
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_DIR = PROJECT_ROOT / "db"


def preprocess_instruction(text: str) -> str:
    if text is None:
        return ""

    match = re.search(r'([0-9a-f]{2} )+', text.lower())
    
    if match:
        # Remove hex bytes and extra whitespace
        hex_end = match.end()
        assembly = text[hex_end:].strip()
        
        # Also clean up extra register annotations if needed
        # The paper might keep "%rsp -> %rsp" annotations
        return assembly
    else:
        # No hex bytes, return as is
        return text.strip()


def ensure_column_exists(conn):
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(cache_stats);")
    columns = [row[1] for row in cursor.fetchall()]

    if "preprocessed_instruction" not in columns:
        cursor.execute("""
            ALTER TABLE cache_stats
            ADD COLUMN preprocessed_instruction TEXT
        """)
        conn.commit()


def update_db_instructions(db_path, batch_size=10_000):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    ensure_column_exists(conn)

    cursor.execute("""
        SELECT rowid, disassembly_string
        FROM cache_stats
        WHERE disassembly_string IS NOT NULL
    """)

    updates = []
    count = 0

    for rowid, raw_instr in cursor:
        processed = preprocess_instruction(raw_instr)
        updates.append((processed, rowid))
        count += 1

        if len(updates) >= batch_size:
            conn.executemany("""
                UPDATE cache_stats
                SET preprocessed_instruction = ?
                WHERE rowid = ?
            """, updates)
            conn.commit()
            updates.clear()
            print(f"Updated {count} rows...")

    # Final flush
    if updates:
        conn.executemany("""
            UPDATE cache_stats
            SET preprocessed_instruction = ?
            WHERE rowid = ?
        """, updates)
        conn.commit()

    conn.close()
    print(f"Done. Total rows updated: {count}")


# if __name__ == "__main__":
#     if len(sys.argv) == 2:
#         DB_PATH = sys.argv[1]
#     else:
#         print("INFO: Using default DB path")

#     update_db_instructions(DB_PATH)



if __name__ == "__main__":
    if len(sys.argv) == 2:
        target = Path(sys.argv[1]).expanduser().resolve()
        if target.is_dir():
            db_files = sorted(str(path) for path in target.glob("cache_stats*.db"))
        else:
            db_files = [str(target)]
    else:
        print(f"INFO: No DB specified. Scanning default DB directory: {DEFAULT_DB_DIR}")
        db_files = sorted(str(path) for path in DEFAULT_DB_DIR.glob("cache_stats*.db"))

    if not db_files:
        print("No matching DB files found.")
        sys.exit(1)

    for db_path in db_files:
        print(f"\n=== Processing {db_path} ===")
        try:
            update_db_instructions(db_path)
        except Exception as e:
            print(f"ERROR processing {db_path}: {e}")
