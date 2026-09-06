"""Render the Spider train split into chat-format SFT data.

Run locally (needs data/spider_data). Produces data/sft/{train,val}.jsonl with
fully-rendered messages so the GPU box never needs the sqlite databases and
cannot drift from the eval/serving prompt template.

    uv run scripts/build_train_data.py
"""

import json
import random
from pathlib import Path

import typer

from text2sql.data import load_split, schema_ddl
from text2sql.prompts import build_messages

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "sft"
VAL_SIZE = 200


def main(seed: int = 13, val_size: int = VAL_SIZE):
    examples = load_split("train")
    schemas = {}
    rows = []
    for ex in examples:
        if ex.db_id not in schemas:
            schemas[ex.db_id] = schema_ddl(ex.db_id, split="train")
        messages = build_messages(schemas[ex.db_id], ex.question)
        messages.append({"role": "assistant", "content": ex.gold_sql})
        rows.append({"messages": messages, "db_id": ex.db_id})

    # Shuffle before splitting: the raw file is grouped by database, and a val
    # slice taken from the tail would cover only a handful of schemas.
    random.Random(seed).shuffle(rows)
    val, train = rows[:val_size], rows[val_size:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_rows in (("train", train), ("val", val)):
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w") as f:
            for row in split_rows:
                f.write(json.dumps(row) + "\n")
        print(f"{path}: {len(split_rows)} examples")


if __name__ == "__main__":
    typer.run(main)
