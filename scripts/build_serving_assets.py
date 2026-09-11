"""Render the files the serving box and the demo need from the Spider dataset.

    data/serving/<split>.schemas.json     db_id -> CREATE TABLE DDL
    data/serving/<split>.questions.json   [{db_id, question, gold_sql}, ...]
    data/serving/databases/<db_id>.sqlite  the small databases, for the demo

The gateway resolves db_id to DDL from the schemas file, the load test replays
the questions, and the demo executes generated SQL against the bundled
databases and compares the rows to the gold query's.

Only databases under MAX_DB_BYTES are bundled. Nineteen of the twenty dev
databases together are about 1MB; wta_1 alone is 105MB, so it is left out and
the image stays small. The demo advertises exactly what it ships.
"""

import json
import shutil
from pathlib import Path

import typer

from text2sql.data import db_path, load_split, schema_ddl

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "serving"
MAX_DB_BYTES = 5 * 1024 * 1024


def main(split: str = "dev", out_dir: str = "") -> None:
    out = Path(out_dir) if out_dir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    examples = load_split(split)
    db_ids = sorted({e.db_id for e in examples})

    schemas = {db_id: schema_ddl(db_id, split=split) for db_id in db_ids}
    questions = [
        {"db_id": e.db_id, "question": e.question, "gold_sql": e.gold_sql} for e in examples
    ]

    db_out = out / "databases"
    db_out.mkdir(parents=True, exist_ok=True)
    bundled, skipped = [], []
    for db_id in db_ids:
        source = db_path(db_id, split=split)
        if source.stat().st_size > MAX_DB_BYTES:
            skipped.append((db_id, source.stat().st_size))
            continue
        shutil.copyfile(source, db_out / f"{db_id}.sqlite")
        bundled.append(db_id)

    (out / f"{split}.schemas.json").write_text(json.dumps(schemas, indent=2, sort_keys=True))
    (out / f"{split}.questions.json").write_text(json.dumps(questions, indent=2))

    total = sum((db_out / f"{db_id}.sqlite").stat().st_size for db_id in bundled)
    print(f"wrote {len(schemas)} schemas and {len(questions)} questions to {out}")
    print(f"bundled {len(bundled)} databases ({total / 1e6:.1f} MB) into {db_out}")
    for db_id, size in skipped:
        print(f"  skipped {db_id} ({size / 1e6:.1f} MB > {MAX_DB_BYTES / 1e6:.0f} MB limit)")


if __name__ == "__main__":
    typer.run(main)
