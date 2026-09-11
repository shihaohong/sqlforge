"""Render the two files the serving box needs from the Spider dataset.

    data/serving/<split>.schemas.json     db_id -> CREATE TABLE DDL
    data/serving/<split>.questions.json   [{db_id, question}, ...]

The gateway resolves db_id to DDL from the schemas file, and the load test
replays the questions file, so neither needs the 166 sqlite databases - those
exist only to score execution accuracy offline.
"""

import json
from pathlib import Path

import typer

from text2sql.data import load_split, schema_ddl

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "serving"


def main(split: str = "dev", out_dir: str = "") -> None:
    out = Path(out_dir) if out_dir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    examples = load_split(split)

    schemas = {
        db_id: schema_ddl(db_id, split=split) for db_id in sorted({e.db_id for e in examples})
    }
    questions = [{"db_id": e.db_id, "question": e.question} for e in examples]

    (out / f"{split}.schemas.json").write_text(json.dumps(schemas, indent=2, sort_keys=True))
    (out / f"{split}.questions.json").write_text(json.dumps(questions, indent=2))
    print(f"wrote {len(schemas)} schemas and {len(questions)} questions to {out}")


if __name__ == "__main__":
    typer.run(main)
