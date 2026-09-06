"""Loading Spider examples and database schemas."""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

SPIDER_ROOT = Path(__file__).resolve().parents[2] / "data" / "spider_data"

SPLIT_FILES = {
    "train": "train_spider.json",
    "dev": "dev.json",
    "test": "test.json",
}


@dataclass(frozen=True)
class Example:
    db_id: str
    question: str
    gold_sql: str


def load_split(split: str, root: Path = SPIDER_ROOT) -> list[Example]:
    raw = json.loads((root / SPLIT_FILES[split]).read_text())
    return [Example(db_id=r["db_id"], question=r["question"], gold_sql=r["query"]) for r in raw]


def db_path(db_id: str, root: Path = SPIDER_ROOT, split: str = "dev") -> Path:
    # The 2024 release ships dev/train databases in database/ and test databases in test_database/.
    primary = "test_database" if split == "test" else "database"
    for folder in (primary, "database", "test_database"):
        p = root / folder / db_id / f"{db_id}.sqlite"
        if p.exists():
            return p
    raise FileNotFoundError(f"No sqlite database found for db_id={db_id!r}")


def schema_ddl(db_id: str, root: Path = SPIDER_ROOT, split: str = "dev") -> str:
    """Serialize a database schema as its CREATE TABLE statements.

    Reading DDL from the sqlite file itself (rather than tables.json) keeps the
    schema the model sees identical to the schema its SQL will run against.
    """
    uri = f"file:{db_path(db_id, root, split)}?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        rows = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table'"
            " AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
    return "\n\n".join(sql for (sql,) in rows)
