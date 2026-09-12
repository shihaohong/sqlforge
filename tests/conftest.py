"""A self-contained stand-in for the demo's serving bundle.

The real bundle (`scripts/build_serving_assets.py`) is rendered from the
841MB Spider download, which the repository does not carry, so tests written
against it skip wherever that data is absent. That is precisely where tests
matter least to the author and most to everyone else: they skipped in CI,
which quietly ran 84 tests while the author's machine ran 101, and the demo's
guardrail, row cap, gold comparison and token enforcement were covered
nowhere but locally.

This fixture builds a tiny sqlite database and the two JSON files that go
with it, in the same shapes the real bundle uses, so those endpoints are
exercised everywhere. Assertions that are genuinely about the Spider bundle
(that it excludes the 105MB database, that it carries 19) still guard on the
real data being present, because those are claims about the dataset.
"""

import json
import sqlite3
from pathlib import Path

import pytest

SINGERS = [
    (1, "Joe Sharp", "Netherlands", 52),
    (2, "Timbaland", "United States", 32),
    (3, "Justin Brown", "France", 29),
    (4, "Rose White", "France", 41),
    (5, "John Nizinik", "France", 43),
    (6, "Tribal King", "France", 25),
    (7, "Marina Lambrini", "Greece", 34),
    (8, "Nana Mouskouri", "Greece", 61),
]

CONCERTS = [(i, f"Concert {i}", 2014 + i % 3) for i in range(1, 9)]

QUESTIONS = [
    {
        "question": "How many singers are there?",
        "gold_sql": "SELECT count(*) FROM singer",
    },
    {
        "question": "What are the names of singers from France?",
        "gold_sql": "SELECT name FROM singer WHERE country = 'France'",
    },
    {
        "question": "What is the average age of all singers?",
        "gold_sql": "SELECT avg(age) FROM singer",
    },
    {
        "question": "List the distinct countries of the singers.",
        "gold_sql": "SELECT DISTINCT country FROM singer",
    },
    {
        "question": "What are the names of all the concerts?",
        "gold_sql": "SELECT name FROM concert",
    },
]

SCHEMA_DDL = """\
CREATE TABLE "singer" (
"singer_id" int,
"name" text,
"country" text,
"age" int,
PRIMARY KEY ("singer_id")
)

CREATE TABLE "concert" (
"concert_id" int,
"name" text,
"year" int,
PRIMARY KEY ("concert_id")
)"""


@pytest.fixture(scope="session")
def bundle(tmp_path_factory) -> Path:
    """Render a miniature serving bundle: databases/, schemas, questions."""
    root = tmp_path_factory.mktemp("serving")
    databases = root / "databases"
    databases.mkdir()

    con = sqlite3.connect(databases / "mini.sqlite")
    con.execute('CREATE TABLE "singer" (singer_id int, name text, country text, age int)')
    con.execute('CREATE TABLE "concert" (concert_id int, name text, year int)')
    con.executemany("INSERT INTO singer VALUES (?, ?, ?, ?)", SINGERS)
    con.executemany("INSERT INTO concert VALUES (?, ?, ?)", CONCERTS)
    con.commit()
    con.close()

    (root / "dev.schemas.json").write_text(json.dumps({"mini": SCHEMA_DDL}))
    (root / "dev.questions.json").write_text(
        json.dumps([{"db_id": "mini", **q} for q in QUESTIONS])
    )
    return root
