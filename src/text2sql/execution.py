"""Execute SQL against Spider sqlite databases and compare results.

Execution accuracy: a prediction is correct when running it returns the same
result set as running the gold query. Rows are compared as an unordered
multiset unless the gold query orders its output (ORDER BY), in which case
row order matters.
"""

import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

Row = tuple
EXEC_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    rows: list[Row] | None = None
    error: str | None = None


def execute_sql(db: Path, sql: str, timeout_s: float = EXEC_TIMEOUT_S) -> ExecResult:
    """Run a single query read-only, interrupting it if it exceeds timeout_s."""
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, check_same_thread=False)
    except sqlite3.Error as e:
        return ExecResult(ok=False, error=f"connect: {e}")
    # Spider databases contain text in mixed encodings; surrogateescape keeps
    # comparisons deterministic instead of raising on non-UTF-8 bytes.
    con.text_factory = lambda b: b.decode("utf-8", errors="surrogateescape")
    timer = threading.Timer(timeout_s, con.interrupt)
    timer.start()
    try:
        rows = con.execute(sql).fetchall()
        return ExecResult(ok=True, rows=[tuple(r) for r in rows])
    except sqlite3.Error as e:
        return ExecResult(ok=False, error=str(e))
    finally:
        timer.cancel()
        con.close()


def _normalize_value(v):
    # Exact float equality is too strict once queries involve AVG/SUM over
    # floating point columns; round to a tolerance instead.
    if isinstance(v, float):
        return round(v, 6)
    return v


def _normalize_rows(rows: list[Row], ordered: bool) -> list[Row]:
    normalized = [tuple(_normalize_value(v) for v in row) for row in rows]
    if ordered:
        return normalized
    return sorted(normalized, key=repr)


def gold_is_ordered(gold_sql: str) -> bool:
    return re.search(r"\border\s+by\b", gold_sql, flags=re.IGNORECASE) is not None


def results_match(pred: ExecResult, gold: ExecResult, gold_sql: str) -> bool:
    if not pred.ok or not gold.ok:
        return False
    ordered = gold_is_ordered(gold_sql)
    return _normalize_rows(pred.rows, ordered) == _normalize_rows(gold.rows, ordered)


@dataclass(frozen=True)
class PreviewResult:
    """A query's result shaped for display rather than for scoring."""

    ok: bool
    columns: list[str]
    rows: list[Row]
    truncated: bool
    error: str | None = None


def execute_preview(
    db: Path, sql: str, max_rows: int = 200, timeout_s: float = 5.0
) -> PreviewResult:
    """Run a query for display: column names, a bounded number of rows.

    Separate from execute_sql because the demo has different needs from the
    eval harness - it wants column headers and must never try to materialize
    a cross join, while scoring wants every row and nothing else. The row cap
    is enforced by fetching one extra row, so `truncated` is exact rather
    than a guess.
    """
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, check_same_thread=False)
    except sqlite3.Error as e:
        return PreviewResult(ok=False, columns=[], rows=[], truncated=False, error=f"connect: {e}")
    con.text_factory = lambda b: b.decode("utf-8", errors="surrogateescape")
    timer = threading.Timer(timeout_s, con.interrupt)
    timer.start()
    try:
        cursor = con.execute(sql)
        fetched = cursor.fetchmany(max_rows + 1)
        columns = [d[0] for d in cursor.description or []]
        return PreviewResult(
            ok=True,
            columns=columns,
            rows=[tuple(r) for r in fetched[:max_rows]],
            truncated=len(fetched) > max_rows,
        )
    except sqlite3.Error as e:
        return PreviewResult(ok=False, columns=[], rows=[], truncated=False, error=str(e))
    finally:
        timer.cancel()
        con.close()
