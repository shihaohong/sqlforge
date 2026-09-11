"""Output guardrails for generated SQL.

The model is not trusted to produce safe or even parseable SQL, so every
completion passes through here before the gateway returns it. Two checks:

1. It parses as SQLite (sqlglot), as exactly one statement.
2. It is read-only: a SELECT (or a UNION/CTE over SELECTs) and nothing else.

Rejecting rather than repairing is deliberate: a caller that receives a query
can trust it, and a rejection is a countable, alertable event.
"""

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

DIALECT = "sqlite"

# Statement types that mutate data or schema. sqlglot parses most unsupported
# or non-query statements (PRAGMA, ATTACH, VACUUM, ...) into exp.Command, so
# banning that catches the long tail without enumerating it.
FORBIDDEN = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
)
ALLOWED_ROOTS = (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.Subquery)


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str | None = None

    #: Short machine-readable label, used as a Prometheus metric dimension.
    code: str = "ok"


OK = Verdict(ok=True)


def check_sql(sql: str) -> Verdict:
    """Decide whether a generated query may be returned to a caller."""
    sql = sql.strip()
    if not sql:
        return Verdict(False, "model produced no SQL", "empty")
    try:
        statements = [s for s in sqlglot.parse(sql, dialect=DIALECT) if s is not None]
    # SqlglotError, not ParseError: an unterminated quote or backtick raises
    # TokenError from the tokenizer, which is a sibling class, and a model
    # does produce those. Any failure to read the SQL is a rejection.
    except SqlglotError as e:
        return Verdict(False, f"unparseable SQL: {e}", "parse_error")
    if len(statements) != 1:
        return Verdict(False, f"expected one statement, got {len(statements)}", "multi_statement")

    stmt = statements[0]
    if not isinstance(stmt, ALLOWED_ROOTS):
        return Verdict(False, f"not a SELECT statement: {type(stmt).__name__}", "not_select")
    for node_type in FORBIDDEN:
        if isinstance(stmt, node_type) or next(stmt.find_all(node_type), None) is not None:
            return Verdict(
                False, f"write operation is not allowed: {node_type.__name__}", "write_op"
            )
    # SELECT ... INTO writes a new table, so it is a write despite the root node.
    if stmt.args.get("into") is not None:
        return Verdict(False, "SELECT ... INTO is not allowed", "write_op")
    return OK
