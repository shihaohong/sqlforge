import pytest

from text2sql.guardrails import check_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a FROM t",
        "select count(*) from singer where age > 20",
        "WITH x AS (SELECT 1 AS a) SELECT a FROM x",
        "SELECT a FROM t UNION SELECT b FROM u",
        "SELECT * FROM (SELECT a FROM t) s",
        "SELECT (SELECT max(b) FROM u) FROM t",
    ],
)
def test_accepts_read_only_queries(sql):
    assert check_sql(sql).ok


@pytest.mark.parametrize(
    ("sql", "code"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("DROP TABLE t", "not_select"),
        ("DELETE FROM t", "not_select"),
        ("UPDATE t SET a = 1", "not_select"),
        ("INSERT INTO t VALUES (1)", "not_select"),
        ("CREATE TABLE t (a int)", "not_select"),
        ("ALTER TABLE t ADD b int", "not_select"),
        ("PRAGMA table_info(t)", "not_select"),
        ("ATTACH DATABASE 'x.db' AS x", "not_select"),
        ("SELECT a FROM t; DROP TABLE t", "multi_statement"),
        ("sel ect ***", "parse_error"),
        # Unterminated quoting raises TokenError from the tokenizer rather
        # than ParseError from the parser; both must read as a rejection.
        ("SELECT `name FROM singer", "parse_error"),
        ("SELECT 'name FROM singer", "parse_error"),
    ],
)
def test_rejects_unsafe_or_broken_sql(sql, code):
    verdict = check_sql(sql)
    assert not verdict.ok
    assert verdict.code == code
    assert verdict.reason


def test_rejection_reason_is_human_readable():
    assert "not a SELECT" in check_sql("DROP TABLE t").reason
