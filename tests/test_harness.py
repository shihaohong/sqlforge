from text2sql.execution import ExecResult, gold_is_ordered, results_match
from text2sql.prompts import extract_sql


class TestExtractSql:
    def test_plain_query(self):
        assert extract_sql("SELECT * FROM t") == "SELECT * FROM t"

    def test_code_fence(self):
        assert extract_sql("```sql\nSELECT a FROM t\n```") == "SELECT a FROM t"

    def test_leading_chatter(self):
        text = "Here is the query:\nSELECT a FROM t"
        assert extract_sql(text) == "SELECT a FROM t"

    def test_trailing_semicolon_and_second_statement(self):
        assert extract_sql("SELECT a FROM t; DROP TABLE t;") == "SELECT a FROM t"

    def test_with_cte(self):
        text = "WITH x AS (SELECT 1) SELECT * FROM x"
        assert extract_sql(text) == text


class TestResultsMatch:
    def test_unordered_match(self):
        pred = ExecResult(ok=True, rows=[(2,), (1,)])
        gold = ExecResult(ok=True, rows=[(1,), (2,)])
        assert results_match(pred, gold, "SELECT a FROM t")

    def test_ordered_mismatch(self):
        pred = ExecResult(ok=True, rows=[(2,), (1,)])
        gold = ExecResult(ok=True, rows=[(1,), (2,)])
        assert not results_match(pred, gold, "SELECT a FROM t ORDER BY a")

    def test_ordered_match(self):
        pred = ExecResult(ok=True, rows=[(1,), (2,)])
        gold = ExecResult(ok=True, rows=[(1,), (2,)])
        assert results_match(pred, gold, "SELECT a FROM t ORDER BY a")

    def test_duplicate_rows_are_multiset(self):
        pred = ExecResult(ok=True, rows=[(1,), (1,)])
        gold = ExecResult(ok=True, rows=[(1,)])
        assert not results_match(pred, gold, "SELECT a FROM t")

    def test_float_tolerance(self):
        pred = ExecResult(ok=True, rows=[(0.3000000004,)])
        gold = ExecResult(ok=True, rows=[(0.3,)])
        assert results_match(pred, gold, "SELECT avg(a) FROM t")

    def test_error_never_matches(self):
        pred = ExecResult(ok=False, error="syntax error")
        gold = ExecResult(ok=True, rows=[])
        assert not results_match(pred, gold, "SELECT a FROM t")

    def test_mixed_types_sort(self):
        # sqlite columns can mix NULL, numbers, and text; sorting must not raise.
        pred = ExecResult(ok=True, rows=[(None,), (1,), ("a",)])
        gold = ExecResult(ok=True, rows=[("a",), (None,), (1,)])
        assert results_match(pred, gold, "SELECT a FROM t")


def test_gold_is_ordered():
    assert gold_is_ordered("SELECT a FROM t ORDER BY a")
    assert not gold_is_ordered("SELECT a FROM orderby_table")
