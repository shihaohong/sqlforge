"""The prompt template is a frozen contract, and this test is the lock.

The model's weights were trained against these exact strings. Training,
evaluation, and serving all render prompts through `build_messages`, so an
innocuous-looking edit here - a reworded instruction, a stray newline, a
changed label - silently moves every served prompt off the distribution the
model was fine-tuned on, and the only symptom is lower accuracy nobody
attributes to the edit.

**If this test fails, the change is not wrong, but it is not free.** Either
revert it, or update the expected strings here *and* re-run the eval gate
(`scripts/eval_gate.py`) to measure what the new template costs. A failure
here is a prompt that no longer matches the weights.
"""

from text2sql.guardrails import check_sql
from text2sql.prompts import build_messages, extract_sql

SCHEMA = 'CREATE TABLE "singer" (\n"id" int,\n"name" text\n)'

EXPECTED_SYSTEM = (
    "You are a text-to-SQL assistant. Given a SQLite database schema and a question, "
    "respond with a single SQLite SELECT query that answers the question. "
    "Respond with only the SQL query, no explanation and no markdown."
)

EXPECTED_USER = """\
Database schema:

CREATE TABLE "singer" (
"id" int,
"name" text
)

Question: How many singers are there?

SQL:"""


class TestPromptContract:
    def test_rendered_prompt_is_unchanged(self):
        messages = build_messages(SCHEMA, "How many singers are there?")
        assert messages == [
            {"role": "system", "content": EXPECTED_SYSTEM},
            {"role": "user", "content": EXPECTED_USER},
        ]

    def test_schema_is_passed_through_verbatim(self):
        """No normalization, so the model sees the DDL its SQL will run against."""
        weird = 'CREATE TABLE "t" (\n\t"a b" TEXT DEFAULT \'x\'\n)'
        user = build_messages(weird, "q")[1]["content"]
        assert weird in user

    def test_question_is_not_reworded_or_punctuated(self):
        question = "how many singers   are there"
        assert f"Question: {question}\n" in build_messages(SCHEMA, question)[1]["content"]


class TestExtractionContract:
    """What the harness accepts as a model's answer, also frozen.

    Loosening extraction inflates scores without improving the model, and
    tightening it fails examples the model actually got right - either way the
    accuracy numbers in PLAN.md stop being comparable across runs.
    """

    def test_plain_completion(self):
        assert extract_sql("SELECT count(*) FROM singer") == "SELECT count(*) FROM singer"

    def test_fenced_completion(self):
        assert extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"

    def test_leading_prose_is_dropped(self):
        assert extract_sql("Sure! Here you go:\nSELECT 1") == "SELECT 1"

    def test_only_the_first_statement_survives(self):
        assert extract_sql("SELECT 1; DROP TABLE t;") == "SELECT 1"

    def test_cte_is_kept_whole(self):
        query = "WITH x AS (SELECT 1) SELECT * FROM x"
        assert extract_sql(query) == query

    def test_empty_completion_stays_empty(self):
        assert extract_sql("") == ""

    def test_a_refusal_never_becomes_valid_sql(self):
        """Prose is returned verbatim, deliberately.

        Extraction keeps whatever the model said when it finds no query, so
        the run's jsonl records the actual completion instead of an empty
        string - far easier to diagnose. The invariant that matters is the
        one below: it must never pass the guardrail.
        """
        refusal = extract_sql("I cannot answer that.")
        assert refusal == "I cannot answer that."
        assert not check_sql(refusal).ok
