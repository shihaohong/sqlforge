"""Prompt construction shared by evaluation, training, and serving.

Keeping one canonical template here is deliberate: train/serve skew in prompt
format is one of the easiest ways to silently lose accuracy.
"""

import re

SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. Given a SQLite database schema and a question, "
    "respond with a single SQLite SELECT query that answers the question. "
    "Respond with only the SQL query, no explanation and no markdown."
)

USER_TEMPLATE = """\
Database schema:

{schema}

Question: {question}

SQL:"""


def build_messages(schema: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(schema=schema, question=question)},
    ]


def extract_sql(completion: str) -> str:
    """Pull a SQL query out of a model completion.

    Models (especially instruction-tuned base models) wrap output in code
    fences or prepend chatter despite instructions; recover the query rather
    than failing the example on formatting.
    """
    text = completion.strip()
    fence = re.search(r"```(?:sql)?\s*(.+?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    match = re.search(r"\b(select|with)\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start() :]
    # Keep only the first statement.
    text = text.split(";")[0].strip()
    return text
