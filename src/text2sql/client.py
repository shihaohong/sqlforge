"""Clients for every backend this project evaluates.

Each one exposes `predict_sql(schema, question) -> sql`, so the eval harness
scores whatever is behind it without knowing the protocol: ChatClient speaks
OpenAI /chat/completions (ollama, vLLM), AnthropicClient uses the Anthropic
SDK for the frontier-API baseline and tracks tokens for $/query, and
GatewayClient goes through our own gateway - which is the only backend where
prompt construction and SQL extraction happen server-side, so scoring it
proves the production path has no skew against the offline numbers.
"""

import os
import threading
import time
from typing import ClassVar

import httpx

from .prompts import build_messages, extract_sql

# Transient transport failures (a dropped SSH-tunnel connection, a server
# restart) must not kill a 1k-example eval run; server-side 4xx/5xx still
# raises immediately because retrying those hides real bugs.
# Backoff totals ~30s, enough for a supervised tunnel to reconnect.
RETRYABLE = (httpx.TransportError,)
MAX_RETRIES = 4


class SqlClient:
    """Turns a chat backend into a SQL predictor.

    Subclasses implement `complete`; prompt rendering and SQL extraction stay
    here so every backend is scored through the same template.
    """

    def complete(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.0) -> str:
        raise NotImplementedError

    def predict_sql(self, schema: str, question: str) -> str:
        return extract_sql(self.complete(build_messages(schema, question)))


class ChatClient(SqlClient):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_connections: int = 8,
    ):
        self.model = model
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key or os.environ.get('OPENAI_API_KEY', 'none')}"},
            limits=httpx.Limits(max_connections=max_connections),
        )

    def complete(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._client.post("/chat/completions", json=payload)
                break
            except RETRYABLE:
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(2.0 * 2**attempt)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class AnthropicClient(SqlClient):
    """Claude baseline via the official Anthropic SDK.

    Sampling params are omitted on purpose: current Claude models reject
    temperature/top_p, and default settings are the fair "what would a
    production user get" comparison anyway.
    """

    # List prices per 1M tokens (input, output), for the cost baseline.
    PRICES: ClassVar[dict[str, tuple[float, float]]] = {
        "claude-opus-5": (5.00, 25.00),
        "claude-haiku-4-5": (1.00, 5.00),
        "claude-sonnet-5": (3.00, 15.00),
    }

    def __init__(self, model: str):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic()
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(self, messages: list[dict], max_tokens: int = 2048, temperature: float = 0.0) -> str:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        resp = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system, messages=chat
        )
        with self._lock:
            self.input_tokens += resp.usage.input_tokens
            self.output_tokens += resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            return ""
        return next((b.text for b in resp.content if b.type == "text"), "")

    def cost_usd(self) -> float | None:
        prices = self.PRICES.get(self.model)
        if prices is None:
            return None
        return self.input_tokens / 1e6 * prices[0] + self.output_tokens / 1e6 * prices[1]


class GatewayClient(SqlClient):
    """Our FastAPI gateway: POST a question, get guardrailed SQL back.

    A completion the guardrail rejected returns an empty string, which the
    harness then scores as a failure - the same thing a caller would see.
    """

    def __init__(self, base_url: str, timeout_s: float = 120.0, max_connections: int = 8):
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            limits=httpx.Limits(max_connections=max_connections),
        )

    def predict_sql(self, schema: str, question: str) -> str:
        payload = {"question": question, "schema": schema}
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._client.post("/v1/sql", json=payload)
                break
            except RETRYABLE:
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(2.0 * 2**attempt)
        resp.raise_for_status()
        return resp.json()["sql"] or ""


BACKENDS = {
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "gateway": "http://localhost:8080",
}


def make_client(backend: str, model: str, api_key: str | None = None):
    if backend == "claude":
        return AnthropicClient(model=model)
    if backend == "gateway":
        return GatewayClient(base_url=BACKENDS["gateway"])
    base_url = BACKENDS.get(backend, backend)  # unknown backend string = literal base URL
    return ChatClient(base_url=base_url, model=model, api_key=api_key)
