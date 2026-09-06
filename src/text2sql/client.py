"""Chat completion clients for every backend this project touches.

ChatClient speaks the OpenAI /chat/completions protocol (ollama locally,
vLLM at M3). AnthropicClient uses the official Anthropic SDK for the
frontier-API baseline and tracks token usage so we can report $/query.
"""

import os
import threading

import httpx


class ChatClient:
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
        resp = self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class AnthropicClient:
    """Claude baseline via the official Anthropic SDK.

    Sampling params are omitted on purpose: current Claude models reject
    temperature/top_p, and default settings are the fair "what would a
    production user get" comparison anyway.
    """

    # List prices per 1M tokens (input, output), for the cost baseline.
    PRICES = {
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


BACKENDS = {
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
}


def make_client(backend: str, model: str, api_key: str | None = None):
    if backend == "claude":
        return AnthropicClient(model=model)
    base_url = BACKENDS.get(backend, backend)  # unknown backend string = literal base URL
    return ChatClient(base_url=base_url, model=model, api_key=api_key)
