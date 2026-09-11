"""Token check and rate limits for the publicly reachable endpoints.

A public gateway in front of a GPU is an abuse target: every request costs
real GPU seconds, and the Claude comparison costs money per call. So the demo
endpoints sit behind a shared token and three limits - a per-IP rate, a daily
total, and a separate, much smaller daily allowance for the paid path.

The counters live in the process, which means limits apply *per replica*: with
three gateway pods the effective ceiling is three times what is configured
here. That is a deliberate trade - a shared counter would mean running Redis
for a demo - so the configured numbers are set low enough that the multiple
still lands somewhere sane, and the real backstop is the daily cap.
"""

import hmac
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from fastapi import HTTPException, Request

TOKEN_HEADER = "X-Demo-Token"


def client_ip(request: Request) -> str:
    """Best-effort client address.

    GKE's external load balancer passes the source IP through, but honour
    X-Forwarded-For first so the same code is correct if an HTTP(S) load
    balancer or ingress is put in front later.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


#: Principals, in descending privilege. "service" skips rate limiting.
SERVICE = "service"
DEMO = "demo"
ANONYMOUS = "anonymous"


def presented_token(request: Request) -> str:
    token = request.headers.get(TOKEN_HEADER) or ""
    if token:
        return token
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[len("bearer ") :].strip()
    return ""


def authenticate(request: Request, demo_token: str, service_token: str) -> str:
    """Identify the caller as a service, a demo visitor, or nobody.

    Two credentials, because the demo token necessarily ships to browsers and
    therefore cannot be trusted with unlimited access to a GPU, while the eval
    harness and the load generator legitimately need thousands of requests at
    full speed. The service token buys exemption from rate limits, nothing
    else.

    With neither configured every caller is anonymous, which is the local
    development case. A public deployment always sets at least the demo token.
    """
    token = presented_token(request)
    # Constant-time comparison: a short-circuiting == leaks the length of the
    # matching prefix to anyone who can time the response.
    if service_token and _constant_time_equal(token, service_token):
        return SERVICE
    if demo_token and _constant_time_equal(token, demo_token):
        return DEMO
    if not demo_token and not service_token:
        return ANONYMOUS
    raise HTTPException(401, f"provide a token in {TOKEN_HEADER}")


def _constant_time_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


@dataclass
class Bucket:
    tokens: float
    updated: float


@dataclass
class RateLimiter:
    """Per-IP token bucket plus daily counters.

    `rate_per_minute` refills continuously and `burst` is how much can be
    spent at once, so a demo visitor clicking a few times in a row is fine
    while a script hammering the endpoint is not.
    """

    rate_per_minute: float
    burst: int
    daily_limit: int
    name: str = "requests"

    _buckets: dict[str, Bucket] = field(default_factory=dict)
    _day: str = ""
    _day_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def check(self, key: str) -> None:
        """Charge one request to `key`, or raise 429."""
        now = time.monotonic()
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        with self._lock:
            if today != self._day:
                self._day, self._day_count = today, 0
                self._buckets.clear()
            if self._day_count >= self.daily_limit:
                raise HTTPException(
                    429,
                    f"daily {self.name} limit reached ({self.daily_limit}); resets at 00:00 UTC",
                )

            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = Bucket(tokens=float(self.burst), updated=now)
                self._buckets[key] = bucket
            refill = (now - bucket.updated) * self.rate_per_minute / 60.0
            bucket.tokens = min(float(self.burst), bucket.tokens + refill)
            bucket.updated = now

            if bucket.tokens < 1.0:
                retry_after = int((1.0 - bucket.tokens) * 60.0 / self.rate_per_minute) + 1
                raise HTTPException(
                    429,
                    f"rate limit: {self.rate_per_minute:.0f} {self.name}/min per client;"
                    f" retry in {retry_after}s",
                    headers={"Retry-After": str(retry_after)},
                )
            bucket.tokens -= 1.0
            self._day_count += 1

    def remaining_today(self) -> int:
        with self._lock:
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            if today != self._day:
                return self.daily_limit
            return max(0, self.daily_limit - self._day_count)
