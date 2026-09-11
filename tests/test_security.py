"""Token handling and rate limiting.

These guard a public endpoint in front of a GPU, so the tests are about what
must not happen: an unauthenticated call getting through, or one client
spending the whole budget.
"""

import pytest
from fastapi import HTTPException, Request

from text2sql.security import ANONYMOUS, DEMO, SERVICE, RateLimiter, authenticate, client_ip


def make_request(headers: dict[str, str] | None = None, ip: str = "203.0.113.7") -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "headers": raw, "client": (ip, 443)})


class TestAuthenticate:
    def test_service_token_wins(self):
        request = make_request({"X-Demo-Token": "svc"})
        assert authenticate(request, demo_token="demo", service_token="svc") == SERVICE

    def test_demo_token_is_recognized(self):
        request = make_request({"X-Demo-Token": "demo"})
        assert authenticate(request, demo_token="demo", service_token="svc") == DEMO

    def test_bearer_header_is_accepted(self):
        request = make_request({"Authorization": "Bearer demo"})
        assert authenticate(request, demo_token="demo", service_token="") == DEMO

    def test_no_configured_tokens_means_open(self):
        assert authenticate(make_request(), demo_token="", service_token="") == ANONYMOUS

    @pytest.mark.parametrize("headers", [{}, {"X-Demo-Token": ""}, {"X-Demo-Token": "wrong"}])
    def test_rejects_missing_or_wrong_token(self, headers):
        with pytest.raises(HTTPException) as excinfo:
            authenticate(make_request(headers), demo_token="demo", service_token="svc")
        assert excinfo.value.status_code == 401

    def test_a_configured_service_token_alone_still_closes_the_door(self):
        with pytest.raises(HTTPException):
            authenticate(make_request(), demo_token="", service_token="svc")


class TestClientIp:
    def test_prefers_forwarded_for(self):
        request = make_request({"X-Forwarded-For": "198.51.100.4, 10.0.0.1"}, ip="10.0.0.1")
        assert client_ip(request) == "198.51.100.4"

    def test_falls_back_to_the_socket(self):
        assert client_ip(make_request(ip="203.0.113.9")) == "203.0.113.9"


class TestRateLimiter:
    def test_allows_up_to_the_burst_then_refuses(self):
        limiter = RateLimiter(rate_per_minute=60, burst=3, daily_limit=100)
        for _ in range(3):
            limiter.check("a")
        with pytest.raises(HTTPException) as excinfo:
            limiter.check("a")
        assert excinfo.value.status_code == 429
        assert "Retry-After" in excinfo.value.headers

    def test_clients_are_limited_independently(self):
        limiter = RateLimiter(rate_per_minute=60, burst=1, daily_limit=100)
        limiter.check("a")
        limiter.check("b")  # b has its own bucket
        with pytest.raises(HTTPException):
            limiter.check("a")

    def test_daily_limit_applies_across_clients(self):
        limiter = RateLimiter(rate_per_minute=600, burst=10, daily_limit=2)
        limiter.check("a")
        limiter.check("b")
        with pytest.raises(HTTPException) as excinfo:
            limiter.check("c")
        assert "daily" in excinfo.value.detail
        assert limiter.remaining_today() == 0

    def test_tokens_refill_over_time(self, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr("text2sql.security.time.monotonic", lambda: clock["t"])
        limiter = RateLimiter(rate_per_minute=60, burst=1, daily_limit=100)

        limiter.check("a")
        with pytest.raises(HTTPException):
            limiter.check("a")

        clock["t"] += 1.0  # one token per second at 60/min
        limiter.check("a")

    def test_remaining_today_counts_down(self):
        limiter = RateLimiter(rate_per_minute=600, burst=5, daily_limit=5)
        assert limiter.remaining_today() == 5
        limiter.check("a")
        assert limiter.remaining_today() == 4
