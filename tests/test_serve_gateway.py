"""The CLI entrypoint's configuration wiring.

Worth testing on its own: the app object is configured by whatever the CLI
hands it, so a flag or an environment variable that silently fails to reach
`Settings` produces a healthy process talking to the wrong address.
"""

import importlib.util
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "serve_gateway.py"


def load_cli(monkeypatch, env: dict[str, str]):
    """Import the script fresh under `env` and return (invoke, captured).

    The module reads the environment at import time (typer evaluates option
    defaults then), so the import has to happen after the env is set. uvicorn
    and create_app are stubbed, so invoking the CLI configures but serves
    nothing.
    """
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    spec = importlib.util.spec_from_file_location("serve_gateway_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    captured: dict = {}
    monkeypatch.setattr(module.uvicorn, "run", lambda app, **kw: None)
    monkeypatch.setattr(module, "create_app", lambda settings: captured.update(settings=settings))

    # The script exposes a plain function through typer.run(), so wrap it to
    # exercise the same option parsing the entrypoint uses.
    app = typer.Typer()
    app.command()(module.main)

    def invoke(args=()):
        result = CliRunner().invoke(app, list(args))
        assert result.exit_code == 0, result.output
        return captured["settings"]

    return invoke, captured


@pytest.fixture
def upstream():
    return "http://vllm.sqlforge.svc.cluster.local:8000/v1"


def test_reads_the_upstream_from_the_environment(monkeypatch, upstream):
    invoke, _ = load_cli(monkeypatch, {"SQLFORGE_UPSTREAM": upstream})
    assert invoke().upstream_url == upstream


def test_reads_every_setting_from_the_environment(monkeypatch):
    invoke, _ = load_cli(
        monkeypatch,
        {
            "SQLFORGE_UPSTREAM": "http://vllm:8000/v1",
            "SQLFORGE_MODEL": "sqlforge-3b",
            "SQLFORGE_TIMEOUT_S": "17",
            "SQLFORGE_MAX_TOKENS": "128",
            "SQLFORGE_SCHEMA_CACHE": "/app/data/serving/dev.schemas.json",
            # Deliberately included: these have no CLI flag, so they only
            # arrive if the entrypoint keeps the environment-derived defaults
            # instead of building a fresh Settings from its flags.
            "SQLFORGE_DEMO_TOKEN": "public-token",
            "SQLFORGE_SERVICE_TOKEN": "service-token",
            "SQLFORGE_DAILY_LIMIT": "77",
        },
    )
    settings = invoke()
    assert settings.upstream_url == "http://vllm:8000/v1"
    assert settings.model == "sqlforge-3b"
    assert settings.timeout_s == 17.0
    assert settings.max_tokens == 128
    assert str(settings.schema_cache) == "/app/data/serving/dev.schemas.json"
    assert settings.demo_token == "public-token"
    assert settings.service_token == "service-token"
    assert settings.daily_limit == 77


def test_a_flag_does_not_discard_settings_that_have_no_flag(monkeypatch):
    """The public endpoint's token must survive any combination of flags."""
    invoke, _ = load_cli(monkeypatch, {"SQLFORGE_DEMO_TOKEN": "public-token"})
    settings = invoke(["--upstream", "http://localhost:9/v1", "--port", "9999"])
    assert settings.upstream_url == "http://localhost:9/v1"
    assert settings.demo_token == "public-token"


def test_explicit_flag_overrides_the_environment(monkeypatch, upstream):
    invoke, _ = load_cli(monkeypatch, {"SQLFORGE_UPSTREAM": upstream})
    assert invoke(["--upstream", "http://localhost:11434/v1"]).upstream_url == (
        "http://localhost:11434/v1"
    )
