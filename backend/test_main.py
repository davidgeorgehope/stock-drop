import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

import os

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("OPENAI_MODEL", "gpt-4o-mini")

from main import app  # noqa: E402


client = TestClient(app)


def test_root_health():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True


def test_analyze_requires_symbols():
    r = client.post("/analyze", json={"symbols": "  "})
    assert r.status_code == 400


def test_analyze_basic_flow(monkeypatch: pytest.MonkeyPatch):
    # Stub search and LLM for determinism
    from main import _search_news_for_symbol, _call_openai

    monkeypatch.setattr(
        "main._search_news_for_symbol",
        lambda sym, days, max_results: [],
    )
    monkeypatch.setattr(
        "main._call_openai",
        lambda prompt: "Test summary",
    )

    r = client.post("/analyze", json={"symbols": "AAPL"})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert data["results"][0]["symbol"] == "AAPL"
    assert isinstance(data["results"][0]["summary"], str)


def test_quote_endpoint_graceful(monkeypatch: pytest.MonkeyPatch):
    # Force upstream 429
    class DummyResp:
        status_code = 429

        def json(self) -> Any:
            return {}

    monkeypatch.setattr("requests.get", lambda *a, **kw: DummyResp())
    r = client.get("/quote/AAPL")
    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "AAPL"
    # Should be empty fields when rate limited
    assert data.get("price") is None


def test_chart_endpoint_graceful(monkeypatch: pytest.MonkeyPatch):
    # Force upstream 429
    class DummyResp:
        status_code = 429

        def json(self) -> Any:
            return {}

    monkeypatch.setattr("requests.get", lambda *a, **kw: DummyResp())
    r = client.get("/chart/TSLA?range=1mo&interval=1d")
    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "TSLA"
    assert data["timestamps"] == []

