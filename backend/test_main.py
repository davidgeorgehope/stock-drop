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


def test_quote_endpoint_ok(monkeypatch: pytest.MonkeyPatch):
    # Stooq-only path: stub history to deterministic values
    from main import _fetch_stooq_history

    monkeypatch.setattr(
        "main._fetch_stooq_history",
        lambda sym: [
            {"date": "2024-01-01", "date_iso": "2024-01-01T00:00:00+00:00", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 1000},
            {"date": "2024-01-02", "date_iso": "2024-01-02T00:00:00+00:00", "open": 10, "high": 12, "low": 9, "close": 11, "volume": 900},
        ],
    )
    r = client.get("/quote/TEST")
    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "TEST"
    assert data["price"] == 11


def test_chart_endpoint_ok(monkeypatch: pytest.MonkeyPatch):
    # Stooq-only path: stub history to deterministic series
    monkeypatch.setattr(
        "main._fetch_stooq_history",
        lambda sym: [
            {"date": "2024-01-01", "date_iso": "2024-01-01T00:00:00+00:00", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 1000},
            {"date": "2024-01-02", "date_iso": "2024-01-02T00:00:00+00:00", "open": 10, "high": 12, "low": 9, "close": 11, "volume": 900},
        ],
    )
    r = client.get("/chart/TSLA?range=1mo&interval=1d")
    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "TSLA"
    assert data["closes"] == [10, 11]

