import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

import os

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("OPENAI_MODEL", "gpt-4o-mini")

from main import app  # noqa: E402
from main import QuoteResponse, ChartResponse  # noqa: E402


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


def test_og_image_endpoint_png_ok(monkeypatch: pytest.MonkeyPatch):
    # Stub quote and chart to deterministic values
    monkeypatch.setattr(
        "main._fetch_yahoo_quote",
        lambda sym: QuoteResponse(
            symbol=sym,
            price=100.0,
            change=-5.0,
            change_percent=-4.76,
            currency="USD",
            market_time=None,
            market_state="CLOSED",
            name="Test Corp",
        ),
    )
    monkeypatch.setattr(
        "main._fetch_yahoo_chart",
        lambda sym, range_, interval: ChartResponse(
            symbol=sym,
            range=range_,
            interval=interval,
            timestamps=[1, 2, 3, 4],
            opens=[100, 99, 98, 97],
            highs=[101, 100, 99, 98],
            lows=[99, 98, 97, 96],
            closes=[100, 99, 97, 95],
            volumes=[1000, 900, 800, 700],
        ),
    )

    r = client.get("/og-image/TEST.png")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/png")
    assert r.headers.get("ETag")
    assert r.headers.get("Cache-Control")
    # PNG signature
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_share_page_meta_and_redirect(monkeypatch: pytest.MonkeyPatch):
    # Ensure canonical uses PUBLIC_WEB_ORIGIN
    monkeypatch.setenv("PUBLIC_WEB_ORIGIN", "http://localhost:5173")
    # Make description deterministic and avoid network
    monkeypatch.setattr("main._build_share_description", lambda sym: "Preview description")

    r = client.get("/s/ESTC")
    assert r.status_code == 200
    html = r.text
    # Canonical should point to the frontend origin with stock param
    assert '<link rel="canonical" href="http://localhost:5173/?stock=ESTC" />' in html
    # Basic OG/Twitter tags present
    assert 'meta property="og:title"' in html
    assert 'meta name="twitter:card" content="summary_large_image"' in html
    # Reference to image path present (absolute or testserver)
    assert "/og-image/ESTC.png" in html
