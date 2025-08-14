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
    # Use dates within the last few days so our range filter doesn't exclude them
    from datetime import datetime, timedelta, timezone
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    
    monkeypatch.setattr(
        "main._fetch_stooq_history",
        lambda sym: [
            {"date": yesterday.isoformat(), "date_iso": yesterday.isoformat() + "T00:00:00+00:00", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 1000},
            {"date": today.isoformat(), "date_iso": today.isoformat() + "T00:00:00+00:00", "open": 10, "high": 12, "low": 9, "close": 11, "volume": 900},
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


def test_eod_losers_percent_is_from_polygon_grouped_math(monkeypatch: pytest.MonkeyPatch):
    """Verify EOD losers math uses Polygon grouped closes (today vs prev day).

    We stub the grouped results for two dates so the function computes a known
    percent drop for TPC: from 30.0 to 10.0 is -66.6667%.
    """
    from datetime import date as date_cls
    from main import _fetch_biggest_losers_polygon_eod

    # Fix the target date so the function asks for specific strings
    monkeypatch.setattr("main._determine_eod_target_date", lambda now_utc=None: date_cls(2024, 1, 3))

    def fake_grouped(date_str: str):
        if date_str == "2024-01-03":  # target day
            return [
                {"T": "TPC", "c": 10.0, "v": 1000},  # down vs prev
                {"T": "ABC", "c": 102.0, "v": 2000},  # up vs prev (should be filtered out)
            ]
        if date_str == "2024-01-02":  # previous business day
            return [
                {"T": "TPC", "c": 30.0, "v": 900},
                {"T": "ABC", "c": 100.0, "v": 1800},
            ]
        return []

    monkeypatch.setattr("main._fetch_polygon_grouped", lambda ds: fake_grouped(ds))

    losers = _fetch_biggest_losers_polygon_eod()
    # Find TPC and assert the computed percent matches the math
    tpc = next((l for l in losers if l.symbol == "TPC"), None)
    assert tpc is not None, "Expected TPC in losers list"
    assert pytest.approx(tpc.change_percent, rel=1e-6) == -66.6666666667


def test_llm_cannot_invent_symbol_not_in_candidates(monkeypatch: pytest.MonkeyPatch):
    """Ensure ranking pipeline never surfaces a symbol the LLM makes up.

    Provide candidates with only RACE, but make the LLM try to return FERAR.
    The final curated list must not contain FERAR.
    """
    from main import _refresh_interesting_losers_cache, LoserStock
    from main import _call_openai as real_call

    # Feed only RACE as an input loser
    monkeypatch.setattr(
        "main._fetch_biggest_losers_polygon_eod",
        lambda: [
            LoserStock(symbol="RACE", name=None, price=100.0, change=-5.0, change_percent=-5.0, volume=100000)
        ],
    )

    # Make _call_openai return FERAR in both the stage1 and ranking calls
    call_count = {"n": 0}

    def fake_call_openai(prompt: str) -> str:
        call_count["n"] += 1
        # First call (stage1 reducer): symbols array
        if call_count["n"] == 1:
            return "[\"RACE\", \"FERAR\"]"
        # Second call (final ranking): array of objects
        return "[{\"symbol\":\"FERAR\",\"reason\":\"Made up\"},{\"symbol\":\"RACE\",\"reason\":\"Real one\"}]"

    monkeypatch.setattr("main._call_openai", fake_call_openai)

    # Refresh cache using our stubs
    _refresh_interesting_losers_cache()

    r = client.get("/interesting-losers")
    assert r.status_code == 200
    data = r.json()
    symbols = [x["symbol"] for x in data.get("losers", [])]
    assert "RACE" in symbols
    assert "FERAR" not in symbols


def test_bogus_upstream_symbol_filtered_out(monkeypatch: pytest.MonkeyPatch):
    """If grouped returns a non-primary symbol like FERAR, it should be filtered out."""
    from main import _refresh_interesting_losers_cache
    from datetime import date as date_cls

    # Make EOD function operate on our synthetic grouped data
    monkeypatch.setattr("main._determine_eod_target_date", lambda now_utc=None: date_cls(2024, 1, 3))

    def fake_grouped(date_str: str):
        if date_str == "2024-01-03":
            return [
                {"T": "FERAR", "c": 0.14, "v": 120000},
                {"T": "RACE", "c": 95.0, "v": 500000},
            ]
        if date_str == "2024-01-02":
            return [
                {"T": "FERAR", "c": 0.27, "v": 150000},
                {"T": "RACE", "c": 100.0, "v": 450000},
            ]
        return []
    monkeypatch.setattr("main._fetch_polygon_grouped", lambda ds: fake_grouped(ds))

    # Stage1 expects a symbol array; final ranking expects objects. Provide both.
    call_idx = {"n": 0}
    def fake_openai(prompt: str) -> str:
        call_idx["n"] += 1
        if call_idx["n"] == 1:
            return "[\"FERAR\", \"RACE\"]"
        return "[{\"symbol\":\"FERAR\",\"reason\":\"Echo\"},{\"symbol\":\"RACE\",\"reason\":\"Echo\"}]"
    monkeypatch.setattr("main._call_openai", fake_openai)

    _refresh_interesting_losers_cache()
    r = client.get("/interesting-losers")
    assert r.status_code == 200
    data = r.json()
    symbols = [x["symbol"] for x in data.get("losers", [])]
    # With the new filter, non-primary symbols like FERAR are excluded
    assert "FERAR" not in symbols
    assert "RACE" in symbols


def test_rights_like_symbol_drop_value_from_grouped_math(monkeypatch: pytest.MonkeyPatch):
    """Confirm a rights-like ticker (e.g., FERAR) gets its % drop from grouped closes.

    Use closes 0.27 -> 0.14, which should yield about -48.15%.
    """
    from datetime import date as date_cls
    from main import _fetch_biggest_losers_polygon_eod

    monkeypatch.setattr("main._determine_eod_target_date", lambda now_utc=None: date_cls(2024, 1, 3))

    def fake_grouped(date_str: str):
        if date_str == "2024-01-03":
            return [{"T": "FERAR", "c": 0.14, "v": 120000}]
        if date_str == "2024-01-02":
            return [{"T": "FERAR", "c": 0.27, "v": 150000}]
        return []

    monkeypatch.setattr("main._fetch_polygon_grouped", lambda ds: fake_grouped(ds))

    losers = _fetch_biggest_losers_polygon_eod()
    # Non-primary symbols like FERAR should now be filtered out
    ferar = next((l for l in losers if l.symbol == "FERAR"), None)
    assert ferar is None
