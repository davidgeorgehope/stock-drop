from datetime import datetime, timezone, date, timedelta, time as dtime
import os
from typing import List, Optional, Union
import asyncio
import threading

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI
from dotenv import load_dotenv
import requests
import time
import csv
from io import StringIO, BytesIO
from hashlib import md5
from PIL import Image, ImageDraw, ImageFont
from zoneinfo import ZoneInfo
import textwrap


def _rank_interesting_losers(candidates: List["LoserStock"], top_n: int = 10) -> List["InterestingLoser"]:
    """Use LLM + lightweight headline search to rank losers by 'newsworthiness'."""
    if not candidates:
        return []
    # Fetch minimal headlines for each candidate (cap to avoid rate limiting)
    enriched: List[tuple["LoserStock", List["SourceItem"]]] = []
    for c in candidates[: min(60, len(candidates))]:
        try:
            news = _search_news_for_symbol(c.symbol, days=3, max_results=4)
        except Exception:
            news = []
        enriched.append((c, news))
    # Build LLM prompt summarizing each candidate
    lines: List[str] = []
    for stock, news in enriched:
        snippet_titles = "; ".join([n.title for n in news[:3] if n.title])
        chg = f"{stock.change_percent:.2f}%" if isinstance(stock.change_percent, (int, float)) else "n/a"
        lines.append(f"{stock.symbol} ({chg}) — {snippet_titles}")
    catalog = "\n".join(lines)
    prompt = (
        "You are a sharp markets editor. From the following decliners, pick the most newsworthy 10.\n"
        "Prefer names with clear catalysts (earnings, guidance, downgrades, litigation, macro, product news) and broad interest.\n"
        "Avoid microcaps and illiquid names unless there is major news.\n"
        "Return a JSON array of objects with: symbol, reason (1 sentence).\n\n"
        f"Candidates (symbol, today's % change, top headlines):\n{catalog}\n\n"
        "Return strictly JSON."
    )
    try:
        raw = _call_openai(prompt)
        import json, re
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        arr = json.loads(m.group(0) if m else raw)
        picked: List[InterestingLoser] = []
        sym_to_stock = {c.symbol: c for c, _ in enriched}
        for item in arr:
            if not isinstance(item, dict):
                continue
            sym = (item.get("symbol") or "").strip().upper()
            if not sym or sym not in sym_to_stock:
                continue
            st = sym_to_stock[sym]
            picked.append(InterestingLoser(**st.dict(), reason=item.get("reason")))
        return picked[:top_n]
    except Exception as e:
        print(f"❌ LLM ranking failed: {e}")
        return []


def _get_interesting_losers(candidates: List["LoserStock"], top_n: int = 10) -> List["InterestingLoser"]:
    global INTERESTING_LOSERS_CACHE
    now = time.time()
    with INTERESTING_LOSERS_LOCK:
        if INTERESTING_LOSERS_CACHE and (now - INTERESTING_LOSERS_CACHE[0]) < INTERESTING_LOSERS_CACHE_TTL_SECONDS:
            return INTERESTING_LOSERS_CACHE[1]
    ranked = _rank_interesting_losers(candidates, top_n)
    with INTERESTING_LOSERS_LOCK:
        INTERESTING_LOSERS_CACHE = (time.time(), ranked)
    return ranked


def _refresh_interesting_losers_cache():
    """Refresh the interesting losers cache by computing EOD losers and ranking."""
    try:
        full = _fetch_biggest_losers_polygon_eod()
        full.sort(key=lambda l: l.change_percent or 0)
        # Stage 1: reduce to 30 via LLM without headlines
        stage1 = full[:300]
        try:
            tick_lines = [f"{s.symbol} {s.change_percent:.2f}%" for s in stage1 if isinstance(s.change_percent, (int, float))]
            prompt = (
                "You are a markets editor. From this list of decliners, pick the 30 most likely to be newsworthy today.\n"
                "Prefer recognizable names, earnings/guidance/catalyst windows, sector moves, litigation, macro.\n"
                "Return a JSON array of symbols only.\n\n" + "\n".join(tick_lines)
            )
            raw = _call_openai(prompt)
            import json, re
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            arr = json.loads(m.group(0) if m else raw)
            pickset = {str(x).strip().upper() for x in arr if isinstance(x, (str,))}
            reduced = [s for s in stage1 if s.symbol in pickset][:30]
        except Exception:
            reduced = stage1[:30]
        ranked = _rank_interesting_losers(reduced, 15)
        # Final Stooq sanity check to weed out extreme mismatches (e.g., -66% vs -1%)
        try:
            tol_pp = float(os.getenv("LOSERS_FINAL_STOOQ_TOLERANCE_PPTS", "25.0"))
            enabled = os.getenv("LOSERS_FINAL_STOOQ_CHECK", "1") not in {"0", "false", "False"}
            if enabled:
                ranked = _filter_ranked_losers_by_stooq(ranked, tol_pp)
        except Exception:
            pass
        ranked = ranked[:12]
        global INTERESTING_LOSERS_CACHE
        with INTERESTING_LOSERS_LOCK:
            INTERESTING_LOSERS_CACHE = (time.time(), ranked)
        print(f"🔄 Interesting losers cache refreshed with {len(ranked)} items")
    except Exception as e:
        print(f"❌ Failed to refresh interesting losers cache: {e}")

from datetime import datetime, timezone, date, timedelta, time as dtime
import os
from typing import List, Optional, Union
import asyncio
import threading

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI
from dotenv import load_dotenv
import requests
import time
import csv
from io import StringIO, BytesIO
from hashlib import md5
from PIL import Image, ImageDraw, ImageFont
from zoneinfo import ZoneInfo
import textwrap

# Load env vars from a local .env if present (dev convenience)
load_dotenv()


app = FastAPI(
    title="Why Is The Stock Plummeting?",
    description="Searches the web and uses an LLM to explain (humorously) why your favorite stock face-planted.",
    version="0.1.0",
)


allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
public_origin = os.getenv("PUBLIC_WEB_ORIGIN")
if public_origin:
    allowed_origins.append(public_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    print("🚀 Application starting up...")
    # Populate caches at startup in threads so FastAPI startup completes
    def init_cache():
        try:
            _refresh_biggest_losers_cache()
        except Exception as e:
            print(f"Startup cache init failed: {e}")
        try:
            _refresh_interesting_losers_cache()
        except Exception as e:
            print(f"Startup interesting cache init failed: {e}")
    threading.Thread(target=init_cache, daemon=True).start()

    # Start daily/background refresh for both caches
    def periodic_refresh():
        while True:
            time.sleep(INTERESTING_LOSERS_CACHE_TTL_SECONDS)
            try:
                _refresh_biggest_losers_cache()
            except Exception as e:
                print(f"Periodic refresh failed: {e}")
            try:
                _refresh_interesting_losers_cache()
            except Exception as e:
                print(f"Periodic interesting refresh failed: {e}")
    threading.Thread(target=periodic_refresh, daemon=True).start()
    print("✅ Startup initialization queued")


class AnalyzeRequest(BaseModel):
    symbols: Union[str, List[str]]
    days: int = 7
    max_results: int = 8
    tone: str = "humorous"


class SourceItem(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None
    published: Optional[str] = None


class SymbolAnalysis(BaseModel):
    symbol: str
    summary: str
    sources: List[SourceItem]


class AnalyzeResponse(BaseModel):
    results: List[SymbolAnalysis]


def _ensure_list_symbols(input_symbols: Union[str, List[str]]) -> List[str]:
    if isinstance(input_symbols, list):
        return [s.strip().upper() for s in input_symbols if s and s.strip()]
    # Comma or whitespace separated
    separators = [",", " "]
    symbols: List[str] = []
    current = input_symbols
    for sep in separators:
        if sep in current:
            parts = [p for p in current.split(sep)]
            symbols = [p.strip().upper() for p in parts if p and p.strip()]
            break
    if not symbols:
        symbols = [current.strip().upper()] if current.strip() else []
    # Deduplicate, preserve order
    seen = set()
    ordered: List[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def _search_news_for_symbol(symbol: str, days: int, max_results: int) -> List[SourceItem]:
    # Query DuckDuckGo News with tiny on-disk cache to limit rate usage
    try:
        if NEWS_CACHE_DIR:
            os.makedirs(NEWS_CACHE_DIR, exist_ok=True)
            cache_key = f"{symbol.upper()}_{days}_{max_results}.json"
            cache_path = os.path.join(NEWS_CACHE_DIR, cache_key)
            if os.path.exists(cache_path):
                try:
                    mtime = os.path.getmtime(cache_path)
                    if time.time() - mtime < NEWS_CACHE_TTL_SECONDS:
                        import json as _json
                        with open(cache_path, "r", encoding="utf-8") as f:
                            cached = _json.load(f)
                        return [SourceItem(**it) for it in cached]
                except Exception:
                    pass
    except Exception:
        pass
    try:
        # Import here to avoid hard dependency at module import time
        from duckduckgo_search import DDGS  # type: ignore
    except Exception as import_err:
        print(f"DuckDuckGo search unavailable: {import_err}")
        return []

    query = f"{symbol} stock"
    timelimit = f"d{max(1, min(days, 30))}"
    items: List[SourceItem] = []

    attempts = 0
    while attempts < 3 and len(items) == 0:
        attempts += 1
        try:
            with DDGS() as ddgs:
                for n in ddgs.news(
                    query,
                    region="us-en",
                    safesearch="moderate",
                    timelimit=timelimit,
                    max_results=max_results,
                ):
                    items.append(
                        SourceItem(
                            title=(n.get("title") or ""),
                            url=(n.get("url") or n.get("link") or ""),
                            snippet=(n.get("excerpt") or n.get("body")),
                            published=n.get("date"),
                        )
                    )
        except Exception as e:
            print(f"DDG news search attempt {attempts} failed for {symbol}: {e}")
            time.sleep(0.8 * attempts)

    cleaned = [i for i in items if i.url]
    # Save to cache
    try:
        if cleaned and NEWS_CACHE_DIR:
            import json as _json
            os.makedirs(NEWS_CACHE_DIR, exist_ok=True)
            cache_key = f"{symbol.upper()}_{days}_{max_results}.json"
            cache_path = os.path.join(NEWS_CACHE_DIR, cache_key)
            with open(cache_path, "w", encoding="utf-8") as f:
                _json.dump([c.dict() for c in cleaned], f)
    except Exception:
        pass
    return cleaned


def _build_llm_prompt(
    symbol: str,
    sources: List[SourceItem],
    tone: str,
    price_context: Optional[str] = None,
    max_words: Optional[int] = None,
) -> str:
    headline_lines = [f"- {s.title} ({s.url})" for s in sources[:10]]
    snippets = [f"{s.title}: {s.snippet}" for s in sources if s.snippet]
    headlines_block = "\n".join(headline_lines) or "(no recent articles found)"
    snippets_block = "\n".join(snippets[:10]) or "(no snippets)"
    price_block = f"\nRecent price context:\n{price_context}\n" if price_context else "\n"
    prompt = (
        f"You are a witty yet insightful markets analyst. The user asks: Why did {symbol} drop?\n"
        f"Use only the following recent headlines and snippets as context. Summarize the likely reasons "
        f"and deliver it in a concise, {tone} tone. Avoid making up facts beyond the provided links.\n\n"
        f"Headlines:\n{headlines_block}\n\nSnippets:\n{snippets_block}\n{price_block}\n"
        f"Output guidelines:\n"
        f"- 2–4 short paragraphs max\n- Include 1–2 tongue-in-cheek jokes\n- If uncertainty remains, say so\n"
    )
    if max_words is not None:
        prompt += (
            f"- HARD LIMIT: Keep the entire response under {max_words} words; fewer is better for a social preview image.\n"
            f"- Prioritize brevity and clarity. Use short sentences.\n"
        )
    return prompt


def _call_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        # Fallback: offline snark mode
        return (
            "LLM key missing, so here's the CliffNotes version: likely earnings jitters, guidance hiccups, "
            "analyst downgrades, or a general case of 'the market woke up on the wrong side of the bed.'"
        )
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a humorous financial analyst who absolutely roasts the stock market and still gets the facts right.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"OpenAI call failed: {e}")
        return (
            "Couldn't reach the LLM this time. Still, the ingredients for a drop are classic: earnings, "
            "guidance, downgrades, macro dramas, or just vibes."
        )


class QuoteResponse(BaseModel):
    symbol: str
    price: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    currency: Optional[str] = None
    market_time: Optional[str] = None
    market_state: Optional[str] = None
    name: Optional[str] = None


class ChartResponse(BaseModel):
    symbol: str
    range: str
    interval: str
    timestamps: List[int]
    opens: List[Optional[float]]
    highs: List[Optional[float]]
    lows: List[Optional[float]]
    closes: List[Optional[float]]
    volumes: List[Optional[int]]


class LoserStock(BaseModel):
    symbol: str
    name: Optional[str] = None
    price: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None


class BiggestLosersResponse(BaseModel):
    losers: List[LoserStock]
    last_updated: str
    session: str | None = None


class InterestingLoser(LoserStock):
    reason: Optional[str] = None


class InterestingLosersResponse(BaseModel):
    losers: List[InterestingLoser]
    last_updated: str
    session: str | None = None


QUOTE_CACHE_TTL_SECONDS = int(os.getenv("QUOTE_CACHE_TTL_SECONDS", "60"))
QUOTE_CACHE: dict[str, tuple[float, QuoteResponse]] = {}


def _fetch_daily_history_prefer_stooq(symbol: str) -> List[dict]:
    """Return recent daily history rows, preferring Stooq; fallback to Polygon aggs.

    Row shape: {date, date_iso, open, high, low, close, volume}
    """
    stooq = _fetch_stooq_history(symbol)
    if stooq:
        return stooq
    # Fallback to cached EOD series built from grouped data (2-point series)
    if symbol in EOD_SERIES_CACHE:
        return EOD_SERIES_CACHE.get(symbol) or []
    # If allowed, try Polygon daily aggs (may not be available on free tier)
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        return []
    try:
        # Compute date range (UTC)
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=90)
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}"
            f"?adjusted=true&sort=asc&limit=200&apiKey={api_key}"
        )
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            snippet = (resp.text or "").strip().replace("\n", " ")[:180]
            print(f"⚠️ Polygon aggs fallback failed for {symbol}: HTTP {resp.status_code} — {snippet}")
            return []
        data = resp.json() or {}
        results = data.get("results") or []
        rows: List[dict] = []
        for r in results:
            try:
                # Polygon 't' is ms since epoch UTC
                ts_ms = int(r.get("t"))
                dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                rows.append(
                    {
                        "date": dt_utc.strftime("%Y-%m-%d"),
                        "date_iso": dt_utc.isoformat(),
                        "open": _safe_float(r.get("o")),
                        "high": _safe_float(r.get("h")),
                        "low": _safe_float(r.get("l")),
                        "close": _safe_float(r.get("c")),
                        "volume": _safe_int(r.get("v")),
                    }
                )
            except Exception:
                continue
        rows = [r for r in rows if r.get("close") is not None]
        rows = rows[-30:] if len(rows) > 30 else rows
        return rows
    except Exception as e:
        print(f"⚠️ Polygon aggs exception for {symbol}: {e}")
        return []


def _fetch_yahoo_quote(symbol: str) -> QuoteResponse:
    # Prefer Stooq; fallback to Polygon daily aggs
    now = time.time()
    cached = QUOTE_CACHE.get(symbol)
    if cached and (now - cached[0]) < QUOTE_CACHE_TTL_SECONDS:
        return cached[1]
    hist = _fetch_daily_history_prefer_stooq(symbol)
    if hist:
        last = hist[-1]
        prev = hist[-2] if len(hist) > 1 else None
        price = _safe_float(last.get("close"))
        change = None
        change_pct = None
        if price is not None and prev is not None:
            prev_close = _safe_float(prev.get("close"))
            if prev_close not in (None, 0):
                change = price - prev_close
                change_pct = (change / prev_close) * 100.0
        result = QuoteResponse(
            symbol=symbol,
            price=price,
            change=change,
            change_percent=change_pct,
            currency=None,
            market_time=(last.get("date_iso") or None),
            market_state="CLOSED",
            name=None,
        )
        QUOTE_CACHE[symbol] = (now, result)
        return result
    empty = QuoteResponse(symbol=symbol)
    QUOTE_CACHE[symbol] = (now, empty)
    return empty


CHART_CACHE_TTL_SECONDS = int(os.getenv("CHART_CACHE_TTL_SECONDS", "300"))
CHART_CACHE: dict[tuple[str, str, str], tuple[float, ChartResponse]] = {}


def _fetch_yahoo_chart(symbol: str, range_: str, interval: str) -> ChartResponse:
    cache_key = (symbol, range_, interval)
    now = time.time()
    cached = CHART_CACHE.get(cache_key)
    if cached and (now - cached[0]) < CHART_CACHE_TTL_SECONDS:
        return cached[1]
    hist = _fetch_daily_history_prefer_stooq(symbol)
    if hist:
        ts = [int(datetime.strptime(r["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) for r in hist]
        response = ChartResponse(
            symbol=symbol,
            range=range_,
            interval=interval,
            timestamps=ts,
            opens=[_safe_float(r.get("open")) for r in hist],
            highs=[_safe_float(r.get("high")) for r in hist],
            lows=[_safe_float(r.get("low")) for r in hist],
            closes=[_safe_float(r.get("close")) for r in hist],
            volumes=[_safe_int(r.get("volume")) for r in hist],
        )
        CHART_CACHE[cache_key] = (now, response)
        return response
    empty = ChartResponse(
        symbol=symbol,
        range=range_,
        interval=interval,
        timestamps=[],
        opens=[],
        highs=[],
        lows=[],
        closes=[],
        volumes=[],
    )
    CHART_CACHE[cache_key] = (now, empty)
    return empty


def _safe_float(v):
    try:
        if v is None:
            return None
        f = float(v)
        if f != f:  # NaN check
            return None
        return f
    except Exception:
        return None


def _safe_int(v):
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _stooq_candidates(symbol: str) -> List[str]:
    s = (symbol or "").strip().lower()
    cands = []
    if not s:
        return cands
    if "." in s:
        cands.append(s)
    else:
        cands.append(f"{s}.us")
        cands.append(s)
    return cands


def _fetch_stooq_history(symbol: str) -> List[dict]:
    # Returns recent daily history as list of dicts asc by date
    candidates = _stooq_candidates(symbol)
    for s in candidates:
        url = f"https://stooq.com/q/d/l/?s={s}&i=d"
        try:
            resp = requests.get(url, timeout=8)
            if resp.status_code != 200 or not resp.text or resp.text.lower().startswith("not found"):
                continue
            # CSV header: Date,Open,High,Low,Close,Volume
            f = StringIO(resp.text.strip())
            reader = csv.DictReader(f)
            rows: List[dict] = []
            for row in reader:
                try:
                    d = row.get("Date") or row.get("date")
                    if not d:
                        continue
                    parsed = {
                        "date": d,
                        "date_iso": datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).isoformat(),
                        "open": _safe_float(row.get("Open") or row.get("open")),
                        "high": _safe_float(row.get("High") or row.get("high")),
                        "low": _safe_float(row.get("Low") or row.get("low")),
                        "close": _safe_float(row.get("Close") or row.get("close")),
                        "volume": _safe_int(row.get("Volume") or row.get("volume")),
                    }
                    rows.append(parsed)
                except Exception:
                    continue
            rows = [r for r in rows if r.get("close") is not None]
            rows.sort(key=lambda r: r["date"])  # ascending
            trimmed = rows[-30:] if len(rows) > 30 else rows
            if trimmed:
                return trimmed
        except Exception as e:
            print(f"Stooq fetch failed for {symbol} ({s}): {e}")
            continue
    return []


def _stooq_day_change_percent(symbol: str) -> Optional[float]:
    """Return 1-day percent change from Stooq using the last two closes.

    Returns None if unavailable.
    """
    try:
        history = _fetch_stooq_history(symbol)
        if not history or len(history) < 2:
            return None
        last_close = _safe_float(history[-1].get("close"))
        prev_close = _safe_float(history[-2].get("close"))
        if last_close is None or prev_close is None or prev_close <= 0:
            return None
        return ((last_close - prev_close) / prev_close) * 100.0
    except Exception:
        return None


def _filter_ranked_losers_by_stooq(
    ranked: List["InterestingLoser"],
    tolerance_pp: float,
    min_abs_for_sign_check: float = 5.0,
) -> List["InterestingLoser"]:
    """Final sanity filter comparing Polygon EOD change to Stooq day-over-day.

    - Drop if absolute difference in percentage points exceeds tolerance_pp.
    - Drop if signs differ and at least one magnitude is >= min_abs_for_sign_check.
    """
    if not ranked:
        return ranked
    filtered: List[InterestingLoser] = []
    for item in ranked:
        try:
            stooq_pct = _stooq_day_change_percent(item.symbol)
            poly_pct = item.change_percent
            if stooq_pct is None or poly_pct is None:
                filtered.append(item)
                continue
            diff_pp = abs(poly_pct - stooq_pct)
            signs_differ = (poly_pct < 0) != (stooq_pct < 0)
            if signs_differ and max(abs(poly_pct), abs(stooq_pct)) >= min_abs_for_sign_check:
                # Likely mismatch due to corporate action or bad data
                continue
            if diff_pp > tolerance_pp:
                continue
            filtered.append(item)
        except Exception:
            filtered.append(item)
    return filtered

## Removed legacy popular symbols list


## Removed legacy losers-from-news search flow


## Removed legacy losers extraction from news


## Removed legacy biggest losers sync fetcher


def _refresh_biggest_losers_cache():
    # Retained for compatibility; now a thin wrapper around interesting cache warm.
    try:
        _refresh_interesting_losers_cache()
    except Exception as e:
        print(f"❌ Failed to run interesting losers warm: {e}")


def _background_refresh_loop():
    # Legacy loop retained but now delegates to interesting cache refresh
    while True:
        try:
            time.sleep(INTERESTING_LOSERS_CACHE_TTL_SECONDS)
            _refresh_interesting_losers_cache()
        except Exception as e:
            print(f"❌ Background refresh error: {e}")
            time.sleep(300)


def _ensure_cache_initialized():
    """Initialize curated losers cache once in the background."""
    global _background_refresh_task, _app_started
    if _app_started:
        return
    threading.Thread(target=_refresh_interesting_losers_cache, daemon=True).start()
    if _background_refresh_task is None:
        _background_refresh_task = threading.Thread(target=_background_refresh_loop, daemon=True)
        _background_refresh_task.start()
    _app_started = True


def _get_cached_biggest_losers() -> List[LoserStock]:
    """Deprecated: always return empty; use /interesting-losers for data."""
    _ensure_cache_initialized()
    return []


def _get_price_context(symbol: str) -> Optional[str]:
    quote = _fetch_yahoo_quote(symbol)
    chart = _fetch_yahoo_chart(symbol, range_="1mo", interval="1d")
    lines: List[str] = []
    if quote.price is not None:
        chg_pct = f"{quote.change_percent:.2f}%" if quote.change_percent is not None else "n/a"
        chg_abs = f"{quote.change:+.2f}" if quote.change is not None else "n/a"
        cur = quote.currency or "USD"
        lines.append(f"Current: {quote.price:.2f} {cur} ({chg_abs}, {chg_pct})")
    closes = [c for c in (chart.closes or []) if isinstance(c, (int, float))]
    if len(closes) >= 2:
        last = closes[-1]
        prev = closes[-2]
        if prev:
            d1 = ((last - prev) / prev) * 100.0
            lines.append(f"1d change: {d1:+.2f}%")
        first = closes[0]
        if first:
            d30 = ((last - first) / first) * 100.0
            lines.append(f"1mo change: {d30:+.2f}%")
    return "\n".join(lines) if lines else None


def _sanitize_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"
    s = "".join(ch for ch in s if ch in allowed)
    return s


def _is_non_primary_equity_symbol(symbol: str) -> bool:
    """Return True for NASDAQ-style fifth-letter suffixes that are not common
    primary equity shares, such as Rights (R), Warrants (W), and Units (U),
    and when-issued (V). This helps avoid confusing instruments like FERAR
    (rights) with primary tickers such as RACE (Ferrari).
    """
    s = (symbol or "").strip().upper()
    if len(s) == 5 and s[-1] in {"R", "W", "U", "V"}:
        return True
    return False


OG_IMAGE_CACHE_TTL_SECONDS = int(os.getenv("OG_IMAGE_CACHE_TTL_SECONDS", "1800"))
OG_IMAGE_CACHE: dict[str, tuple[float, bytes]] = {}

# Background refresh state
_background_refresh_task = None
_app_started = False

# Cache for interesting losers (LLM-ranked)
INTERESTING_LOSERS_CACHE_TTL_SECONDS = int(os.getenv("INTERESTING_LOSERS_CACHE_TTL_SECONDS", "86400"))
INTERESTING_LOSERS_CACHE: Optional[tuple[float, List[InterestingLoser]]] = None
INTERESTING_LOSERS_LOCK = threading.RLock()

# Optional on-disk cache for news lookups to reduce DDG usage across restarts
NEWS_CACHE_DIR = os.getenv("NEWS_CACHE_DIR") or os.path.join(os.path.dirname(__file__), ".cache", "news")
NEWS_CACHE_TTL_SECONDS = int(os.getenv("NEWS_CACHE_TTL_SECONDS", "604800"))  # 7 days

# In-memory EOD series cache built from grouped results (prev + target dates)
EOD_SERIES_CACHE: dict[str, list[dict]] = {}


def _fetch_biggest_losers_polygon() -> List[LoserStock]:
    """Fetch biggest losers using Polygon.io Full Market Snapshot.
    
    Requires env var POLYGON_API_KEY. Returns up to 50 losers sorted by most negative %.
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("❌ Polygon API key not set. Please set POLYGON_API_KEY in environment or .env")
        return []
    url = (
        f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers?apiKey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            body = None
            try:
                body = resp.text
            except Exception:
                body = None
            snippet = (body or "").strip().replace("\n", " ")[:300]
            print(f"❌ Polygon snapshot failed: HTTP {resp.status_code} — {snippet}")
            return []
        payload = resp.json() or {}
        tickers = payload.get("tickers") or []
        losers: List[LoserStock] = []
        for t in tickers:
            try:
                symbol = (t.get("ticker") or "").strip().upper()
                if not symbol:
                    continue
                last_trade = t.get("lastTrade") or {}
                price = _safe_float(last_trade.get("p"))
                if price is None:
                    day_obj = t.get("day") or {}
                    price = _safe_float(day_obj.get("c"))
                change = _safe_float(t.get("todaysChange"))
                change_pct = _safe_float(t.get("todaysChangePerc"))
                day_obj = t.get("day") or {}
                volume = _safe_int(day_obj.get("v"))
                losers.append(
                    LoserStock(
                        symbol=symbol,
                        name=None,
                        price=price,
                        change=change,
                        change_percent=change_pct,
                        volume=volume,
                    )
                )
            except Exception:
                continue
        losers = [l for l in losers if isinstance(l.change_percent, (int, float)) and l.change_percent < 0]
        losers.sort(key=lambda l: l.change_percent)
        return losers[:50]
    except Exception as e:
        print(f"❌ Polygon snapshot exception: {e}")
        return []


def _is_business_day(d: date) -> bool:
    return d.weekday() < 5


def _prev_business_day(d: date) -> date:
    cur = d - timedelta(days=1)
    while not _is_business_day(cur):
        cur -= timedelta(days=1)
    return cur


def _determine_eod_target_date(now_utc: Optional[datetime] = None) -> date:
    """Return strictly the previous business day in America/New_York.

    This avoids 403s on free plans that cannot access today's grouped data
    until after end of day processing is complete.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    ny = now_utc.astimezone(ZoneInfo("America/New_York"))
    return _prev_business_day(ny.date())


def _fetch_polygon_grouped(date_str: str) -> list:
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("❌ Polygon API key not set. Please set POLYGON_API_KEY in environment or .env")
        return []
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date_str}?adjusted=true&apiKey={api_key}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            snippet = (resp.text or "").strip().replace("\n", " ")[:300]
            print(f"❌ Polygon grouped failed for {date_str}: HTTP {resp.status_code} — {snippet}")
            return []
        payload = resp.json() or {}
        return payload.get("results") or []
    except Exception as e:
        print(f"❌ Polygon grouped exception for {date_str}: {e}")
        return []


def _fetch_biggest_losers_polygon_eod() -> List[LoserStock]:
    """Compute biggest losers from Polygon grouped EOD for the latest trading day."""
    target = _determine_eod_target_date()
    prev = _prev_business_day(target)
    target_str = target.isoformat()
    prev_str = prev.isoformat()
    print(f"ℹ️ Computing EOD losers for {target_str} (prev {prev_str})")
    today_group = _fetch_polygon_grouped(target_str)
    prev_group = _fetch_polygon_grouped(prev_str)
    if not today_group or not prev_group:
        print("⚠️ Missing grouped data for one or both days; returning empty losers list")
        return []
    prev_close_by_ticker: dict[str, float] = {}
    for r in prev_group:
        try:
            tkr = (r.get("T") or "").strip().upper()
            c_prev = _safe_float(r.get("c"))
            if tkr and c_prev is not None and c_prev > 0:
                prev_close_by_ticker[tkr] = c_prev
        except Exception:
            continue
    losers: List[LoserStock] = []
    for r in today_group:
        try:
            tkr = (r.get("T") or "").strip().upper()
            if not tkr or not tkr.isalpha() or len(tkr) > 5:
                continue
            # Filter out non-primary symbols like rights/warrants/units to
            # avoid confusing tickers (e.g., FERAR) with primary equities.
            if _is_non_primary_equity_symbol(tkr):
                continue
            c_today = _safe_float(r.get("c"))
            v_today = _safe_int(r.get("v"))
            c_prev = prev_close_by_ticker.get(tkr)
            if c_today is None or c_prev is None or c_prev <= 0:
                continue
            change = c_today - c_prev
            change_pct = (change / c_prev) * 100.0
            if change_pct < 0:
                losers.append(
                    LoserStock(
                        symbol=tkr,
                        name=None,
                        price=c_today,
                        change=change,
                        change_percent=change_pct,
                        volume=v_today,
                    )
                )
        except Exception:
            continue
    # Keep broad universe here; filtering may be applied by callers when needed
    losers.sort(key=lambda l: l.change_percent or 0)
    return losers[:50]


## Removed S&P/NASDAQ universe filtering to avoid redundancy per product direction

def _generate_og_image_png(symbol: str) -> bytes:
    # Match the actual site's look - dark background, detailed commentary
    width, height = 1200, 630
    
    quote = _fetch_yahoo_quote(symbol)
    chart = _fetch_yahoo_chart(symbol, range_="1mo", interval="1d")
    closes = [c for c in (chart.closes or []) if isinstance(c, (int, float))]

    # Site-matching colors
    bg = (15, 18, 24)  # Dark blue-gray like the site
    white = (255, 255, 255)
    gray = (156, 163, 175)
    red = (248, 113, 113)  # Softer red like site
    green = (34, 197, 94)
    
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    # Font setup
    def _load_font(size: int):
        """Load a scalable TrueType font so size changes actually apply.

        Falls back through common system font paths and an OG_FONT_PATH env var.
        If none are available, uses PIL's default bitmap font (fixed size).
        """
        # Allow override via env var
        env_font = os.getenv("OG_FONT_PATH")
        candidates = [env_font] if env_font else []
        # Common locations
        candidates.extend([
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            # Common on Ubuntu minimal images
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        ])
        for path in candidates:
            try:
                if path and os.path.exists(path) or path and not os.path.isabs(path):
                    return ImageFont.truetype(path, size)
            except Exception:
                continue
        # Try PIL's bundled DejaVu
        try:
            import PIL  # type: ignore
            pil_dir = os.path.dirname(PIL.__file__)
            bundled = os.path.join(pil_dir, "fonts", "DejaVuSans.ttf")
            return ImageFont.truetype(bundled, size)
        except Exception:
            pass
        # Last resort (fixed size)
        return ImageFont.load_default()

    # Slightly larger fonts so preview text is easier to read
    big_font = _load_font(56)
    med_font = _load_font(36)
    small_font = _load_font(28)
    tiny_font = _load_font(20)

    # ASCII-safe text
    def _clean_text(text: str) -> str:
        text = text.replace("\u2014", "-").replace("\u2013", "-")
        return text.encode("ascii", "ignore").decode("ascii")

    # Layout: symbol + price on top row, chart on right, commentary filling left
    # Top row: Symbol and price
    padding = 40
    y = 40
    
    # Symbol
    draw.text((padding, y), _clean_text(symbol), font=big_font, fill=white)
    
    # Price on same line, right side
    if quote.price is not None:
        chg = quote.change or 0.0
        chg_pct = quote.change_percent or 0.0
        price_color = red if chg < 0 else green
        price_text = f"${quote.price:.2f}  {chg:+.2f} ({chg_pct:+.2f}%)"
        # Measure text to position it right-aligned
        bbox = draw.textbbox((0, 0), _clean_text(price_text), font=med_font)
        price_width = bbox[2] - bbox[0]
        draw.text((width - padding - price_width, y + 5), _clean_text(price_text), font=med_font, fill=price_color)
    
    # Chart area - compact, top right
    # Move chart down a touch to make room for the larger title/price row
    chart_y = y + 80
    chart_height = 180
    chart_width = 400
    chart_left = width - padding - chart_width
    chart_right = width - padding
    chart_top = chart_y
    chart_bottom = chart_y + chart_height
    
    if len(closes) >= 2:
        # Sparkline similar to site
        min_close = min(closes)
        max_close = max(closes)
        price_range = max_close - min_close if max_close != min_close else 1
        
        points = []
        for i, price in enumerate(closes):
            x = chart_left + (i / (len(closes) - 1)) * chart_width
            y_norm = (price - min_close) / price_range
            y = chart_bottom - (y_norm * chart_height)
            points.append((x, y))
        
        # Chart line - match site style
        line_color = red if closes[-1] < closes[0] else green
        draw.line(points, fill=line_color, width=3)
        
        # High/Low labels like site
        draw.text((chart_right - 100, chart_top - 25), f"High: ${max_close:.2f}", font=tiny_font, fill=gray)
        draw.text((chart_right - 100, chart_bottom + 5), f"Low: ${min_close:.2f}", font=tiny_font, fill=gray)
    else:
        print(f"⚠️ OG chart missing series for {symbol}; history points: {len(closes)}")
    
    # Commentary - get actual analysis like the site shows
    def _get_full_commentary() -> str:
        try:
            # Try to get actual analysis from the backend
            sources = _search_news_for_symbol(symbol, days=7, max_results=8)
            price_ctx = _get_price_context(symbol)
            # Keep OG image preview succinct
            prompt = _build_llm_prompt(symbol, sources, "humorous", price_ctx, max_words=80)
            analysis = _call_openai(prompt)
            if analysis and len(analysis.strip()) > 20:
                return analysis.strip()
        except Exception:
            pass
        return f"The market's been rough on {symbol} lately. Between earnings volatility, analyst downgrades, and general market jitters, it's showing the classic signs of a stock in consolidation mode. Recent price action suggests investors are taking profits and reassessing valuations amid broader market uncertainty."
    
    commentary = _clean_text(_get_full_commentary())
    # Safety net: trim to max words to avoid awkward mid-word cuts
    def _trim_words(text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + "…"

    commentary = _trim_words(commentary, 85)
    
    # Commentary area - left side, flowing around chart
    text_left = padding
    text_right = chart_left - 20  # Leave gap before chart
    text_top = chart_y
    text_width = text_right - text_left
    
    # Wrap text carefully
    wrapped_lines = []
    words = commentary.split()
    current_line = ""
    
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        bbox = draw.textbbox((0, 0), test_line, font=small_font)
        line_width = bbox[2] - bbox[0]
        
        if line_width <= text_width:
            current_line = test_line
        else:
            if current_line:
                wrapped_lines.append(current_line)
            current_line = word
    
    if current_line:
        wrapped_lines.append(current_line)
    
    # Draw commentary lines with dynamic height based on font metrics
    bbox_sample = draw.textbbox((0, 0), "Ag", font=small_font)
    line_height = (bbox_sample[3] - bbox_sample[1]) + 8
    y = text_top
    for line in wrapped_lines:
        if y > height - 60:  # Don't go too close to bottom
            break
        draw.text((text_left, y), line, font=small_font, fill=white)
        y += line_height
    
    # Footer
    draw.text((padding, height - 30), "whyisthestockplummeting.com", font=tiny_font, fill=gray)
    
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _build_share_description(symbol: str) -> str:
    parts: List[str] = []
    price_ctx = _get_price_context(symbol)
    if price_ctx:
        parts.append(price_ctx)
    # Add up to two recent headlines for flavor
    try:
        news = _search_news_for_symbol(symbol, days=7, max_results=3)
        if news:
            titles = [n.title for n in news[:2] if n.title]
            if titles:
                parts.append("; ".join(titles))
    except Exception:
        pass
    if not parts:
        parts.append("Tap to see a quick chart and summary.")
    # Keep description short for social previews
    desc = " — ".join(parts)
    return desc[:280]


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    symbols = _ensure_list_symbols(request.symbols)
    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols provided")

    results: List[SymbolAnalysis] = []
    for symbol in symbols:
        sources = _search_news_for_symbol(symbol, request.days, request.max_results)
        price_ctx = _get_price_context(symbol)
        prompt = _build_llm_prompt(symbol, sources, request.tone, price_ctx)
        summary = _call_openai(prompt)
        results.append(
            SymbolAnalysis(symbol=symbol, summary=summary, sources=sources)
        )

    return AnalyzeResponse(results=results)


@app.get("/quote/{symbol}", response_model=QuoteResponse)
async def get_quote(symbol: str) -> QuoteResponse:
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    return _fetch_yahoo_quote(symbol)


@app.get("/chart/{symbol}", response_model=ChartResponse)
async def get_chart(
    symbol: str,
    range: str = Query("1mo", pattern=r"^(1d|5d|1mo|3mo|6mo|1y|2y|5y|max)$"),
    interval: str = Query("1d", pattern=r"^(1m|2m|5m|15m|30m|60m|90m|1h|1d|1wk|1mo|3mo)$"),
) -> ChartResponse:
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    return _fetch_yahoo_chart(symbol, range, interval)


## Removed legacy /biggest-losers endpoint in favor of /interesting-losers


@app.get("/interesting-losers", response_model=InterestingLosersResponse)
async def get_interesting_losers(
    candidates: int = Query(200, ge=20, le=1000),
    top: int = Query(12, ge=5, le=25),
) -> InterestingLosersResponse:
    """Return curated losers from cache only; never compute in request path.

    If cache is empty/not warmed yet, returns an empty list immediately.
    """
    with INTERESTING_LOSERS_LOCK:
        cached = INTERESTING_LOSERS_CACHE
    if cached and (time.time() - cached[0]) < INTERESTING_LOSERS_CACHE_TTL_SECONDS:
        ranked = cached[1][:top]
        ts = cached[0]
    else:
        ranked = []
        ts = time.time()
    return InterestingLosersResponse(
        losers=ranked,
        last_updated=datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
        session="EOD",
    )



@app.get("/og-image/{symbol}.png")
async def og_image(symbol: str) -> Response:
    symbol = _sanitize_symbol(symbol)
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    now = time.time()
    cached = OG_IMAGE_CACHE.get(symbol)
    if cached and (now - cached[0]) < OG_IMAGE_CACHE_TTL_SECONDS:
        image_bytes = cached[1]
    else:
        image_bytes = _generate_og_image_png(symbol)
        OG_IMAGE_CACHE[symbol] = (now, image_bytes)
    etag = md5(image_bytes).hexdigest()
    headers = {
        "Cache-Control": "public, max-age=1800",
        "ETag": etag,
    }
    return Response(content=image_bytes, media_type="image/png", headers=headers)


@app.post("/og-image/warm/{symbol}")
async def og_image_warm(symbol: str) -> Response:
    """Pre-generate and cache the OG image for a symbol without returning the bytes.

    Returns 201 if a new image was generated, 204 if a fresh cached image already exists.
    """
    symbol = _sanitize_symbol(symbol)
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    now = time.time()
    cached = OG_IMAGE_CACHE.get(symbol)
    if cached and (now - cached[0]) < OG_IMAGE_CACHE_TTL_SECONDS:
        return Response(status_code=204)
    image_bytes = _generate_og_image_png(symbol)
    OG_IMAGE_CACHE[symbol] = (time.time(), image_bytes)
    return Response(status_code=201)


@app.get("/s/{symbol}")
async def share(symbol: str, request: Request) -> Response:
    symbol = _sanitize_symbol(symbol)
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    origin = os.getenv("PUBLIC_WEB_ORIGIN")
    try:
        image_url = str(request.url_for("og_image", symbol=symbol))  # type: ignore[arg-type]
    except Exception:
        image_url = f"/og-image/{symbol}.png"
    if origin and image_url.startswith("/"):
        image_url = origin.rstrip("/") + image_url
    title = f"{symbol} — Why Is The Stock Plummeting?"
    description = _build_share_description(symbol)
    canonical = f"{(origin or '').rstrip('/')}/?stock={symbol}" if origin else f"/?stock={symbol}"
    html = f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{title}</title>
  <link rel=\"canonical\" href=\"{canonical}\" />
  <meta name=\"description\" content=\"{description}\" />
  <meta property=\"og:type\" content=\"website\" />
  <meta property=\"og:site_name\" content=\"Why Is The Stock Plummeting?\" />
  <meta property=\"og:title\" content=\"{title}\" />
  <meta property=\"og:description\" content=\"{description}\" />
  <meta property=\"og:image\" content=\"{image_url}\" />
  <meta name=\"twitter:card\" content=\"summary_large_image\" />
  <meta name=\"twitter:title\" content=\"{title}\" />
  <meta name=\"twitter:description\" content=\"{description}\" />
  <meta name=\"twitter:image\" content=\"{image_url}\" />
  <meta http-equiv=\"refresh\" content=\"0;url={canonical}\" />
  <style>body{{background:#0f1218;color:#eef;font-family:system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif}}.wrap{{max-width:720px;margin:80px auto;padding:0 16px}}</style>
  </head>
<body>
  <div class=\"wrap\">
    <h1>{title}</h1>
    <p>{description}</p>
    <img src=\"{image_url}\" alt=\"{symbol} chart\" width=\"600\" />
    <p>Redirecting to the app… If it doesn't, <a href=\"{canonical}\">click here</a>.</p>
  </div>
</body>
</html>
"""
    return Response(content=html, media_type="text/html; charset=utf-8")


@app.get("/")
async def root():
    return {"ok": True, "service": "whyisthestockplummeting"}