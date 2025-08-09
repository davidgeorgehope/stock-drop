from datetime import datetime, timezone, date
import os
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI
try:
    from ddgs import DDGS  # type: ignore
except Exception:  # pragma: no cover - optional dependency warning safe-guard
    DDGS = None  # type: ignore
from dotenv import load_dotenv
import requests
import time
import csv
from io import StringIO

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
    # Keep the query broad so we don't accidentally filter out relevant articles
    query = f"{symbol} stock"
    timelimit = f"d{max(1, min(days, 30))}"
    items: List[SourceItem] = []
    # Simple retry loop to handle transient rate limits
    attempts = 0
    while attempts < 3 and len(items) == 0:
        attempts += 1
        try:
            if DDGS is not None:
                with DDGS() as ddgs:
                    # ddgs.news expects the query as the first positional argument (or named as 'query')
                    for n in ddgs.news(query, region="us-en", safesearch="moderate", timelimit=timelimit, max_results=max_results):
                        items.append(
                            SourceItem(
                                title=n.get("title") or "",
                                url=n.get("url") or n.get("link") or "",
                                snippet=n.get("excerpt") or n.get("body"),
                                published=n.get("date"),
                            )
                        )
        except Exception as e:
            print(f"DDG news search attempt {attempts} failed for {symbol}: {e}")
            # brief backoff
            import time
            time.sleep(0.8 * attempts)
    cleaned = [i for i in items if i.url]
    if cleaned:
        return cleaned

    # Fallback: Google News RSS
    try:
        print(f"DDG returned no items for {symbol}. Falling back to Google News RSS…")
        from urllib.parse import quote_plus
        import requests
        import re
        import html as _html
        import xml.etree.ElementTree as ET

        # Add recent time window using when:{days}d
        rss_query = quote_plus(f"{symbol} stock when:{max(1, min(days, 30))}d")
        url = f"https://news.google.com/rss/search?q={rss_query}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.text)
        channel = root.find("channel")
        if channel is None:
            return []
        results: List[SourceItem] = []
        for item in channel.findall("item")[:max_results]:
            title_el = item.find("title")
            link_el = item.find("link")
            pub_el = item.find("pubDate")
            desc_el = item.find("description")
            title = title_el.text if title_el is not None else ""
            link = link_el.text if link_el is not None else ""
            published = pub_el.text if pub_el is not None else None
            snippet_html = desc_el.text if desc_el is not None else None
            snippet = None
            if snippet_html:
                # strip HTML tags and unescape entities
                snippet = re.sub(r"<[^>]+>", "", _html.unescape(snippet_html))
            if link:
                results.append(SourceItem(title=title or link, url=link, snippet=snippet, published=published))
        return results
    except Exception as e:
        print(f"Google News RSS fallback failed for {symbol}: {e}")
        return []


def _build_llm_prompt(symbol: str, sources: List[SourceItem], tone: str, price_context: Optional[str] = None) -> str:
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


def _yahoo_headers() -> dict:
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Connection": "close",
    }


QUOTE_CACHE_TTL_SECONDS = int(os.getenv("QUOTE_CACHE_TTL_SECONDS", "60"))
QUOTE_CACHE: dict[str, tuple[float, QuoteResponse]] = {}


def _fetch_yahoo_quote(symbol: str) -> QuoteResponse:
    # Cache check
    now = time.time()
    cached = QUOTE_CACHE.get(symbol)
    if cached and (now - cached[0]) < QUOTE_CACHE_TTL_SECONDS:
        return cached[1]
    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
    try:
        resp = requests.get(url, headers=_yahoo_headers(), timeout=8)
        if resp.status_code != 200:
            # Try Stooq fallback on common upstream failures
            stooq = _fetch_stooq_history(symbol)
            if stooq:
                last = stooq[-1]
                prev = stooq[-2] if len(stooq) > 1 else None
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
            raise HTTPException(status_code=502, detail=f"Quote upstream error {resp.status_code}")
        data = resp.json()
        results = (data or {}).get("quoteResponse", {}).get("result", [])
        if not results:
            # Fallback to Stooq last close
            stooq = _fetch_stooq_history(symbol)
            if stooq:
                last = stooq[-1]
                prev = stooq[-2] if len(stooq) > 1 else None
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
        q = results[0]
        ts = q.get("regularMarketTime")
        market_time = None
        if ts:
            try:
                market_time = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
            except Exception:
                market_time = None
        result = QuoteResponse(
            symbol=symbol,
            price=q.get("regularMarketPrice"),
            change=q.get("regularMarketChange"),
            change_percent=q.get("regularMarketChangePercent"),
            currency=q.get("currency"),
            market_time=market_time,
            market_state=q.get("marketState"),
            name=q.get("shortName") or q.get("longName"),
        )
        QUOTE_CACHE[symbol] = (now, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Quote fetch failed for {symbol}: {e}")
        stooq = _fetch_stooq_history(symbol)
        if stooq:
            last = stooq[-1]
            prev = stooq[-2] if len(stooq) > 1 else None
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
    # Yahoo uses "range" param values like: 1d,5d,1mo,3mo,6mo,1y,2y,5y,max
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_}&interval={interval}&includePrePost=false"
    )
    try:
        resp = requests.get(url, headers=_yahoo_headers(), timeout=10)
        if resp.status_code != 200:
            # Fallback to Stooq for non-200
            stooq = _fetch_stooq_history(symbol)
            if stooq:
                ts = [int(datetime.strptime(r["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) for r in stooq]
                response = ChartResponse(
                    symbol=symbol,
                    range=range_,
                    interval=interval,
                    timestamps=ts,
                    opens=[_safe_float(r.get("open")) for r in stooq],
                    highs=[_safe_float(r.get("high")) for r in stooq],
                    lows=[_safe_float(r.get("low")) for r in stooq],
                    closes=[_safe_float(r.get("close")) for r in stooq],
                    volumes=[_safe_int(r.get("volume")) for r in stooq],
                )
                CHART_CACHE[cache_key] = (now, response)
                return response
            raise HTTPException(status_code=502, detail=f"Chart upstream error {resp.status_code}")
        data = resp.json()
        result = ((data or {}).get("chart", {}) or {}).get("result")
        if not result:
            # Fallback to Stooq history (daily only)
            stooq = _fetch_stooq_history(symbol)
            if stooq:
                ts = [int(datetime.strptime(r["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) for r in stooq]
                response = ChartResponse(
                    symbol=symbol,
                    range=range_,
                    interval=interval,
                    timestamps=ts,
                    opens=[_safe_float(r.get("open")) for r in stooq],
                    highs=[_safe_float(r.get("high")) for r in stooq],
                    lows=[_safe_float(r.get("low")) for r in stooq],
                    closes=[_safe_float(r.get("close")) for r in stooq],
                    volumes=[_safe_int(r.get("volume")) for r in stooq],
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
        r0 = result[0]
        timestamps = r0.get("timestamp", []) or []
        indicators = (r0.get("indicators", {}) or {}).get("quote", [])
        q0 = indicators[0] if indicators else {}
        response = ChartResponse(
            symbol=symbol,
            range=range_,
            interval=interval,
            timestamps=timestamps,
            opens=q0.get("open", []) or [],
            highs=q0.get("high", []) or [],
            lows=q0.get("low", []) or [],
            closes=q0.get("close", []) or [],
            volumes=q0.get("volume", []) or [],
        )
        CHART_CACHE[cache_key] = (now, response)
        return response
    except HTTPException:
        raise
    except Exception as e:
        print(f"Chart fetch failed for {symbol}: {e}")
        stooq = _fetch_stooq_history(symbol)
        if stooq:
            ts = [int(datetime.strptime(r["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) for r in stooq]
            response = ChartResponse(
                symbol=symbol,
                range=range_,
                interval=interval,
                timestamps=ts,
                opens=[_safe_float(r.get("open")) for r in stooq],
                highs=[_safe_float(r.get("high")) for r in stooq],
                lows=[_safe_float(r.get("low")) for r in stooq],
                closes=[_safe_float(r.get("close")) for r in stooq],
                volumes=[_safe_int(r.get("volume")) for r in stooq],
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
    range: str = Query("1mo", regex=r"^(1d|5d|1mo|3mo|6mo|1y|2y|5y|max)$"),
    interval: str = Query("1d", regex=r"^(1m|2m|5m|15m|30m|60m|90m|1h|1d|1wk|1mo|3mo)$"),
) -> ChartResponse:
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    return _fetch_yahoo_chart(symbol, range, interval)


@app.get("/")
async def root():
    return {"ok": True, "service": "whyisthestockplummeting"}