from datetime import datetime, timezone, date
import os
from typing import List, Optional, Union

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
    # Query DuckDuckGo News; no Google fallback
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


QUOTE_CACHE_TTL_SECONDS = int(os.getenv("QUOTE_CACHE_TTL_SECONDS", "60"))
QUOTE_CACHE: dict[str, tuple[float, QuoteResponse]] = {}


def _fetch_yahoo_quote(symbol: str) -> QuoteResponse:
    # Simplified: Use Stooq as the sole data source
    now = time.time()
    cached = QUOTE_CACHE.get(symbol)
    if cached and (now - cached[0]) < QUOTE_CACHE_TTL_SECONDS:
        return cached[1]
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


def _sanitize_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"
    s = "".join(ch for ch in s if ch in allowed)
    return s


OG_IMAGE_CACHE_TTL_SECONDS = int(os.getenv("OG_IMAGE_CACHE_TTL_SECONDS", "1800"))
OG_IMAGE_CACHE: dict[str, tuple[float, bytes]] = {}


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