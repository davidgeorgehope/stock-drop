"""Market data fetching services for Stooq, Polygon, and derived quote/chart functions."""

import os
import time
import csv
import requests
import threading
from datetime import datetime, timezone, date, timedelta
from typing import List, Optional, Dict, Tuple
from io import StringIO
from zoneinfo import ZoneInfo

from database.repositories.price_repo import PriceRepository
from pydantic import BaseModel

# Thread lock for Polygon API calls to prevent race conditions
_polygon_fetch_lock = threading.Lock()


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


class PolygonRateLimiter:
    """Simple rate limiter for Polygon API to avoid 429 errors.
    
    Polygon's free tier typically allows 5 requests per minute.
    """
    def __init__(self, max_requests_per_minute=5):
        self.max_requests_per_minute = max_requests_per_minute
        self.min_interval = 60.0 / max_requests_per_minute  # seconds between requests
        self.last_request_time = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_request_time
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                print(f"Rate limiting: waiting {sleep_time:.1f}s before next Polygon request")
                time.sleep(sleep_time)
            self.last_request_time = time.time()


# Initialize Polygon rate limiter (5 requests per minute for free tier)
polygon_rate_limiter = PolygonRateLimiter(max_requests_per_minute=5)

# Keep a small in-memory cache for recent lookups (1 hour TTL) to avoid hitting DB too often
POLYGON_GROUPED_MEMORY_CACHE: Dict[str, Tuple[float, List[dict]]] = {}
POLYGON_GROUPED_MEMORY_CACHE_TTL = 3600  # 1 hour for memory cache

# Disable in-memory caches for quotes and charts; rely on SQLite + direct fetch
QUOTE_CACHE_TTL_SECONDS = 0
QUOTE_CACHE: Dict[str, Tuple[float, QuoteResponse]] = {}

CHART_CACHE_TTL_SECONDS = 0
CHART_CACHE: Dict[Tuple[str, str, str], Tuple[float, ChartResponse]] = {}

# In-memory EOD series cache built from grouped results (prev + target dates)
EOD_SERIES_CACHE: Dict[str, List[dict]] = {}


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


def _is_business_day(d: date) -> bool:
    return d.weekday() < 5


def _prev_business_day(d: date) -> date:
    cur = d - timedelta(days=1)
    while not _is_business_day(cur):
        cur -= timedelta(days=1)
    return cur


def _determine_eod_target_date(now_utc: Optional[datetime] = None) -> date:
    """Return the appropriate EOD date based on current time and market hours.
    
    After 5 PM ET on a trading day, returns today's date.
    Otherwise returns the previous business day.
    
    This allows fetching today's EOD data after market close while avoiding
    403s on free plans that cannot access today's grouped data during market hours.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    ny = now_utc.astimezone(ZoneInfo("America/New_York"))
    
    # If it's after 5 PM ET (17:00) and today was a business day, use today
    if ny.hour >= 17 and _is_business_day(ny.date()):
        return ny.date()
    
    # Otherwise use the previous business day
    return _prev_business_day(ny.date())


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
    """Returns recent daily history as list of dicts asc by date"""
    candidates = _stooq_candidates(symbol)
    # Prefer HTTPS over HTTP
    stooq_hosts = [
        "https://stooq.com",
        "https://stooq.pl",
    ]
    
    # Only try HTTP if explicitly enabled (for debugging)
    if os.getenv("STOOQ_ALLOW_HTTP", "0") == "1":
        stooq_hosts.extend([
            "http://stooq.com",
            "http://stooq.pl",
        ])
    
    for s in candidates:
        for base in stooq_hosts:
            url = f"{base}/q/d/l/?s={s}&i=d"
            try:
                # Add headers to appear more like a browser
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                resp = requests.get(url, timeout=10, headers=headers)
                
                if resp.status_code == 403:
                    print(f"Stooq access forbidden for {symbol} ({s}) via {base} - may need different headers or API key")
                    continue
                    
                if resp.status_code != 200:
                    print(f"Stooq returned status {resp.status_code} for {symbol} ({s}) via {base}")
                    continue
                    
                if not resp.text:
                    continue
                    
                # Check for rate limiting message
                if "daily hits limit" in resp.text.lower() or "exceeded" in resp.text.lower():
                    print(f"Stooq rate limit exceeded for {symbol} - falling back to Polygon if available")
                    # Don't try other Stooq hosts if we're rate limited
                    return []
                    
                if resp.text.lower().startswith("not found"):
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
            except requests.exceptions.ConnectionError as e:
                print(f"Stooq connection failed for {symbol} ({s}) via {base}: Connection refused or network issue")
                continue
            except requests.exceptions.Timeout as e:
                print(f"Stooq timeout for {symbol} ({s}) via {base}: Request took too long")
                continue
            except Exception as e:
                print(f"Stooq fetch failed for {symbol} ({s}) via {base}: {type(e).__name__}: {e}")
                continue
    
    print(f"Warning: Could not fetch Stooq data for {symbol} from any source")
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


def _extract_stock_from_grouped_cache(symbol: str) -> List[dict]:
    """Extract individual stock data from cached Polygon grouped data."""
    # Always get from SQLite - it's our persistent source of truth
    rows = PriceRepository.get_price_history(symbol, days=30, source="polygon_grouped")
    return rows


def _fetch_polygon_grouped(date_str: str) -> list:
    # Check memory cache first (outside lock for performance)
    now = time.time()
    cached = POLYGON_GROUPED_MEMORY_CACHE.get(date_str)
    if cached and (now - cached[0]) < POLYGON_GROUPED_MEMORY_CACHE_TTL:
        print(f"Using memory cached Polygon grouped data for {date_str}")
        return cached[1]
    
    # Use lock to prevent multiple threads from fetching the same data
    with _polygon_fetch_lock:
        # Double-check cache inside lock in case another thread just fetched it
        cached = POLYGON_GROUPED_MEMORY_CACHE.get(date_str)
        if cached and (now - cached[0]) < POLYGON_GROUPED_MEMORY_CACHE_TTL:
            print(f"Using memory cached Polygon grouped data for {date_str} (after lock)")
            return cached[1]
        
        # Check SQLite cache
        if PriceRepository.has_data_for_date(date_str, source="polygon_grouped"):
            print(f"Loading Polygon grouped data from SQLite for {date_str}")
            grouped_data = PriceRepository.get_grouped_data_for_date(date_str)
            if grouped_data:
                # Cache in memory for quick access
                POLYGON_GROUPED_MEMORY_CACHE[date_str] = (now, grouped_data)
                return grouped_data
        
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            print("❌ Polygon API key not set. Please set POLYGON_API_KEY in environment or .env")
            return []
        url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date_str}?adjusted=true&apiKey={api_key}"
        try:
            # Rate limit Polygon API calls
            polygon_rate_limiter.wait_if_needed()
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                snippet = (resp.text or "").strip().replace("\n", " ")[:300]
                print(f"❌ Polygon grouped failed for {date_str}: HTTP {resp.status_code} — {snippet}")
                return []
            payload = resp.json() or {}
            results = payload.get("results") or []
            
            # Save to SQLite
            try:
                records_saved = PriceRepository.save_polygon_grouped_data(date_str, results)
                print(f"Saved {records_saved} price records to SQLite for {date_str}")
            except Exception as e:
                print(f"Warning: Failed to save to SQLite: {e}")
            
            # Cache in memory
            POLYGON_GROUPED_MEMORY_CACHE[date_str] = (now, results)
            print(f"Cached Polygon grouped data in memory for {date_str} ({len(results)} symbols)")
            
            return results
        except Exception as e:
            print(f"❌ Polygon grouped exception for {date_str}: {e}")
            return []


def _fetch_daily_history_prefer_stooq(symbol: str) -> List[dict]:
    """Return recent daily history rows, preferring Stooq; fallback to SQLite cache.

    Row shape: {date, date_iso, open, high, low, close, volume}
    
    IMPORTANT: This function should NEVER make direct Polygon API calls.
    All Polygon data should come from SQLite cache populated by scheduled tasks.
    """
    stooq = _fetch_stooq_history(symbol)
    if stooq:
        return stooq
    
    # Always use whatever data we have in SQLite cache
    # Don't require a minimum number of days - that's for the caller to decide
    cached_rows = _extract_stock_from_grouped_cache(symbol)
    if cached_rows:
        print(f"Using SQLite cache for {symbol} ({len(cached_rows)} days)")
        return cached_rows
    
    # Fallback to cached EOD series built from grouped data (2-point series)
    if symbol in EOD_SERIES_CACHE:
        return EOD_SERIES_CACHE.get(symbol) or []
    
    # If we don't have data, return empty rather than hitting Polygon API
    # The scheduled cache population will handle fetching new data
    return []


def _fetch_daily_history_sqlite_only(symbol: str, days: int = 30) -> List[dict]:
    """Return recent daily history rows strictly from SQLite Polygon cache.

    Never calls Stooq or external APIs. Returns an empty list if not available.
    """
    try:
        rows = PriceRepository.get_price_history(symbol, days=days, source="polygon_grouped")
        return rows or []
    except Exception:
        return []


def _fetch_yahoo_quote(symbol: str) -> QuoteResponse:
    # Prefer Stooq; fallback to Polygon daily aggs
    now = time.time()
    if QUOTE_CACHE_TTL_SECONDS > 0:
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
        if QUOTE_CACHE_TTL_SECONDS > 0:
            QUOTE_CACHE[symbol] = (now, result)
        return result
    empty = QuoteResponse(symbol=symbol)
    if QUOTE_CACHE_TTL_SECONDS > 0:
        QUOTE_CACHE[symbol] = (now, empty)
    return empty


# --- Stooq-only helpers for fresh lookups (no Polygon fallback) ---
def _fetch_stooq_quote_only(symbol: str) -> QuoteResponse:
    now = time.time()
    if QUOTE_CACHE_TTL_SECONDS > 0:
        cached = QUOTE_CACHE.get(symbol)
        if cached and (now - cached[0]) < QUOTE_CACHE_TTL_SECONDS:
            return cached[1]
    hist = _fetch_stooq_history(symbol)
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
        if QUOTE_CACHE_TTL_SECONDS > 0:
            QUOTE_CACHE[symbol] = (now, result)
        return result
    return QuoteResponse(symbol=symbol)


def _fetch_stooq_chart_only(symbol: str, range_: str, interval: str) -> ChartResponse:
    cache_key = (symbol, range_, interval)
    now = time.time()
    if CHART_CACHE_TTL_SECONDS > 0:
        cached = CHART_CACHE.get(cache_key)
        if cached and (now - cached[0]) < CHART_CACHE_TTL_SECONDS:
            return cached[1]
    hist = _fetch_stooq_history(symbol)
    if hist:
        range_days = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "max": None
        }.get(range_)
        if range_days is not None:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=range_days)
            filtered_hist = []
            for r in hist:
                try:
                    row_date = datetime.strptime(r["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    if row_date >= cutoff_date:
                        filtered_hist.append(r)
                except Exception:
                    continue
            hist = filtered_hist
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
        if CHART_CACHE_TTL_SECONDS > 0:
            CHART_CACHE[cache_key] = (now, response)
        return response
    return ChartResponse(symbol=symbol, range=range_, interval=interval, timestamps=[], opens=[], highs=[], lows=[], closes=[], volumes=[])


def _get_price_context_stooq_only(symbol: str) -> Optional[str]:
    quote = _fetch_stooq_quote_only(symbol)
    chart = _fetch_stooq_chart_only(symbol, range_="5d", interval="1d")
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

def _fetch_yahoo_chart(symbol: str, range_: str, interval: str) -> ChartResponse:
    cache_key = (symbol, range_, interval)
    now = time.time()
    if CHART_CACHE_TTL_SECONDS > 0:
        cached = CHART_CACHE.get(cache_key)
        if cached and (now - cached[0]) < CHART_CACHE_TTL_SECONDS:
            return cached[1]
    hist = _fetch_daily_history_prefer_stooq(symbol)
    if hist:
        # Filter data based on requested range
        range_days = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "max": None
        }.get(range_)
        
        if range_days is not None:
            # Filter to only include data within the requested range
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=range_days)
            filtered_hist = []
            for r in hist:
                try:
                    row_date = datetime.strptime(r["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    if row_date >= cutoff_date:
                        filtered_hist.append(r)
                except Exception:
                    continue
            hist = filtered_hist
        
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
        if CHART_CACHE_TTL_SECONDS > 0:
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
    if CHART_CACHE_TTL_SECONDS > 0:
        CHART_CACHE[cache_key] = (now, empty)
    return empty


def _fetch_biggest_losers_polygon_eod() -> List[LoserStock]:
    """Compute biggest losers from Polygon grouped EOD for the latest trading day."""
    target = _determine_eod_target_date()
    target_str = target.isoformat()
    
    # Get today's data
    today_group = _fetch_polygon_grouped(target_str)
    if not today_group:
        print(f"⚠️ No data for target date {target_str}; returning empty losers list")
        return []
    
    # Find the previous trading day with data (skip holidays/weekends)
    prev_group = None
    prev_date = target
    max_lookback = 10  # Look back up to 10 days to find data
    
    for _ in range(max_lookback):
        prev_date = _prev_business_day(prev_date)
        prev_str = prev_date.isoformat()
        print(f"ℹ️ Checking for previous day data: {prev_str}")
        prev_group = _fetch_polygon_grouped(prev_str)
        if prev_group:
            print(f"ℹ️ Computing EOD losers for {target_str} vs previous trading day {prev_str}")
            break
    
    if not prev_group:
        print(f"⚠️ No previous trading day data found after checking {max_lookback} days; returning empty losers list")
        return []
    prev_close_by_ticker: Dict[str, float] = {}
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


# Note: _fetch_biggest_losers_polygon() has been removed as it makes direct API calls
# Always use _fetch_biggest_losers_polygon_eod() which uses cached grouped data instead


def _get_price_context(symbol: str) -> Optional[str]:
    quote = _fetch_yahoo_quote(symbol)
    chart = _fetch_yahoo_chart(symbol, range_="5d", interval="1d")
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


def prepopulate_polygon_cache():
    """Pre-populate Polygon grouped cache for recent trading days.
    
    IMPORTANT: This is the ONLY function that should make Polygon API calls.
    All other functions should read from the SQLite cache populated by this function.
    This should only be called during scheduled tasks (startup, market close, etc).
    """
    if not os.getenv("POLYGON_API_KEY"):
        return
        
    # Start from yesterday to avoid trying to fetch today's data (not available on free tier)
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    dates_to_cache = []
    
    # Get last 25 business days to ensure we have enough for 20-day volume metrics
    current_date = yesterday
    for _ in range(35):  # Look back up to 35 calendar days
        # Skip weekends
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            dates_to_cache.append(current_date.isoformat())
            if len(dates_to_cache) >= 25:  # Get 25 to be safe
                break
        current_date -= timedelta(days=1)
    
    print(f"Checking which of {len(dates_to_cache)} trading days need to be cached...")
    
    # Check which dates we actually need to fetch
    dates_to_fetch = []
    for date_str in dates_to_cache:
        if not PriceRepository.has_data_for_date(date_str, source="polygon_grouped"):
            dates_to_fetch.append(date_str)
    
    if not dates_to_fetch:
        print(f"✅ All {len(dates_to_cache)} days already cached in SQLite!")
        return
        
    print(f"📊 Need to fetch {len(dates_to_fetch)} days: {dates_to_fetch[:3]}...{dates_to_fetch[-1] if len(dates_to_fetch) > 3 else ''}")
    
    # Fetch all missing dates with rate limiting
    for i, date_str in enumerate(dates_to_fetch):
        print(f"Fetching {date_str} ({i+1}/{len(dates_to_fetch)})...")
        _fetch_polygon_grouped(date_str)
        
        # Rate limit between requests (15 seconds to be safe with free tier)
        if i < len(dates_to_fetch) - 1:
            print(f"Rate limiting: waiting 15 seconds before next request...")
            time.sleep(15)
