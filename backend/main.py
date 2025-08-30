"""Main FastAPI application module."""

from datetime import datetime, timezone, timedelta
import threading
import hashlib
import json as _json
import uuid
import time
from typing import List, Optional, Union, Generator
from hashlib import md5

from fastapi import FastAPI, HTTPException, Query, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from zoneinfo import ZoneInfo

# Import configuration
import config

# Import services
from services.market_data import (
    QuoteResponse, ChartResponse, LoserStock,
    _fetch_yahoo_quote, _fetch_yahoo_chart, _get_price_context,
    _fetch_daily_history_prefer_stooq, prepopulate_polygon_cache,
    _fetch_stooq_quote_only, _fetch_stooq_chart_only, _get_price_context_stooq_only,
    _fetch_daily_history_sqlite_only,
)
import services.market_data as _md
from services.news import _search_news_for_symbol, SourceItem
from services.analysis import _build_llm_prompt, _call_openai
from services.news_scoring import (
    compute_news_score,
    blend_scores,
    build_news_classification_prompt,
)
import services.losers as losers_service
from services.losers import (
    InterestingLoser,
    _refresh_interesting_losers_cache, market_aware_refresh_loop
)
from database.repositories.losers_repo import LosersRepository
from services.og_image import (
    OG_IMAGE_CACHE, _generate_og_image_png, _build_share_description
)
from services.jobs import submit_job, get_job, start_cleanup_daemon

# Import database components
from database.connection import get_engine, init_db
from database.repositories.signal_repo import SignalRepository
from database.repositories.features_repo import FeaturesRepository
from database.models import TradingSignal
from analysis.oversold_detector import compute_oversold_metrics

# Import utilities
from utils import _ensure_list_symbols, _sanitize_symbol

# Import for oversold scanning
import concurrent.futures
import asyncio
from sqlalchemy.orm import sessionmaker

# Imports needed for OG image functions that are still in main.py
import os
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO



app = FastAPI(
    title="Why Is The Stock Plummeting?",
    description="Searches the web and uses an LLM to explain (humorously) why your favorite stock face-planted.",
    version="0.1.0",
)
# --- Testing bridges for monkeypatch compatibility ---
# Keep original references from services.market_data
_orig_fetch_stooq_history = getattr(_md, "_fetch_stooq_history", None)
_orig_fetch_polygon_grouped = getattr(_md, "_fetch_polygon_grouped", None)
_orig_determine_eod_target_date = getattr(_md, "_determine_eod_target_date", None)

def _fetch_stooq_history(symbol: str):  # re-export for tests
    if _orig_fetch_stooq_history is None:
        raise AttributeError("_fetch_stooq_history not available")
    return _orig_fetch_stooq_history(symbol)

def _fetch_polygon_grouped(date_str: str):  # re-export for tests
    if _orig_fetch_polygon_grouped is None:
        raise AttributeError("_fetch_polygon_grouped not available")
    return _orig_fetch_polygon_grouped(date_str)

def _determine_eod_target_date(now_utc=None):  # re-export for tests
    if _orig_determine_eod_target_date is None:
        raise AttributeError("_determine_eod_target_date not available")
    return _orig_determine_eod_target_date(now_utc)

def _fetch_biggest_losers_polygon_eod():  # re-export for tests
    return _md._fetch_biggest_losers_polygon_eod()

# Bridge internal calls inside services.market_data to resolve via main.* names,
# so tests that monkeypatch main.* paths affect downstream calls.
try:
    _md._fetch_stooq_history = lambda symbol: _fetch_stooq_history(symbol)  # type: ignore
    _md._fetch_polygon_grouped = lambda date_str: _fetch_polygon_grouped(date_str)  # type: ignore
    _md._determine_eod_target_date = lambda now_utc=None: _determine_eod_target_date(now_utc)  # type: ignore
except Exception:
    pass

# Bridge functions for services.og_image so tests monkeypatching main.* affect it
try:
    import services.og_image as _og
    _og._fetch_stooq_quote_only = lambda symbol: _fetch_yahoo_quote(symbol)  # type: ignore
    _og._fetch_stooq_chart_only = lambda symbol, range_, interval: _fetch_yahoo_chart(symbol, range_, interval)  # type: ignore
    _og._get_price_context_stooq_only = lambda symbol: _get_price_context_stooq_only(symbol)  # type: ignore
except Exception:
    pass



# CORS configuration
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
if config.PUBLIC_WEB_ORIGIN:
    allowed_origins.append(config.PUBLIC_WEB_ORIGIN)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class AnalyzeRequest(BaseModel):
    symbols: Union[str, List[str]]
    days: int = 7
    max_results: int = 8
    tone: str = "humorous"


class SymbolAnalysis(BaseModel):
    symbol: str
    summary: str
    sources: List[SourceItem]


class AnalyzeResponse(BaseModel):
    results: List[SymbolAnalysis]


class BiggestLosersResponse(BaseModel):
    losers: List[LoserStock]
    last_updated: str
    session: str | None = None


class InterestingLosersResponse(BaseModel):
    losers: List[InterestingLoser]
    last_updated: str
    session: str | None = None


class OversoldScanRequest(BaseModel):
    symbols: Union[str, List[str]]
    top: int = 20
    include_news: bool = False
    news_timeout_seconds: Optional[int] = None


class PromoteRequest(BaseModel):
    top: int = 10
    threshold: float = -0.5  # oversold_score <= threshold
    cooldown_minutes: int = 1440


# Oversold scan cache
_OVERSOLD_SCAN_CACHE = {}


def get_db_session() -> Generator[Session, None, None]:
    engine = get_engine()
    maker = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = maker()
    try:
        yield db
    finally:
        db.close()


def _ensure_cache_initialized():
    """Initialize curated losers cache once in the background."""
    if config._app_started:
        return
    threading.Thread(target=_refresh_interesting_losers_cache, daemon=True).start()
    config._app_started = True


def _oversold_scan(symbols: List[str], rid: Optional[str] = None) -> List[dict]:
    # Pre-fetch yesterday's grouped data if we have many symbols
    if len(symbols) > 5:
        from services.market_data import _determine_eod_target_date, _fetch_polygon_grouped
        yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
        # Try to populate cache for recent completed trading day
        for days_back in range(5):
            date = yesterday - timedelta(days=days_back)
            if date.weekday() < 5:  # Skip weekends
                _fetch_polygon_grouped(date.isoformat())
                break
    
    def worker(sym: str):
        try:
            # Use strictly SQLite Polygon cache for scans to avoid Stooq rate limits
            t0 = time.time()
            hist = _fetch_daily_history_sqlite_only(sym)
            t_hist = (time.time() - t0) * 1000.0
            try:
                if rid is not None:
                    print(f"[{rid}] hist {sym}: rows={len(hist)} in {t_hist:.1f}ms")
            except Exception:
                pass
            if not hist:
                return None
            t1 = time.time()
            m = compute_oversold_metrics(hist)
            t_metrics = (time.time() - t1) * 1000.0
            try:
                if rid is not None:
                    print(f"[{rid}] metrics {sym}: oversold={m.oversold_score:.3f} z={m.zscore_close:.2f} in {t_metrics:.1f}ms")
            except Exception:
                pass
            return {
                "symbol": sym,
                "metrics": {
                    "return_1d": m.return_1d,
                    "return_3d": m.return_3d,
                    "gap_pct": m.gap_pct,
                    "volume_ratio_20d": m.volume_ratio_20d,
                    "true_range_pct": m.true_range_pct,
                    "zscore_close": m.zscore_close,
                    "oversold_score": m.oversold_score,
                },
            }
        except Exception:
            return None

    # Threaded fetch to avoid long sequential waits
    # Note: Using fewer workers to respect rate limits
    results: List[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for res in pool.map(worker, symbols):
            if res and res.get("metrics", {}).get("oversold_score") is not None:
                results.append(res)
    # Rank: most negative oversold_score first
    results.sort(key=lambda r: r["metrics"]["oversold_score"])
    return results


def _build_news_classification_prompt(symbol: str, headlines: List[SourceItem]) -> str:
    # Backward compatibility: delegate to service
    return build_news_classification_prompt(symbol, headlines)

def _run_oversold_scheduled_scan():
    try:
        from datetime import timedelta
        # Skip weekends (ET)
        now = datetime.now(ZoneInfo("America/New_York"))
        if now.weekday() >= 5:
            return
        # Compute
        ranked = _oversold_scan(config.OVERSOLD_UNIVERSE)
        # Persist top 100 to features table
        engine = get_engine()
        maker = sessionmaker(bind=engine, autoflush=False, autocommit=False)
        with maker() as db:
            repo = FeaturesRepository(db)
            for item in ranked[:100]:
                payload = {
                    "symbol": item["symbol"],
                    "metrics": item["metrics"],
                }
                repo.add_features(symbol=item["symbol"], feature_set="oversold_v1", features_json=_json.dumps(payload))
    except Exception as e:
        print(f"❌ scheduled oversold scan failed: {e}")


def _start_oversold_scheduler():
    if not config.OVERSOLD_SCHEDULE_ENABLED:
        print("⚠️ Oversold scheduler disabled via env")
        return
    def loop():
        while True:
            try:
                _run_oversold_scheduled_scan()
            except Exception as e:
                print(f"oversold scheduler error: {e}")
            time.sleep(900)  # 15 minutes
    threading.Thread(target=loop, daemon=True).start()


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    print("🚀 Application starting up...")
    # Ensure this only runs once (avoids duplicate threads under reload/watchers)
    if config._app_started:
        print("⚠️ Startup already initialized; skipping duplicate init")
        return
    
    # Initialize SQLite database
    try:
        init_db()
    except Exception as e:
        print(f"⚠️ DB init failed: {e}")

    # Populate caches at startup in threads so FastAPI startup completes
    def init_cache():
        try:
            # Skip pre-population in development to avoid rate limiting on frequent restarts
            if not config.SKIP_CACHE_PREPOPULATION:
                # Only prepopulate if we haven't done it recently
                from services.market_data import POLYGON_GROUPED_MEMORY_CACHE
                last_prepop_key = "_last_polygon_prepopulation"
                last_prepop = POLYGON_GROUPED_MEMORY_CACHE.get(last_prepop_key)
                now = time.time()
                
                # Only prepopulate if it's been more than 30 minutes since last time
                if not last_prepop or (now - last_prepop[0]) > 1800:
                    print("📊 Ensuring 25 days of market data is cached in SQLite...")
                    prepopulate_polygon_cache()
                    POLYGON_GROUPED_MEMORY_CACHE[last_prepop_key] = (now, True)
                else:
                    print("Skipping Polygon cache pre-population (too recent)")
            else:
                print("Skipping cache pre-population (SKIP_CACHE_PREPOPULATION=1)")
                
            # Refresh interesting losers only if today's EOD batch is missing
            try:
                from database.repositories.losers_repo import LosersRepository
                from services.market_data import _determine_eod_target_date
                batch_id = _determine_eod_target_date().isoformat()
                existing = LosersRepository.get_ranked(batch_id=batch_id, limit=1)
                if existing:
                    print(f"✅ Skipping losers refresh: EOD batch {batch_id} already present")
                else:
                    print(f"🔄 Refreshing interesting losers for EOD {batch_id}...")
                    _refresh_interesting_losers_cache()
            except Exception as e:
                print(f"Startup losers check failed: {e}")
        except Exception as e:
            print(f"Startup interesting cache init failed: {e}")
    
    threading.Thread(target=init_cache, daemon=True).start()

    # Start market-aware background refresh
    threading.Thread(target=market_aware_refresh_loop, daemon=True).start()
    
    # Start oversold scheduler
    _start_oversold_scheduler()
    
    # Start background job cleanup daemon
    try:
        start_cleanup_daemon()
    except Exception as e:
        print(f"Job cleanup daemon failed to start: {e}")
    
    config._app_started = True
    print("✅ Startup initialization queued")



@app.get("/premium/signals/active")
def premium_signals_active(
    limit: int = Query(20, ge=1, le=100),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    db: Session = Depends(get_db_session),
):
    repo = SignalRepository(db)
    rows = repo.list_active(limit=limit, min_confidence=min_confidence)
    # Shape response minimally for now
    payload = [
        {
            "id": r.id,
            "symbol": r.symbol,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "entry_price": r.entry_price,
            "current_price": r.current_price,
            "target_price": r.target_price,
            "stop_loss_price": r.stop_loss_price,
            "confidence_score": r.confidence_score,
            "status": r.status,
        }
        for r in rows
    ]
    return {"signals": payload, "total": len(payload)}


@app.get("/premium/oversold/{symbol}")
def premium_oversold_symbol(symbol: str):
    """Compute oversold metrics for a single symbol using available data sources."""
    # Use the function with Polygon fallback
    history = _fetch_daily_history_prefer_stooq(symbol)
    if not history:
        raise HTTPException(404, detail="No history available from any source")
    metrics = compute_oversold_metrics(history)
    return {
        "symbol": symbol.upper(),
        "metrics": {
            "return_1d": metrics.return_1d,
            "return_3d": metrics.return_3d,
            "gap_pct": metrics.gap_pct,
            "volume_ratio_20d": metrics.volume_ratio_20d,
            "true_range_pct": metrics.true_range_pct,
            "zscore_close": metrics.zscore_close,
            "oversold_score": metrics.oversold_score,
        },
    }

# --- Oversold batch scan with simple in-memory TTL cache ---
_OVERSOLD_SCAN_CACHE = {}
_OVERSOLD_SCAN_CACHE_TTL_SECONDS = int(config.OVERSOLD_SCAN_CACHE_TTL_SECONDS)


class OversoldScanRequest(BaseModel):
    symbols: Union[str, List[str]]
    top: int = 20
    include_news: bool = False


@app.post("/premium/oversold/scan")
def premium_oversold_scan(req: OversoldScanRequest):
    symbols = _ensure_list_symbols(req.symbols)
    if not symbols:
        raise HTTPException(400, detail="No symbols provided")
    # Basic request diagnostics
    try:
        rid = (uuid.uuid4().hex[:6])
        print(f"[{rid}] ↪️ oversold_scan start: {len(symbols)} symbols, top={req.top}, include_news={getattr(req, 'include_news', False)}, news_timeout={getattr(req, 'news_timeout_seconds', None)}")
        print(f"[{rid}] symbols: {', '.join(symbols[:12])}{' …' if len(symbols) > 12 else ''}")
    except Exception:
        rid = None
    # Cache key
    key_raw = ",".join(sorted(symbols)) + f"|{req.top}"
    key = hashlib.md5(key_raw.encode("utf-8")).hexdigest()
    now_ts = time.time()
    hit = _OVERSOLD_SCAN_CACHE.get(key)
    if hit and (now_ts - hit[0]) < _OVERSOLD_SCAN_CACHE_TTL_SECONDS:
        ranked = hit[1]
    else:
        t0 = time.time()
        ranked = _oversold_scan(symbols, rid=rid)
        try:
            if rid is not None:
                print(f"[{rid}] ✅ oversold metrics computed for {len(ranked)} symbols in {int((time.time()-t0)*1000)}ms")
        except Exception:
            pass
        _OVERSOLD_SCAN_CACHE[key] = (now_ts, ranked)

    out = ranked[: req.top]
    if req.include_news:
        # Compute news scores in parallel. If no timeout specified, wait for all (default 10 minutes).
        import concurrent.futures as _cf
        from concurrent.futures import ThreadPoolExecutor, wait
        symbols_for_news = [it.get("symbol") for it in out]
        effective_cap = req.news_timeout_seconds if isinstance(getattr(req, "news_timeout_seconds", None), int) else 600
        try:
            if rid is not None:
                cap_txt = "all" if effective_cap is None else str(effective_cap)
                print(f"[{rid}] 📰 news scoring start for {len(out)} symbols (cap {cap_txt}s)…")
        except Exception:
            pass

        def _score_one(sym: str):
            try:
                return compute_news_score(sym)
            except Exception as e:
                print(f"[{rid}] ❌ news_score failed for {sym}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                return None

        scores = [None] * len(out)
        executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=4)
        try:
            future_to_idx = {executor.submit(_score_one, s): idx for idx, s in enumerate(symbols_for_news)}
            t_ns0 = time.time()
            done, not_done = wait(set(future_to_idx.keys()), timeout=effective_cap)
            for fut in done:
                idx = future_to_idx[fut]
                try:
                    scores[idx] = fut.result()
                except Exception:
                    scores[idx] = None
            # Best-effort cancel stragglers; do not block shutdown
            for fut in not_done:
                try:
                    fut.cancel()
                except Exception:
                    pass
            try:
                if rid is not None:
                    finished = sum(1 for s in scores if isinstance(s, (int, float)))
                    elapsed_ms = int((time.time() - t_ns0) * 1000)
                    print(f"[{rid}] 📰 news scoring done {finished}/{len(scores)} in {elapsed_ms}ms (cap {effective_cap}s)")
            except Exception:
                pass
        finally:
            # If we had a timeout, don't wait for stragglers; let threads finish in background
            executor.shutdown(wait=(effective_cap is None))

        # Apply scores
        for it, nscore in zip(out, scores):
            m = it.get("metrics") or {}
            oscore = m.get("oversold_score")
            m["news_score"] = nscore
            m["blended_score"] = blend_scores(oscore, nscore)
            it["metrics"] = m
    result = {"candidates": out, "total": len(ranked), "cached": hit is not None and (now_ts - hit[0]) < _OVERSOLD_SCAN_CACHE_TTL_SECONDS}
    try:
        if rid is not None:
            print(f"[{rid}] ↩️ oversold_scan done: returning {len(out)} items")
    except Exception:
        pass
    return result


@app.get("/premium/oversold/top")
def premium_oversold_top(
    limit: int = Query(50, ge=1, le=200),
    since_minutes: int = Query(60, ge=1, le=1440),
    min_abs_oversold: Optional[float] = Query(None, ge=0.0),
    include_news: bool = Query(False),
    db: Session = Depends(get_db_session),
):
    repo = FeaturesRepository(db)
    rows = repo.list_recent(feature_set="oversold_v1", lookback_minutes=since_minutes, limit=limit)
    items = []
    seen = set()
    for r in rows:
        try:
            payload = _json.loads(r.features_json or "{}")
            symbol = (payload.get("symbol") or r.symbol or "").upper()
            if not symbol or symbol in seen:
                continue
            metrics = payload.get("metrics") or {}
            score = metrics.get("oversold_score")
            if min_abs_oversold is not None and (score is None or abs(score) < min_abs_oversold):
                continue
            # Optionally enrich with news and blended score
            if include_news:
                try:
                    nscore = compute_news_score(symbol)
                except Exception:
                    nscore = None
                metrics["news_score"] = nscore
                metrics["blended_score"] = blend_scores(score, nscore)
            items.append({"symbol": symbol, "timestamp": r.timestamp.isoformat() if r.timestamp else None, "metrics": metrics})
            seen.add(symbol)
            if len(items) >= limit:
                break
        except Exception:
            continue
    items.sort(key=lambda x: (x["metrics"].get("oversold_score") if x.get("metrics") else 0))
    return {"candidates": items[:limit], "total": len(items)}


class PromoteRequest(BaseModel):
    top: int = 10
    threshold: float = -0.5  # oversold_score <= threshold
    cooldown_minutes: int = 1440


@app.post("/premium/oversold/promote")
def premium_oversold_promote(req: PromoteRequest, db: Session = Depends(get_db_session)):
    # Get top candidates from recent store
    top_resp = premium_oversold_top(limit=req.top * 2, since_minutes=120, min_abs_oversold=None, db=db)  # type: ignore
    candidates = top_resp.get("candidates", [])
    repo_sig = SignalRepository(db)
    promoted = []
    for it in candidates:
        sym = it.get("symbol")
        m = (it.get("metrics") or {})
        score = m.get("oversold_score")
        if score is None or score > req.threshold:
            continue
        if repo_sig.has_recent_active(sym, minutes=req.cooldown_minutes):
            continue
        # Compute news_score and gate promotions with clearly sticky news
        news_score = compute_news_score(sym)
        if news_score is not None and news_score < 0.4:
            # News suggests low near-term reversion odds; skip
            continue
        # Get current price from last close (Stooq or Polygon)
        hist = _fetch_daily_history_prefer_stooq(sym)
        if not hist or not hist[-1].get("close"):
            continue
        price = float(hist[-1]["close"])  # entry reference price
        # Simple MVP: +3% target, -3% stop
        target = price * 1.03
        stop = price * 0.97
        # Blend confidence: oversold strength with news_score (neutral 0.5 if unavailable)
        oversold_strength = min(1.0, abs(float(score))) if isinstance(score, (int, float)) else 0.5
        news_component = news_score if isinstance(news_score, (int, float)) else 0.5
        confidence = max(0.1, min(1.0, (0.6 * oversold_strength) + (0.4 * news_component)))
        s = TradingSignal(
            id=str(uuid.uuid4()),
            symbol=sym,
            signal_type="mean_reversion",
            entry_price=price,
            current_price=price,
            target_price=target,
            stop_loss_price=stop,
            position_size_pct=None,
            confidence_score=confidence,
            oversold_score=float(score) if score is not None else 0.0,
            news_score=news_score,
            ml_prediction=None,
            triggering_event_id=None,
            analysis_summary="Rule-based oversold promotion",
            features_json=_json.dumps({**m, "news_score": news_score}),
            status="active",
            expires_at=datetime.utcnow() + timedelta(days=5),
        )
        repo_sig.create(s)
        promoted.append({"id": s.id, "symbol": s.symbol})
        if len(promoted) >= req.top:
            break
    return {"promoted": promoted, "count": len(promoted)}


@app.get("/premium/oversold/{symbol}/news")
def premium_oversold_news(symbol: str, db: Session = Depends(get_db_session)):
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise HTTPException(400, detail="Symbol required")
    # Fetch recent headlines
    headlines = _search_news_for_symbol(symbol, days=3, max_results=6)
    # Build classification prompt via helper
    prompt = _build_news_classification_prompt(symbol, headlines)
    raw = _call_openai(prompt)
    # Persist alongside oversold features for this symbol
    try:
        repo = FeaturesRepository(db)
        payload = {"symbol": symbol, "headlines": [h.model_dump() for h in headlines], "llm": raw}
        repo.add_features(symbol=symbol, feature_set="oversold_news_v1", features_json=_json.dumps(payload))
    except Exception:
        pass
    return {"symbol": symbol, "headlines": [h.model_dump() for h in headlines], "llm": raw}






# DB bootstrap moved to database/connection.py










## Removed legacy popular symbols list


## Removed legacy losers-from-news search flow


## Removed legacy losers extraction from news


## Removed legacy biggest losers sync fetcher









## Removed S&P/NASDAQ universe filtering to avoid redundancy per product direction


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    symbols = _ensure_list_symbols(request.symbols)
    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols provided")

    results: List[SymbolAnalysis] = []
    for symbol in symbols:
        sources = _search_news_for_symbol(symbol, request.days, request.max_results)
        price_ctx = _get_price_context_stooq_only(symbol)
        prompt = _build_llm_prompt(symbol, sources, request.tone, price_ctx)
        summary = _call_openai(prompt)
        results.append(
            SymbolAnalysis(symbol=symbol, summary=summary, sources=sources)
        )

    return AnalyzeResponse(results=results)


@app.post("/analyze/async", status_code=202)
async def analyze_async(request: AnalyzeRequest):
    symbols = _ensure_list_symbols(request.symbols)
    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols provided")

    def job_func():
        out: List[dict] = []
        for symbol in symbols:
            sources = _search_news_for_symbol(symbol, request.days, request.max_results)
            price_ctx = _get_price_context_stooq_only(symbol)
            prompt = _build_llm_prompt(symbol, sources, request.tone, price_ctx)
            summary = _call_openai(prompt)
            out.append({
                "symbol": symbol,
                "summary": summary,
                "sources": [s.model_dump() for s in sources],
            })
        return {"results": out}

    job_id = submit_job("analyze", job_func)
    return {"job_id": job_id, "status_url": f"/jobs/{job_id}"}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, detail="Job not found")
    # Shape for frontend
    return {
        "id": job.get("id"),
        "type": job.get("type"),
        "status": job.get("status"),
        "result": job.get("result"),
        "error": job.get("error"),
        "progress": job.get("progress"),
    }


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
    """Return curated losers from SQLite if available; fallback to memory cache.

    Never compute in the request path. Data is refreshed by background jobs.
    """
    # Try DB first
    try:
        rows = LosersRepository.get_ranked(limit=top)
        if rows:
            losers = [
                InterestingLoser(
                    symbol=r.symbol,
                    name=None,
                    price=r.price,
                    change=r.change,
                    change_percent=r.change_percent,
                    volume=r.volume,
                    reason=r.reason,
                ) for r in rows
            ]
            # Determine last_updated from most recent created_at
            latest_ts = max([getattr(r, "created_at", None) for r in rows if getattr(r, "created_at", None)], default=None)
            last_updated = (latest_ts if latest_ts else datetime.now(timezone.utc)).isoformat()
            return InterestingLosersResponse(losers=losers, last_updated=last_updated, session="EOD")
    except Exception as e:
        print(f"⚠️ Failed to read losers from DB: {e}")

    # Fallback: empty if DB not available
    return InterestingLosersResponse(losers=[], last_updated=datetime.now(timezone.utc).isoformat(), session="EOD")



@app.get("/og-image/{symbol}.png")
async def og_image(symbol: str) -> Response:
    symbol = _sanitize_symbol(symbol)
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    
    # Use date-based cache key
    today = datetime.now(ZoneInfo("America/New_York")).date()
    cache_key = f"{symbol}_{today.isoformat()}"
    
    cached = OG_IMAGE_CACHE.get(cache_key)
    if cached:
        image_bytes = cached
    else:
        image_bytes = _generate_og_image_png(symbol)
        OG_IMAGE_CACHE[cache_key] = image_bytes
    
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
    
    # Use date-based cache key
    today = datetime.now(ZoneInfo("America/New_York")).date()
    cache_key = f"{symbol}_{today.isoformat()}"
    
    if cache_key in OG_IMAGE_CACHE:
        return Response(status_code=204)
    
    image_bytes = _generate_og_image_png(symbol)
    OG_IMAGE_CACHE[cache_key] = image_bytes
    return Response(status_code=201)


@app.post("/og-image/warm/{symbol}/async", status_code=202)
async def og_image_warm_async(symbol: str):
    """Queue background OG image generation and return a job id for polling."""
    symbol = _sanitize_symbol(symbol)
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    today = datetime.now(ZoneInfo("America/New_York")).date()
    cache_key = f"{symbol}_{today.isoformat()}"

    def job_func():
        # Generate and cache
        image_bytes = _generate_og_image_png(symbol)
        OG_IMAGE_CACHE[cache_key] = image_bytes
        return {"cache_key": cache_key}

    job_id = submit_job("og_image_warm", job_func)
    return {"job_id": job_id, "status_url": f"/jobs/{job_id}", "status_check": f"/og-image/status/{symbol}"}


@app.get("/og-image/status/{symbol}")
async def og_image_status(symbol: str):
    symbol = _sanitize_symbol(symbol)
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    today = datetime.now(ZoneInfo("America/New_York")).date()
    cache_key = f"{symbol}_{today.isoformat()}"
    cached = OG_IMAGE_CACHE.get(cache_key)
    if not cached:
        return {"symbol": symbol, "ready": False}
    etag = md5(cached).hexdigest()
    return {"symbol": symbol, "ready": True, "etag": etag}


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


