"""Service for identifying and ranking interesting stock losers."""

import os
import time
from typing import List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pydantic import BaseModel

from services.market_data import (
	LoserStock, 
	_stooq_day_change_percent
)
from services.news import _search_news_for_symbol, SourceItem
from database.repositories.losers_repo import LosersRepository
from services.analysis import _call_openai


class InterestingLoser(LoserStock):
    reason: Optional[str] = None


# No in-memory losers cache: persisted in SQLite and fetched on demand


def _call_openai_compat(prompt: str) -> str:
    """Use main._call_openai if present (test monkeypatch), else services.analysis._call_openai."""
    try:
        import main as main_module  # type: ignore
        func = getattr(main_module, "_call_openai", None)
        if callable(func):
            return func(prompt)
    except Exception:
        pass
    return _call_openai(prompt)


def _rank_interesting_losers(candidates: List[LoserStock], top_n: int = 10) -> List[InterestingLoser]:
    """Use LLM + lightweight headline search to rank losers by 'newsworthiness'."""
    if not candidates:
        return []
    # Fetch minimal headlines for each candidate (cap to avoid rate limiting)
    enriched: List[Tuple[LoserStock, List[SourceItem]]] = []
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
        f"You are a sharp markets editor. From the following decliners, pick the most newsworthy {top_n}.\n"
        "Prefer names with clear catalysts (earnings, guidance, downgrades, litigation, macro, product news) and broad interest.\n"
        "Avoid microcaps and illiquid names unless there is major news.\n"
        "Return a JSON array of objects with: symbol, reason (1 sentence).\n\n"
        f"Candidates (symbol, today's % change, top headlines):\n{catalog}\n\n"
        "Return strictly JSON."
    )
    # Compat: prefer main._call_openai if monkeypatched in tests
    def _call_openai_compat(p: str) -> str:
        try:
            import main as main_module  # type: ignore
            func = getattr(main_module, "_call_openai", None)
            if callable(func):
                return func(p)
        except Exception:
            pass
        return _call_openai(p)

    try:
        raw = _call_openai_compat(prompt)
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
            picked.append(InterestingLoser(**st.model_dump(), reason=item.get("reason")))
        # Fallback: if LLM picked none, return first N candidates with simple reasons
        if not picked and enriched:
            fallback: List[InterestingLoser] = []
            for c, news in enriched[:top_n]:
                reason = "; ".join([n.title for n in news[:2] if n.title]) or "Notable decliner; headlines pending"
                fallback.append(InterestingLoser(**c.model_dump(), reason=reason))
            return fallback
        return picked[:top_n]
    except Exception as e:
        print(f"❌ LLM ranking failed: {e}")
        # Robust fallback
        fallback: List[InterestingLoser] = []
        for c, news in enriched[:top_n]:
            reason = "; ".join([n.title for n in news[:2] if n.title]) or "Notable decliner; headlines pending"
            fallback.append(InterestingLoser(**c.model_dump(), reason=reason))
        return fallback


def _filter_ranked_losers_by_stooq(
    ranked: List[InterestingLoser],
    tolerance_pp: float,
    min_abs_for_sign_check: float = 5.0,
) -> List[InterestingLoser]:
    """Downrank items whose Polygon EOD % differs materially from Stooq.

    Instead of removing items, apply a penalty and re-order so that
    mismatches fall toward the bottom while keeping the list length intact.

    Penalties are calculated based on:
      - Base: 0 for within tolerance
      - No Stooq data: If move > threshold (default 30%), penalty = 15
      - Stepwise: +1 for each step_pp (default 2%) over tolerance
      - Sign bonus: Additional penalty (default +10) for sign mismatches
      - Max penalty: Capped at max_penalty (default 50)
    
    Note: When Stooq data is unavailable (e.g., rate limited), extreme
    moves are penalized to prevent unverifiable outliers from ranking high.
    """
    if not ranked:
        return ranked

    # Tunables via env
    try:
        step_pp = float(os.getenv("LOSERS_STOOQ_PENALTY_STEP_PPTS", "2.0"))
    except Exception:
        step_pp = 2.0
    try:
        sign_bonus = int(os.getenv("LOSERS_STOOQ_PENALTY_SIGN_BONUS", "10"))
    except Exception:
        sign_bonus = 10
    try:
        max_penalty = int(os.getenv("LOSERS_STOOQ_MAX_PENALTY", "50"))
    except Exception:
        max_penalty = 50
    try:
        no_data_extreme_threshold = float(os.getenv("LOSERS_STOOQ_NO_DATA_EXTREME_THRESHOLD", "20.0"))
    except Exception:
        no_data_extreme_threshold = 20.0
    try:
        no_data_extreme_penalty = int(os.getenv("LOSERS_STOOQ_NO_DATA_EXTREME_PENALTY", "30"))
    except Exception:
        no_data_extreme_penalty = 30

    scored: List[Tuple[int, int, InterestingLoser]] = []
    for idx, item in enumerate(ranked):
        penalty = 0
        try:
            stooq_pct = _stooq_day_change_percent(item.symbol)
            poly_pct = item.change_percent
            if poly_pct is None:
                penalty = 0
            elif stooq_pct is None:
                # No Stooq data available - treat extreme moves as suspicious
                if poly_pct is not None and abs(poly_pct) > no_data_extreme_threshold:
                    penalty = no_data_extreme_penalty
                else:
                    penalty = 0
            else:
                diff_pp = abs(poly_pct - stooq_pct)
                signs_differ = (poly_pct < 0) != (stooq_pct < 0)
                # Base stepwise penalty once over tolerance
                if diff_pp > tolerance_pp and step_pp > 0:
                    overflow = diff_pp - tolerance_pp
                    penalty = int(overflow // step_pp) + 1
                # Extra penalty for meaningful sign mismatches
                if signs_differ and max(abs(poly_pct), abs(stooq_pct)) >= min_abs_for_sign_check:
                    penalty += max(sign_bonus, 1)
                # Cap to keep sort keys bounded
                if penalty > max_penalty:
                    penalty = max_penalty
        except Exception:
            penalty = 0
        scored.append((penalty, idx, item))

    scored.sort(key=lambda t: (t[0], t[1]))  # stable: lower penalty first, then original order
    return [it for _, __, it in scored]


def _refresh_interesting_losers_cache():
    """Refresh the interesting losers cache by computing EOD losers and ranking."""
    try:
        # Compat: prefer main._fetch_biggest_losers_polygon_eod if monkeypatched in tests
        def _get_eod_losers():
            try:
                import main as main_module  # type: ignore
                func = getattr(main_module, "_fetch_biggest_losers_polygon_eod", None)
                if callable(func):
                    return func()
            except Exception:
                pass
            from services.market_data import _fetch_biggest_losers_polygon_eod as _fallback
            return _fallback()

        full = _get_eod_losers()
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
            raw = _call_openai_compat(prompt)
            import json, re
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            arr = json.loads(m.group(0) if m else raw)
            pickset = {str(x).strip().upper() for x in arr if isinstance(x, (str,))}
            reduced = [s for s in stage1 if s.symbol in pickset][:30]
        except Exception:
            reduced = stage1[:30]
        ranked = _rank_interesting_losers(reduced, 15)
        # Final Stooq reconciliation: downrank mismatches instead of dropping
        try:
            tol_pp = float(os.getenv("LOSERS_FINAL_STOOQ_TOLERANCE_PPTS", "10.0"))
            enabled = os.getenv("LOSERS_FINAL_STOOQ_CHECK", "1") not in {"0", "false", "False"}
            if enabled:
                ranked = _filter_ranked_losers_by_stooq(ranked, tol_pp)
        except Exception:
            pass
        # Persist to SQLite with a batch id (target EOD date)
        try:
            def _determine_eod_target_date_compat():
                try:
                    import main as main_module  # type: ignore
                    func = getattr(main_module, "_determine_eod_target_date", None)
                    if callable(func):
                        return func()
                except Exception:
                    pass
                from services.market_data import _determine_eod_target_date as _fallback
                return _fallback()

            batch_id = _determine_eod_target_date_compat().isoformat()
        except Exception:
            batch_id = datetime.now(timezone.utc).date().isoformat()
        # Save to DB
        try:
            LosersRepository.save_ranked(
                batch_id,
                [r.model_dump() for r in ranked],
                session_label="EOD",
            )
        except Exception as e:
            print(f"⚠️ Failed to save interesting losers to DB: {e}")

        print(f"🔄 Interesting losers cache refreshed with {len(ranked)} items (saved to DB batch {batch_id})")
    except Exception as e:
        print(f"❌ Failed to refresh interesting losers cache: {e}")


def _get_interesting_losers(candidates: List[LoserStock], top_n: int = 10) -> List[InterestingLoser]:
    # Compute from candidates; persistence handled by refresh function
    return _rank_interesting_losers(candidates, top_n)


def market_aware_refresh_loop():
    """Refresh losers cache ~1 hour after market close (5pm ET)."""
    while True:
        try:
            # Calculate time until next refresh (5pm ET)
            now = datetime.now(ZoneInfo("America/New_York"))
            today_refresh = now.replace(hour=17, minute=0, second=0, microsecond=0)  # 5pm ET
            
            # If we're past today's refresh time, schedule for tomorrow
            if now >= today_refresh:
                next_refresh = today_refresh + timedelta(days=1)
            else:
                next_refresh = today_refresh
            
            # Skip weekends
            while next_refresh.weekday() >= 5:  # Saturday = 5, Sunday = 6
                next_refresh += timedelta(days=1)
            
            # Calculate seconds until next refresh
            sleep_seconds = (next_refresh - now).total_seconds()
            
            # Ensure we're not sleeping for too long (max 25 hours) or negative time
            if sleep_seconds < 0:
                print(f"⚠️ Calculated negative sleep time ({sleep_seconds}s), scheduling for tomorrow")
                next_refresh = next_refresh + timedelta(days=1)
                while next_refresh.weekday() >= 5:
                    next_refresh += timedelta(days=1)
                sleep_seconds = (next_refresh - now).total_seconds()
            elif sleep_seconds > 90000:  # More than 25 hours
                print(f"⚠️ Calculated excessive sleep time ({sleep_seconds/3600:.1f}h), capping at 25 hours")
                sleep_seconds = 90000
            
            print(f"📅 Next losers refresh scheduled for {next_refresh.strftime('%Y-%m-%d %H:%M %Z')} ({sleep_seconds/3600:.1f} hours from now)")
            
            # Sleep until refresh time, but wake up at least every hour to check
            while sleep_seconds > 0:
                sleep_chunk = min(3600, sleep_seconds)  # Sleep in 1-hour chunks max
                time.sleep(sleep_chunk)
                sleep_seconds -= sleep_chunk
                if sleep_seconds > 0:
                    now = datetime.now(ZoneInfo("America/New_York"))
                    sleep_seconds = (next_refresh - now).total_seconds()
            
            # Fetch the latest trading day's data first
            try:
                from services.market_data import _determine_eod_target_date, _fetch_polygon_grouped
                from database.repositories.price_repo import PriceRepository
                
                latest_trading_day = _determine_eod_target_date().isoformat()
                print(f"📊 Fetching EOD data for {latest_trading_day}...")
                if not PriceRepository.has_data_for_date(latest_trading_day, source="polygon_grouped"):
                    _fetch_polygon_grouped(latest_trading_day)
                    print(f"✅ Cached EOD data for {latest_trading_day}")
                else:
                    print(f"✅ EOD data for {latest_trading_day} already cached")
            except Exception as e:
                print(f"❌ Failed to fetch EOD data: {e}")
            
            # Perform the refresh
            print("🔄 Running scheduled market-close losers refresh...")
            _refresh_interesting_losers_cache()
            print("✅ Market-close losers refresh completed")
            
            # Also cleanup old caches
            from services.news import cleanup_old_news_cache
            from services.og_image import cleanup_old_og_cache
            cleanup_old_news_cache()
            cleanup_old_og_cache()
            
            # Important: After a successful refresh, ensure we schedule for the NEXT trading day
            # not immediately recalculate which might give us the same day
            # Force scheduling for tomorrow by updating our reference time
            now = datetime.now(ZoneInfo("America/New_York"))
            # If we just ran, we're definitely past 5 PM, so ensure we skip to tomorrow
            if now.hour >= 17:
                # We just ran after 5 PM, so definitely schedule for tomorrow or next trading day
                tomorrow = now.replace(hour=17, minute=0, second=0, microsecond=0) + timedelta(days=1)
                while tomorrow.weekday() >= 5:  # Skip weekends
                    tomorrow += timedelta(days=1)
                wait_seconds = (tomorrow - now).total_seconds()
                print(f"📅 After refresh, next run scheduled for {tomorrow.strftime('%Y-%m-%d %H:%M %Z')} ({wait_seconds/3600:.1f} hours from now)")
                # Sleep a bit before continuing the loop to avoid any race conditions
                time.sleep(60)
            
        except Exception as e:
            print(f"❌ Market-aware refresh error: {e}")
            time.sleep(300)  # 5 minutes on error
