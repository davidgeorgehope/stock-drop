"""News scoring helpers and caching.

Encapsulates LLM-assisted news classification, score caching in SQLite,
and simple blending utilities used by oversold endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
import json as _json

from sqlalchemy.orm import sessionmaker

from database.connection import get_engine
from database.repositories.features_repo import FeaturesRepository
from services.news import _search_news_for_symbol, SourceItem
from services.analysis import _call_openai


NEWS_SCORE_FEATURE_SET = "oversold_news_score_v1"
NEWS_SCORE_TTL_SECONDS = 86400  # 1 day


def build_news_classification_prompt(symbol: str, headlines: List[SourceItem]) -> str:
    """Build strict JSON classification prompt for recent headlines."""
    prompt = (
        "Classify the following headlines for {sym}. Return ONLY a JSON object (no markdown, no explanation) with these exact keys:\n"
        "- event_type: string describing the event\n"
        "- severity: one of [minor, moderate, severe]\n"
        "- time_horizon: one of [days, quarters, permanent]\n"
        "- company_specific: boolean\n"
        "- has_numbers: boolean\n"
        "- credibility: one of [low, medium, high]\n"
        "- novelty_score: float between 0 and 1\n\n"
        "Headlines:\n"
    ).format(sym=symbol)
    items = [f"- {h.title}" for h in headlines[:6]]
    prompt += "\n".join(items)
    prompt += "\n\nReturn only the JSON object, nothing else."
    return prompt


def extract_first_json_block(text: str) -> Optional[dict]:
    """Best-effort extraction of the first JSON object from a blob of text."""
    if not text:
        return None
    
    # Try direct parse first
    try:
        return _json.loads(text.strip())
    except Exception:
        pass
    
    # Remove markdown code blocks if present
    import re
    # Pattern for ```json ... ``` or ``` ... ```
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        try:
            return _json.loads(matches[0].strip())
        except Exception:
            pass
    
    # Try to find JSON object boundaries
    try:
        # Find all potential JSON objects
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        potential_jsons = re.findall(json_pattern, text)
        
        for candidate in potential_jsons:
            try:
                obj = _json.loads(candidate)
                # Verify it has expected fields for news classification
                if isinstance(obj, dict) and any(key in obj for key in ['event_type', 'severity', 'time_horizon']):
                    return obj
            except Exception:
                continue
    except Exception:
        pass
    
    # Last resort: find first { and last }
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            return _json.loads(candidate)
    except Exception:
        pass
    
    return None


def blend_scores(oversold_score: Optional[float], news_score: Optional[float], weight: float = 0.5) -> Optional[float]:
    """Blend price-based oversold score with news_score ∈ [0,1].

    Strong reversion news (>0.5) makes the score more negative; sticky bad news (<0.5)
    makes it less negative, achieved by shifting by (0.5 - news) scaled by weight.
    """
    try:
        if oversold_score is None:
            return None
        if not isinstance(news_score, (int, float)):
            return oversold_score
        ns = max(0.0, min(1.0, float(news_score)))
        w = max(0.0, min(2.0, float(weight)))
        return float(oversold_score) + (w * (0.5 - ns))
    except Exception:
        return oversold_score


def compute_news_score(symbol: str) -> Optional[float]:
    """Compute a 0..1 news_score indicating likelihood of short-term mean reversion.

    Heuristics:
    - severity: minor=0.8, moderate=0.5, severe=0.2
    - time_horizon: days=0.8, quarters=0.4, permanent=0.1
    - credibility: low=0.8, medium=0.5, high=0.2
    - company_specific: if False, +0.05
    - has_numbers: if True, -0.1
    - novelty_score in [0,1]: add (1 - novelty)*0.2
    """
    try:
        # 1) Check SQL cache (1-day TTL)
        try:
            engine = get_engine()
            maker = sessionmaker(bind=engine, autoflush=False, autocommit=False)
            with maker() as db:
                repo = FeaturesRepository(db)
                row = repo.latest_for_symbol(symbol, feature_set=NEWS_SCORE_FEATURE_SET)
                if row and getattr(row, "timestamp", None):
                    # Handle both timezone-aware and naive timestamps
                    if row.timestamp.tzinfo is None:
                        # Assume UTC for naive timestamps from database
                        age = (datetime.now(timezone.utc) - row.timestamp.replace(tzinfo=timezone.utc)).total_seconds()
                    else:
                        age = (datetime.now(timezone.utc) - row.timestamp).total_seconds()
                    if age < NEWS_SCORE_TTL_SECONDS:
                        try:
                            payload = _json.loads(row.features_json or "{}")
                            cached = payload.get("news_score")
                            if isinstance(cached, (int, float)):
                                print(f"💾 news_score cache hit for {symbol}: {cached}")
                                return float(cached)
                        except Exception as e:
                            print(f"❌ Failed to parse cached news_score for {symbol}: {e}")
        except Exception as e:
            print(f"❌ Failed to check news_score cache for {symbol}: {e}")

        # 2) Fresh computation
        t0 = datetime.now(timezone.utc)
        headlines = _search_news_for_symbol(symbol, days=3, max_results=6)
        if not headlines:
            print(f"⚠️ No headlines found for {symbol}")
            return None
        t1 = datetime.now(timezone.utc)
        prompt = build_news_classification_prompt(symbol, headlines)
        raw = _call_openai(prompt)
        t2 = datetime.now(timezone.utc)
        data = extract_first_json_block(raw)
        if not isinstance(data, dict):
            print(f"❌ Failed to extract JSON from LLM response for {symbol}")
            print(f"   Raw response: {raw[:200]}...")
            return None

        severity = (data.get("severity") or "").strip().lower()
        horizon = (data.get("time_horizon") or "").strip().lower()
        credibility = (data.get("credibility") or "").strip().lower()
        company_specific = bool(data.get("company_specific"))
        has_numbers = bool(data.get("has_numbers"))
        novelty = data.get("novelty_score")

        sev_map = {"minor": 0.8, "moderate": 0.5, "severe": 0.2}
        hor_map = {"days": 0.8, "quarters": 0.4, "permanent": 0.1}
        cred_map = {"low": 0.8, "medium": 0.5, "high": 0.2}

        sev_score = sev_map.get(severity)
        hor_score = hor_map.get(horizon)
        cred_score = cred_map.get(credibility)

        components: List[float] = []
        if isinstance(sev_score, (int, float)):
            components.append(float(sev_score))
        if isinstance(hor_score, (int, float)):
            components.append(float(hor_score))
        if isinstance(cred_score, (int, float)):
            components.append(float(cred_score))
        if not components:
            return None

        base = sum(components) / len(components)
        if not company_specific:
            base += 0.05
        if has_numbers:
            base -= 0.1
        if isinstance(novelty, (int, float)):
            nv = max(0.0, min(1.0, float(novelty)))
            base += (1.0 - nv) * 0.2

        base = max(0.0, min(1.0, base))
        score_out = round(base, 3)

        # 3) Persist to SQL cache
        try:
            engine = get_engine()
            maker = sessionmaker(bind=engine, autoflush=False, autocommit=False)
            with maker() as db:
                repo = FeaturesRepository(db)
                payload = {
                    "symbol": symbol,
                    "news_score": score_out,
                    "llm": data,
                    "headlines": [h.model_dump() for h in headlines],
                    "timings_ms": {
                        "ddg": int((t1 - t0).total_seconds() * 1000),
                        "llm": int((t2 - t1).total_seconds() * 1000),
                    },
                }
                repo.add_features(symbol=symbol, feature_set=NEWS_SCORE_FEATURE_SET, features_json=_json.dumps(payload))
                print(f"💾 cached news_score for {symbol}: {score_out} (ddg={(t1-t0).total_seconds():.2f}s, llm={(t2-t1).total_seconds():.2f}s)")
        except Exception as e:
            print(f"⚠️ Failed to cache news_score for {symbol}: {e}")

        return score_out
    except Exception as e:
        print(f"❌ compute_news_score failed for {symbol}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


