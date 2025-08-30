from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class OversoldMetrics:
    return_1d: Optional[float]
    return_3d: Optional[float]
    gap_pct: Optional[float]
    volume_ratio_20d: Optional[float]
    true_range_pct: Optional[float]
    zscore_1d: Optional[float]
    zscore_close: Optional[float]
    oversold_score: Optional[float]


def _pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return (a - b) / b


def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _std(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return var ** 0.5


def compute_oversold_metrics(history: List[dict]) -> OversoldMetrics:
    """Compute simple, data-lite oversold metrics from daily bars.

    history: list of dicts with keys: open, high, low, close, volume (ascending by date)
    """
    if not history or len(history) < 5:
        return OversoldMetrics(None, None, None, None, None, None, None, None)

    closes = [h.get("close") for h in history]
    volumes = [h.get("volume") for h in history]
    opens = [h.get("open") for h in history]
    highs = [h.get("high") for h in history]
    lows = [h.get("low") for h in history]

    # 1-day and 3-day returns
    r1 = _pct(closes[-1], closes[-2])
    r3 = _pct(closes[-1], closes[-4]) if len(closes) >= 4 else None

    # Gap percent vs previous close
    gap = _pct(opens[-1], closes[-2])

    # Volume ratio vs 20-day average
    vol20 = _mean([v for v in volumes[-20:] if v is not None])
    vol_ratio = (volumes[-1] / vol20) if vol20 and volumes[-1] is not None else None

    # True range percentage of today's open
    if opens[-1] and highs[-1] is not None and lows[-1] is not None:
        tr_pct = (highs[-1] - lows[-1]) / opens[-1] if opens[-1] else None
    else:
        tr_pct = None

    # Z-score of last close vs last 20 closes
    last20 = [c for c in closes[-20:] if c is not None]
    mu20 = _mean(last20) if last20 else None
    sd20 = _std(last20) if last20 else None
    z_close = ((closes[-1] - mu20) / sd20) if (mu20 is not None and sd20 and sd20 > 0) else None

    # Simple composite oversold score (more negative is more oversold)
    components = []
    if r1 is not None:
        components.append(min(0.0, r1) * 4.0)  # weight 1d drop heavier
    if r3 is not None:
        components.append(min(0.0, r3) * 2.0)
    if gap is not None:
        components.append(min(0.0, gap) * 3.0)
    if z_close is not None:
        components.append(min(0.0, z_close / 3.0))  # scale z
    if vol_ratio is not None and vol_ratio > 1:
        components.append(min(0.0, -0.02 * (vol_ratio - 1)))  # bigger volume strengthens oversold

    oversold = sum(components) if components else None

    return OversoldMetrics(
        return_1d=r1,
        return_3d=r3,
        gap_pct=gap,
        volume_ratio_20d=vol_ratio,
        true_range_pct=tr_pct,
        zscore_1d=None,
        zscore_close=z_close,
        oversold_score=oversold,
    )


