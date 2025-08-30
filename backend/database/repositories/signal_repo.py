from __future__ import annotations

from typing import List, Optional
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from sqlalchemy import select

from ..models import TradingSignal


class SignalRepository:
    def __init__(self, session: Session):
        self.session = session

    def list_active(self, limit: int = 20, min_confidence: Optional[float] = None) -> List[TradingSignal]:
        stmt = select(TradingSignal).where(TradingSignal.status == "active").order_by(TradingSignal.created_at.desc()).limit(limit)
        if min_confidence is not None:
            stmt = stmt.where(TradingSignal.confidence_score >= min_confidence)
        return list(self.session.execute(stmt).scalars())

    def get(self, signal_id: str) -> Optional[TradingSignal]:
        return self.session.get(TradingSignal, signal_id)

    def has_recent_active(self, symbol: str, minutes: int = 1440) -> bool:
        since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        stmt = (
            select(TradingSignal)
            .where(TradingSignal.symbol == symbol)
            .where(TradingSignal.status == "active")
            .where(TradingSignal.created_at >= since)
            .limit(1)
        )
        return self.session.execute(stmt).first() is not None

    def create(self, signal: TradingSignal) -> TradingSignal:
        self.session.add(signal)
        self.session.commit()
        return signal


