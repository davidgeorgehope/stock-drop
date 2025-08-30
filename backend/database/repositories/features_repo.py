from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from ..models import ComputedFeatures


class FeaturesRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_features(self, symbol: str, feature_set: str, features_json: str, ts: datetime | None = None) -> None:
        row = ComputedFeatures(
            symbol=symbol,
            timestamp=(ts or datetime.now(timezone.utc)),
            feature_set=feature_set,
            features_json=features_json,
        )
        self.session.add(row)
        self.session.commit()

    def list_recent(self, feature_set: str, lookback_minutes: int = 60, limit: int = 100) -> List[ComputedFeatures]:
        since = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
        stmt = (
            select(ComputedFeatures)
            .where(ComputedFeatures.feature_set == feature_set)
            .where(ComputedFeatures.timestamp >= since)
            .order_by(ComputedFeatures.timestamp.desc())
            .limit(limit * 10)  # fetch extra for client-side sort
        )
        return list(self.session.execute(stmt).scalars())

    def latest_for_symbol(self, symbol: str, feature_set: str) -> Optional[ComputedFeatures]:
        stmt = (
            select(ComputedFeatures)
            .where(ComputedFeatures.symbol == symbol)
            .where(ComputedFeatures.feature_set == feature_set)
            .order_by(ComputedFeatures.timestamp.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalars().first()


