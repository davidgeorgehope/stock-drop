from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select, and_, desc
from sqlalchemy.orm import Session, sessionmaker

from ..connection import get_engine
from ..models import InterestingLoserDB


class LosersRepository:
    """Repository for curated/interesting losers persisted in SQLite.

    We store each daily run as a batch (batch_id = ISO date) so the API can
    fetch the latest ranked list reliably across processes.
    """

    @staticmethod
    def _get_session() -> Session:
        engine = get_engine()
        Maker = sessionmaker(bind=engine)
        return Maker()

    @staticmethod
    def save_ranked(batch_id: str, losers: List[dict], session_label: str = "EOD") -> int:
        """Save a ranked losers list for a batch. Replaces any existing batch rows.

        losers items should contain: symbol, price, change, change_percent, volume, reason.
        """
        session = LosersRepository._get_session()
        try:
            # delete existing batch
            session.query(InterestingLoserDB).filter(InterestingLoserDB.batch_id == batch_id).delete()

            created = 0
            for idx, it in enumerate(losers):
                row = InterestingLoserDB(
                    batch_id=batch_id,
                    symbol=(it.get("symbol") or "").upper(),
                    price=it.get("price"),
                    change=it.get("change"),
                    change_percent=it.get("change_percent"),
                    volume=it.get("volume"),
                    reason=it.get("reason"),
                    rank=idx + 1,
                    session=session_label,
                )
                session.add(row)
                created += 1
            session.commit()
            return created
        finally:
            session.close()

    @staticmethod
    def latest_batch_id() -> Optional[str]:
        session = LosersRepository._get_session()
        try:
            stmt = (
                select(InterestingLoserDB.batch_id)
                .order_by(desc(InterestingLoserDB.created_at))
                .limit(1)
            )
            row = session.execute(stmt).first()
            return row[0] if row else None
        finally:
            session.close()

    @staticmethod
    def get_ranked(batch_id: Optional[str] = None, limit: int = 15) -> List[InterestingLoserDB]:
        session = LosersRepository._get_session()
        try:
            if batch_id is None:
                batch_id = LosersRepository.latest_batch_id()
                if not batch_id:
                    return []
            stmt = (
                select(InterestingLoserDB)
                .where(InterestingLoserDB.batch_id == batch_id)
                .order_by(InterestingLoserDB.rank.asc())
                .limit(limit)
            )
            return list(session.execute(stmt).scalars())
        finally:
            session.close()


