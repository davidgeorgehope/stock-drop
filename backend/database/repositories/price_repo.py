from datetime import datetime, timedelta
from typing import List, Optional, Dict
from sqlalchemy import select, and_
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from ..models import PriceData
from ..connection import get_engine


class PriceRepository:
    """Repository for managing price data in SQLite."""
    
    @staticmethod
    def _get_session():
        """Create a new SQLAlchemy session."""
        engine = get_engine()
        Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
        return Session()
    
    @staticmethod
    def save_polygon_grouped_data(date_str: str, grouped_data: List[dict]) -> int:
        """Idempotently upsert Polygon grouped data into SQLite.
        
        Uses SQLite ON CONFLICT to avoid duplicate insert errors when multiple
        threads/processes attempt to save the same trading day concurrently.
        Returns the number of rows processed (inserts + updates best-effort).
        """
        session = PriceRepository._get_session()
        try:
            timestamp = datetime.strptime(date_str, "%Y-%m-%d")

            # Prepare rows (filter out empty symbols)
            values: List[Dict] = []
            for stock_data in grouped_data:
                symbol = (stock_data.get("T") or "").strip().upper()
                if not symbol:
                    continue
                values.append({
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "open": stock_data.get("o"),
                    "high": stock_data.get("h"),
                    "low": stock_data.get("l"),
                    "close": stock_data.get("c"),
                    "volume": stock_data.get("v"),
                    "source": "polygon_grouped",
                })

            if not values:
                return 0

            # Batch upserts to keep statement sizes reasonable
            batch_size = 1000
            total_processed = 0
            for i in range(0, len(values), batch_size):
                batch = values[i:i + batch_size]
                stmt = sqlite_insert(PriceData).values(batch)
                update_cols = {c: stmt.excluded[c] for c in [
                    "open", "high", "low", "close", "volume"
                ]}
                stmt = stmt.on_conflict_do_update(
                    index_elements=[PriceData.symbol, PriceData.timestamp, PriceData.source],
                    set_=update_cols,
                )
                session.execute(stmt)
                total_processed += len(batch)

            session.commit()
            return total_processed
        finally:
            session.close()
    
    @staticmethod
    def get_price_history(symbol: str, days: int = 30, source: Optional[str] = None) -> List[dict]:
        """Fetch price history for a symbol from the database.
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            source: Optional source filter (e.g., 'polygon_grouped', 'stooq')
            
        Returns:
            List of price data dicts sorted by date ascending
        """
        session = PriceRepository._get_session()
        try:
            # Calculate date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            # Build query
            query = select(PriceData).where(
                and_(
                    PriceData.symbol == symbol.upper(),
                    PriceData.timestamp >= start_date,
                    PriceData.timestamp <= end_date
                )
            )
            
            if source:
                query = query.where(PriceData.source == source)
            
            # Order by timestamp ascending
            query = query.order_by(PriceData.timestamp)
            
            # Execute and convert to dicts
            results = session.execute(query).scalars().all()
            
            return [
                {
                    "date": r.timestamp.strftime("%Y-%m-%d"),
                    "date_iso": r.timestamp.isoformat() + "Z",
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                    "source": r.source
                }
                for r in results
            ]
        finally:
            session.close()
    
    @staticmethod
    def get_cached_dates(source: str = "polygon_grouped", days_back: int = 30) -> List[str]:
        """Get list of dates that have cached data for the given source.
        
        Returns:
            List of date strings (YYYY-MM-DD) that have data
        """
        session = PriceRepository._get_session()
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            # Get distinct dates
            query = (
                select(PriceData.timestamp)
                .where(
                    and_(
                        PriceData.source == source,
                        PriceData.timestamp >= start_date,
                        PriceData.timestamp <= end_date
                    )
                )
                .distinct()
                .order_by(PriceData.timestamp.desc())
            )
            
            results = session.execute(query).scalars().all()
            
            return [dt.strftime("%Y-%m-%d") for dt in results]
        finally:
            session.close()
    
    @staticmethod
    def has_data_for_date(date_str: str, source: str = "polygon_grouped") -> bool:
        """Check if we have data for a specific date and source."""
        session = PriceRepository._get_session()
        try:
            timestamp = datetime.strptime(date_str, "%Y-%m-%d")
            
            query = select(PriceData).where(
                and_(
                    PriceData.source == source,
                    PriceData.timestamp == timestamp
                )
            ).limit(1)
            
            result = session.execute(query).first()
            return result is not None
        finally:
            session.close()
    
    @staticmethod
    def get_grouped_data_for_date(date_str: str) -> List[dict]:
        """Retrieve all stocks for a given date in Polygon grouped format.
        
        Returns:
            List of dicts in Polygon grouped format with keys: T, o, h, l, c, v
        """
        session = PriceRepository._get_session()
        try:
            timestamp = datetime.strptime(date_str, "%Y-%m-%d")
            
            query = select(PriceData).where(
                and_(
                    PriceData.source == "polygon_grouped",
                    PriceData.timestamp == timestamp
                )
            ).order_by(PriceData.symbol)
            
            results = session.execute(query).scalars().all()
            
            # Convert to Polygon grouped format
            grouped_data = []
            for r in results:
                grouped_data.append({
                    "T": r.symbol,
                    "o": r.open,
                    "h": r.high,
                    "l": r.low,
                    "c": r.close,
                    "v": r.volume
                })
            
            return grouped_data
        finally:
            session.close()
