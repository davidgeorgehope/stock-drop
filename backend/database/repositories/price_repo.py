from datetime import datetime, timedelta
from typing import List, Optional, Dict
from sqlalchemy import select, and_
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import IntegrityError

from ..models import PriceData
from ..connection import get_engine


class PriceRepository:
    """Repository for managing price data in SQLite."""
    
    @staticmethod
    def _get_session():
        """Create a new SQLAlchemy session."""
        engine = get_engine()
        Session = sessionmaker(bind=engine)
        return Session()
    
    @staticmethod
    def save_polygon_grouped_data(date_str: str, grouped_data: List[dict]) -> int:
        """Save Polygon grouped data to the database.
        
        Returns the number of records saved.
        """
        session = PriceRepository._get_session()
        try:
            records_saved = 0
            
            for stock_data in grouped_data:
                symbol = stock_data.get("T", "").upper()
                if not symbol:
                    continue
                    
                # Convert timestamp to datetime
                timestamp = datetime.strptime(date_str, "%Y-%m-%d")
                
                # Check if record exists
                existing = session.query(PriceData).filter(
                    and_(
                        PriceData.symbol == symbol,
                        PriceData.timestamp == timestamp,
                        PriceData.source == "polygon_grouped"
                    )
                ).first()
                
                if existing:
                    # Update existing record
                    existing.open = stock_data.get("o")
                    existing.high = stock_data.get("h")
                    existing.low = stock_data.get("l")
                    existing.close = stock_data.get("c")
                    existing.volume = stock_data.get("v")
                else:
                    # Create new record
                    price_data = PriceData(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=stock_data.get("o"),
                        high=stock_data.get("h"),
                        low=stock_data.get("l"),
                        close=stock_data.get("c"),
                        volume=stock_data.get("v"),
                        source="polygon_grouped"
                    )
                    session.add(price_data)
                
                records_saved += 1
            
            session.commit()
            return records_saved
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
