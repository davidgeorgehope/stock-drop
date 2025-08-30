import os
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """Return a process-wide SQLAlchemy Engine, initializing if needed."""
    global _engine
    if _engine is not None:
        return _engine

    # Default to backend/data/stockdrop.db relative to this backend package
    backend_dir = os.path.dirname(os.path.dirname(__file__))
    default_db_path = os.path.join(backend_dir, "data", "stockdrop.db")
    db_path = os.getenv("SQLITE_DB_PATH", default_db_path)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    uri = f"sqlite:///{db_path}"
    engine = create_engine(uri, connect_args={"check_same_thread": False, "timeout": 30})
    _engine = engine
    return engine


def init_db() -> None:
    """Apply connection pragmas and create schema if missing."""
    # Connection-level pragmas
    engine = get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
        conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
        conn.exec_driver_sql("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);")
        conn.exec_driver_sql("INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version','0');")

    # Create ORM tables if not present
    try:
        from .models import Base  # local import to avoid circulars at import time
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        # Fail soft; app can still run without full schema until features are used
        print(f"⚠️ DB schema creation skipped: {e}")


