"""Minimal background job manager for async tasks.

This module provides a lightweight, in-memory job queue suitable for
short-lived background work (seconds to a few minutes). Jobs are stored
with a TTL and cleaned up periodically.
"""

from __future__ import annotations

import threading
import time
import uuid
import json
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, Optional

from sqlalchemy.orm import sessionmaker
from database.connection import get_engine
from database.models import JobRecord


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStore:
    """SQLite-backed job store so multiple workers share state.

    Execution still happens in a local thread; state changes are persisted.
    """

    def __init__(self):
        engine = get_engine()
        self._maker = sessionmaker(bind=engine, autoflush=False, autocommit=False)
        self._cleanup_started = False

    def _write(self, record: JobRecord):
        with self._maker() as db:
            try:
                db.merge(record)
                db.commit()
            except Exception:
                db.rollback()
                raise

    def submit(self, job_type: str, func: Callable[[], Any]) -> str:
        job_id = str(uuid.uuid4())
        now = _utcnow()
        rec = JobRecord(
            id=job_id,
            type=job_type,
            status="queued",
            created_at=now,
            updated_at=now,
            result_json=None,
            error=None,
            progress=0.0,
            expires_at=now + timedelta(hours=1),
        )
        self._write(rec)

        def runner():
            try:
                # mark running
                rec.status = "running"
                rec.updated_at = _utcnow()
                self._write(rec)

                result = func()
                rec.result_json = json.dumps(result) if result is not None else None
                rec.status = "completed"
                rec.progress = 1.0
                rec.updated_at = _utcnow()
                rec.expires_at = _utcnow() + timedelta(hours=1)
                self._write(rec)
            except Exception as e:
                rec.error = f"{type(e).__name__}: {e}"
                rec.status = "failed"
                rec.updated_at = _utcnow()
                self._write(rec)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        return job_id

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._maker() as db:
            row = db.get(JobRecord, job_id)
            if not row:
                return None
            return {
                "id": row.id,
                "type": row.type,
                "status": row.status,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "result": json.loads(row.result_json) if row.result_json else None,
                "error": row.error,
                "progress": row.progress or 0.0,
                "expires_at": row.expires_at.isoformat() if row.expires_at else None,
            }

    def cleanup_once(self):
        now = _utcnow()
        with self._maker() as db:
            try:
                db.query(JobRecord).filter(JobRecord.expires_at <= now).delete()
                db.commit()
            except Exception:
                db.rollback()

    def start_cleanup_daemon(self):
        if self._cleanup_started:
            return

        def loop():
            while True:
                try:
                    self.cleanup_once()
                except Exception:
                    pass
                time.sleep(30)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self._cleanup_started = True


JOB_STORE = JobStore()


def submit_job(job_type: str, func: Callable[[], Any]) -> str:
    return JOB_STORE.submit(job_type, func)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    return JOB_STORE.get(job_id)


def start_cleanup_daemon():
    JOB_STORE.start_cleanup_daemon()


