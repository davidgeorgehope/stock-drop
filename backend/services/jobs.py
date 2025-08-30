"""Minimal background job manager for async tasks.

This module provides a lightweight, in-memory job queue suitable for
short-lived background work (seconds to a few minutes). Jobs are stored
with a TTL and cleaned up periodically.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional


class Job:
    def __init__(self, job_type: str):
        self.id: str = str(uuid.uuid4())
        self.type: str = job_type
        self.status: str = "queued"  # queued | running | completed | failed
        self.created_at: float = time.time()
        self.updated_at: float = self.created_at
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.progress: float = 0.0
        self.expires_at: float = self.created_at + 3600  # 1 hour TTL by default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "expires_at": self.expires_at,
        }


class JobStore:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._cleanup_started = False

    def submit(self, job_type: str, func: Callable[[], Any]) -> str:
        job = Job(job_type)
        with self._lock:
            self._jobs[job.id] = job

        def runner():
            try:
                job.status = "running"
                job.updated_at = time.time()
                result = func()
                job.result = result
                job.status = "completed"
                job.progress = 1.0
                job.updated_at = time.time()
                # Extend TTL after completion so clients can retrieve
                job.expires_at = time.time() + 3600
            except Exception as e:
                job.error = f"{type(e).__name__}: {e}"
                job.status = "failed"
                job.updated_at = time.time()

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        return job.id

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def cleanup_once(self):
        now = time.time()
        with self._lock:
            to_delete = [jid for jid, j in self._jobs.items() if j.expires_at <= now]
            for jid in to_delete:
                try:
                    del self._jobs[jid]
                except Exception:
                    pass

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


# Global store used by the application
JOB_STORE = JobStore()


def submit_job(job_type: str, func: Callable[[], Any]) -> str:
    return JOB_STORE.submit(job_type, func)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    return JOB_STORE.get(job_id)


def start_cleanup_daemon():
    JOB_STORE.start_cleanup_daemon()


