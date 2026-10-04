"""Кеш ответов в SQLite. Ключ: SHA-256 от нормализованного вопроса + фильтр + top_k."""
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from . import config


class AnswerCache:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = str(db_path or config.CACHE_DB)
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY, query TEXT NOT NULL, answer TEXT NOT NULL,
                    sources TEXT, created_at TEXT)"""
            )

    def _conn(self):
        return sqlite3.connect(self.db_path)

    @staticmethod
    def key(query: str, source: str | None = None, top_k: int = config.TOP_K) -> str:
        norm = " ".join(query.lower().split())
        return hashlib.sha256(f"{norm}|{source or ''}|{top_k}".encode()).hexdigest()

    def get(self, query: str, source: str | None = None, top_k: int = config.TOP_K):
        with self._conn() as c:
            row = c.execute("SELECT answer, sources, created_at FROM cache WHERE key=?",
                            (self.key(query, source, top_k),)).fetchone()
        if not row:
            return None
        return {"answer": row[0], "sources": json.loads(row[1] or "[]"), "created_at": row[2]}

    def set(self, query: str, answer: str, sources: list[dict], source: str | None = None,
            top_k: int = config.TOP_K) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)",
                      (self.key(query, source, top_k), query, answer,
                       json.dumps(sources, ensure_ascii=False), datetime.now().isoformat()))

    def clear(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM cache")

    def stats(self) -> dict:
        with self._conn() as c:
            n, first, last = c.execute("SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM cache").fetchone()
        return {"entries": n, "first": first, "last": last}
