"""
Persistent response cache for benchmark runs.

Caches LLM responses and NLI scores so re-runs with different
thresholds or aggregation strategies cost $0.

Storage: SQLite file at benchmarks/cache/responses.db
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"
DB_PATH = CACHE_DIR / "responses.db"


def _get_db() -> sqlite3.Connection:
    CACHE_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            key TEXT PRIMARY KEY,
            model TEXT,
            prompt_hash TEXT,
            response TEXT,
            tokens_used INTEGER,
            created_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nli_cache (
            key TEXT PRIMARY KEY,
            premise_hash TEXT,
            hypothesis_hash TEXT,
            scores TEXT,
            created_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embedding_cache (
            key TEXT PRIMARY KEY,
            text_hash TEXT,
            model TEXT,
            vector TEXT,
            created_at REAL
        )
    """)
    conn.commit()
    return conn


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


class ResponseCache:
    """Persistent cache for LLM responses, NLI scores, and embeddings."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._conn: sqlite3.Connection | None = None
        self.hits = 0
        self.misses = 0

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = _get_db()
        return self._conn

    def get_llm(self, model: str, prompt: str) -> str | None:
        if not self.enabled:
            return None
        key = _hash(f"{model}:{prompt}")
        row = self.conn.execute(
            "SELECT response FROM llm_cache WHERE key = ?", (key,)
        ).fetchone()
        if row:
            self.hits += 1
            return row[0]
        self.misses += 1
        return None

    def put_llm(self, model: str, prompt: str, response: str, tokens: int = 0) -> None:
        if not self.enabled:
            return
        key = _hash(f"{model}:{prompt}")
        self.conn.execute(
            "INSERT OR REPLACE INTO llm_cache (key, model, prompt_hash, response, tokens_used, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (key, model, _hash(prompt), response, tokens, time.time()),
        )
        self.conn.commit()

    def get_nli(self, premise: str, hypothesis: str, model: str = "") -> list[float] | None:
        if not self.enabled:
            return None
        key = _hash(f"nli:{model}:{premise}:{hypothesis}")
        row = self.conn.execute(
            "SELECT scores FROM nli_cache WHERE key = ?", (key,)
        ).fetchone()
        if row:
            self.hits += 1
            return json.loads(row[0])
        self.misses += 1
        return None

    def put_nli(self, premise: str, hypothesis: str, scores: list[float], model: str = "") -> None:
        if not self.enabled:
            return
        key = _hash(f"nli:{model}:{premise}:{hypothesis}")
        self.conn.execute(
            "INSERT OR REPLACE INTO nli_cache (key, premise_hash, hypothesis_hash, scores, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (key, _hash(premise), _hash(hypothesis), json.dumps(scores), time.time()),
        )
        self.conn.commit()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
        }

    def clear(self) -> None:
        self.conn.execute("DELETE FROM llm_cache")
        self.conn.execute("DELETE FROM nli_cache")
        self.conn.execute("DELETE FROM embedding_cache")
        self.conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
