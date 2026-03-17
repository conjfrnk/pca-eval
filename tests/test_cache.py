"""
Tests for the benchmark response cache.

Tests the ResponseCache class which provides persistent
caching for LLM responses, NLI scores, and embeddings:
- Cache hit/miss tracking
- LLM response caching
- NLI score caching
- Cache statistics
- Disabled cache behavior
- Cache clearing
"""

import tempfile

from benchmarks.cache import ResponseCache, _hash

# =============================================================================
# Hash Function Tests
# =============================================================================


class TestHash:
    def test_deterministic(self):
        assert _hash("hello") == _hash("hello")

    def test_different_inputs_different_hashes(self):
        assert _hash("hello") != _hash("world")

    def test_returns_32_chars(self):
        assert len(_hash("test")) == 32

    def test_empty_string(self):
        h = _hash("")
        assert len(h) == 32
        assert isinstance(h, str)


# =============================================================================
# Response Cache Tests
# =============================================================================


class TestResponseCache:
    def setup_method(self):
        # Use a temp directory for the test database
        self.tmpdir = tempfile.mkdtemp()
        self.cache = ResponseCache(enabled=True)
        # Patch the DB to use a temp directory
        import sqlite3
        self.cache._conn = sqlite3.connect(":memory:")
        self.cache._conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                key TEXT PRIMARY KEY, model TEXT, prompt_hash TEXT,
                response TEXT, tokens_used INTEGER, created_at REAL
            )
        """)
        self.cache._conn.execute("""
            CREATE TABLE IF NOT EXISTS nli_cache (
                key TEXT PRIMARY KEY, premise_hash TEXT, hypothesis_hash TEXT,
                scores TEXT, created_at REAL
            )
        """)
        self.cache._conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                key TEXT PRIMARY KEY, text_hash TEXT, model TEXT,
                vector TEXT, created_at REAL
            )
        """)
        self.cache._conn.commit()

    def teardown_method(self):
        self.cache.close()

    def test_llm_cache_miss(self):
        result = self.cache.get_llm("gpt-4", "test prompt")
        assert result is None
        assert self.cache.misses == 1

    def test_llm_cache_put_and_get(self):
        self.cache.put_llm("gpt-4", "test prompt", "test response", tokens=100)
        result = self.cache.get_llm("gpt-4", "test prompt")
        assert result == "test response"
        assert self.cache.hits == 1

    def test_llm_cache_different_models(self):
        self.cache.put_llm("gpt-4", "prompt", "response4")
        self.cache.put_llm("claude", "prompt", "response_claude")
        assert self.cache.get_llm("gpt-4", "prompt") == "response4"
        assert self.cache.get_llm("claude", "prompt") == "response_claude"

    def test_nli_cache_miss(self):
        result = self.cache.get_nli("premise", "hypothesis")
        assert result is None
        assert self.cache.misses == 1

    def test_nli_cache_put_and_get(self):
        scores = [0.1, 0.8, 0.1]
        self.cache.put_nli("premise", "hypothesis", scores)
        result = self.cache.get_nli("premise", "hypothesis")
        assert result == scores
        assert self.cache.hits == 1

    def test_nli_cache_model_specific(self):
        self.cache.put_nli("p", "h", [0.1, 0.8, 0.1], model="model_a")
        self.cache.put_nli("p", "h", [0.3, 0.5, 0.2], model="model_b")
        result_a = self.cache.get_nli("p", "h", model="model_a")
        result_b = self.cache.get_nli("p", "h", model="model_b")
        assert result_a == [0.1, 0.8, 0.1]
        assert result_b == [0.3, 0.5, 0.2]

    def test_stats(self):
        self.cache.get_llm("m", "p1")  # miss
        self.cache.put_llm("m", "p1", "r1")
        self.cache.get_llm("m", "p1")  # hit
        self.cache.get_llm("m", "p2")  # miss
        stats = self.cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert abs(stats["hit_rate"] - 1/3) < 0.01

    def test_clear(self):
        self.cache.put_llm("m", "p", "r")
        self.cache.put_nli("p", "h", [0.5])
        self.cache.clear()
        assert self.cache.get_llm("m", "p") is None
        assert self.cache.get_nli("p", "h") is None

    def test_close_and_reconnect(self):
        self.cache.put_llm("m", "p", "r")
        # Close sets _conn to None
        self.cache._conn.close()
        self.cache._conn = None


# =============================================================================
# Disabled Cache Tests
# =============================================================================


class TestDisabledCache:
    def test_disabled_llm_returns_none(self):
        cache = ResponseCache(enabled=False)
        cache.put_llm("m", "p", "r")
        assert cache.get_llm("m", "p") is None

    def test_disabled_nli_returns_none(self):
        cache = ResponseCache(enabled=False)
        cache.put_nli("p", "h", [0.5])
        assert cache.get_nli("p", "h") is None

    def test_disabled_no_hit_tracking(self):
        cache = ResponseCache(enabled=False)
        cache.get_llm("m", "p")
        assert cache.hits == 0
        assert cache.misses == 0
