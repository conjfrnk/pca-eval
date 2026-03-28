"""Tests for benchmarks.scifact module.

Tests SciFact label mapping, corpus loading, and claim parsing
without requiring NLI models or real dataset downloads.
"""

import json

import pytest

from benchmarks.scifact import SciFact

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------


class TestMapNLILabel:
    """Tests for SciFact.map_nli_label."""

    def setup_method(self):
        self.suite = SciFact()

    def test_supports_passes_through(self):
        assert self.suite.map_nli_label("SUPPORTS") == "SUPPORTS"

    def test_refutes_passes_through(self):
        assert self.suite.map_nli_label("REFUTES") == "REFUTES"

    def test_nei_passes_through(self):
        assert self.suite.map_nli_label("NOT_ENOUGH_INFO") == "NOT_ENOUGH_INFO"

    def test_unknown_maps_to_nei(self):
        assert self.suite.map_nli_label("UNKNOWN") == "NOT_ENOUGH_INFO"


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


class TestLoadCorpus:
    """Tests for SciFact._load_corpus."""

    def test_load_corpus_jsonl(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "scifact"
        data_dir.mkdir()

        corpus = [
            {"doc_id": "1", "title": "Paper 1", "abstract": ["Sent 1.", "Sent 2."]},
            {"doc_id": "2", "title": "Paper 2", "abstract": ["Single abstract."]},
        ]
        corpus_file = data_dir / "corpus.jsonl"
        with open(corpus_file, "w") as f:
            for doc in corpus:
                f.write(json.dumps(doc) + "\n")

        monkeypatch.setattr("benchmarks.scifact.BenchmarkSuite.data_dir", property(lambda self: data_dir))
        suite = SciFact()
        result = suite._load_corpus()
        assert len(result) == 2
        assert result["1"]["title"] == "Paper 1"
        assert len(result["1"]["abstract"]) == 2

    def test_corpus_caching(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "scifact"
        data_dir.mkdir()
        corpus_file = data_dir / "corpus.jsonl"
        corpus_file.write_text(json.dumps({"doc_id": "1", "title": "T", "abstract": ["A"]}) + "\n")

        monkeypatch.setattr("benchmarks.scifact.BenchmarkSuite.data_dir", property(lambda self: data_dir))
        suite = SciFact()
        first = suite._load_corpus()
        second = suite._load_corpus()
        assert first is second  # Same object, cached

    def test_corpus_string_abstract(self, tmp_path, monkeypatch):
        """Handle abstract as a single string instead of list."""
        data_dir = tmp_path / "scifact"
        data_dir.mkdir()
        corpus_file = data_dir / "corpus.jsonl"
        corpus_file.write_text(json.dumps({"doc_id": "1", "title": "T", "abstract": "Single string."}) + "\n")

        monkeypatch.setattr("benchmarks.scifact.BenchmarkSuite.data_dir", property(lambda self: data_dir))
        suite = SciFact()
        result = suite._load_corpus()
        assert result["1"]["abstract"] == ["Single string."]

    def test_corpus_missing_raises(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "scifact"
        data_dir.mkdir()
        monkeypatch.setattr("benchmarks.scifact.BenchmarkSuite.data_dir", property(lambda self: data_dir))
        suite = SciFact()
        with pytest.raises(FileNotFoundError, match="SciFact corpus not found"):
            suite._load_corpus()


# ---------------------------------------------------------------------------
# Claims loading
# ---------------------------------------------------------------------------


class TestLoad:
    """Tests for SciFact.load."""

    def _setup_scifact_data(self, data_dir):
        """Create minimal SciFact dataset in data_dir."""
        corpus = [
            {"doc_id": "100", "title": "Study A", "abstract": ["Cells grow.", "Mitosis occurs.", "Division ends."]},
            {"doc_id": "200", "title": "Study B", "abstract": ["Water is wet."]},
        ]
        corpus_file = data_dir / "corpus.jsonl"
        with open(corpus_file, "w") as f:
            for doc in corpus:
                f.write(json.dumps(doc) + "\n")

        claims = [
            {
                "id": 1,
                "claim": "Cells undergo mitosis.",
                "evidence": {
                    "100": [{"sentences": [1], "label": "SUPPORT"}],
                },
            },
            {
                "id": 2,
                "claim": "Water is dry.",
                "evidence": {
                    "200": [{"sentences": [0], "label": "CONTRADICT"}],
                },
            },
            {
                "id": 3,
                "claim": "Aliens exist.",
                "evidence": {},
            },
        ]
        claims_file = data_dir / "claims_dev.jsonl"
        with open(claims_file, "w") as f:
            for claim in claims:
                f.write(json.dumps(claim) + "\n")

    def test_load_dev_examples(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "scifact"
        data_dir.mkdir()
        self._setup_scifact_data(data_dir)
        monkeypatch.setattr("benchmarks.scifact.BenchmarkSuite.data_dir", property(lambda self: data_dir))

        suite = SciFact()
        examples = suite.load(split="dev")

        # 2 claims with evidence + 1 NEI = 3 total
        assert len(examples) == 3

    def test_support_label_mapping(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "scifact"
        data_dir.mkdir()
        self._setup_scifact_data(data_dir)
        monkeypatch.setattr("benchmarks.scifact.BenchmarkSuite.data_dir", property(lambda self: data_dir))

        suite = SciFact()
        examples = suite.load(split="dev")

        support_examples = [e for e in examples if e.gold_label == "SUPPORTS"]
        assert len(support_examples) == 1
        assert support_examples[0].claim_or_query == "Cells undergo mitosis."

    def test_contradict_label_mapping(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "scifact"
        data_dir.mkdir()
        self._setup_scifact_data(data_dir)
        monkeypatch.setattr("benchmarks.scifact.BenchmarkSuite.data_dir", property(lambda self: data_dir))

        suite = SciFact()
        examples = suite.load(split="dev")

        refutes = [e for e in examples if e.gold_label == "REFUTES"]
        assert len(refutes) == 1
        assert refutes[0].claim_or_query == "Water is dry."

    def test_nei_no_evidence(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "scifact"
        data_dir.mkdir()
        self._setup_scifact_data(data_dir)
        monkeypatch.setattr("benchmarks.scifact.BenchmarkSuite.data_dir", property(lambda self: data_dir))

        suite = SciFact()
        examples = suite.load(split="dev")

        nei = [e for e in examples if e.gold_label == "NOT_ENOUGH_INFO"]
        assert len(nei) == 1
        assert nei[0].evidence_sentences == []

    def test_evidence_sentence_extraction(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "scifact"
        data_dir.mkdir()
        self._setup_scifact_data(data_dir)
        monkeypatch.setattr("benchmarks.scifact.BenchmarkSuite.data_dir", property(lambda self: data_dir))

        suite = SciFact()
        examples = suite.load(split="dev")

        support = [e for e in examples if e.gold_label == "SUPPORTS"][0]
        assert support.evidence_sentences == ["Mitosis occurs."]
        assert support.evidence_sentence_indices == [1]

    def test_metadata_contains_abstract(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "scifact"
        data_dir.mkdir()
        self._setup_scifact_data(data_dir)
        monkeypatch.setattr("benchmarks.scifact.BenchmarkSuite.data_dir", property(lambda self: data_dir))

        suite = SciFact()
        examples = suite.load(split="dev")

        support = [e for e in examples if e.gold_label == "SUPPORTS"][0]
        assert "all_abstract_sentences" in support.metadata
        assert len(support.metadata["all_abstract_sentences"]) == 3

    def test_missing_claims_file_raises(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "scifact"
        data_dir.mkdir()
        # Create corpus but no claims
        corpus_file = data_dir / "corpus.jsonl"
        corpus_file.write_text(json.dumps({"doc_id": "1", "title": "T", "abstract": ["A"]}) + "\n")

        monkeypatch.setattr("benchmarks.scifact.BenchmarkSuite.data_dir", property(lambda self: data_dir))
        suite = SciFact()
        with pytest.raises(FileNotFoundError, match="SciFact claims not found"):
            suite.load(split="dev")
