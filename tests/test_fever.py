"""Tests for benchmarks.fever module.

Tests FEVER label mapping, data loading, evidence extraction,
and wiki sentence parsing without requiring full dataset downloads.
"""

import json

import pytest

from benchmarks.fever import FEVER

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------


class TestMapNLILabel:
    """Tests for FEVER.map_nli_label."""

    def setup_method(self):
        self.suite = FEVER()

    def test_supports_passes_through(self):
        assert self.suite.map_nli_label("SUPPORTS") == "SUPPORTS"

    def test_refutes_passes_through(self):
        assert self.suite.map_nli_label("REFUTES") == "REFUTES"

    def test_nei_passes_through(self):
        assert self.suite.map_nli_label("NOT_ENOUGH_INFO") == "NOT_ENOUGH_INFO"

    def test_unknown_maps_to_nei(self):
        assert self.suite.map_nli_label("UNKNOWN") == "NOT_ENOUGH_INFO"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


class TestLoad:
    """Tests for FEVER.load."""

    def _write_fever_data(self, data_dir, items, filename="shared_task_dev.jsonl"):
        data_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / filename
        with open(data_file, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    def test_load_supports_claim(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "fever"
        items = [
            {
                "id": 1,
                "claim": "The sun is a star.",
                "label": "SUPPORTS",
                "evidence": [[[0, 0, "Sun", 0]]],
            },
        ]
        self._write_fever_data(data_dir, items)
        monkeypatch.setattr("benchmarks.fever.BenchmarkSuite.data_dir", property(lambda self: data_dir))

        suite = FEVER()
        examples = suite.load(split="dev")
        assert len(examples) == 1
        assert examples[0].gold_label == "SUPPORTS"
        assert examples[0].claim_or_query == "The sun is a star."

    def test_load_refutes_claim(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "fever"
        items = [
            {
                "id": 2,
                "claim": "The sun is cold.",
                "label": "REFUTES",
                "evidence": [[[0, 0, "Sun", 0]]],
            },
        ]
        self._write_fever_data(data_dir, items)
        monkeypatch.setattr("benchmarks.fever.BenchmarkSuite.data_dir", property(lambda self: data_dir))

        suite = FEVER()
        examples = suite.load(split="dev")
        assert examples[0].gold_label == "REFUTES"

    def test_load_nei_claim(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "fever"
        items = [
            {
                "id": 3,
                "claim": "Aliens exist.",
                "label": "NOT ENOUGH INFO",
                "evidence": [],
            },
        ]
        self._write_fever_data(data_dir, items)
        monkeypatch.setattr("benchmarks.fever.BenchmarkSuite.data_dir", property(lambda self: data_dir))

        suite = FEVER()
        examples = suite.load(split="dev")
        assert examples[0].gold_label == "NOT_ENOUGH_INFO"

    def test_label_normalization_supported(self, tmp_path, monkeypatch):
        """'SUPPORTED' variant normalizes to 'SUPPORTS'."""
        data_dir = tmp_path / "fever"
        items = [{"id": 1, "claim": "C", "label": "SUPPORTED", "evidence": []}]
        self._write_fever_data(data_dir, items)
        monkeypatch.setattr("benchmarks.fever.BenchmarkSuite.data_dir", property(lambda self: data_dir))

        suite = FEVER()
        examples = suite.load(split="dev")
        assert examples[0].gold_label == "SUPPORTS"

    def test_label_normalization_refuted(self, tmp_path, monkeypatch):
        """'REFUTED' variant normalizes to 'REFUTES'."""
        data_dir = tmp_path / "fever"
        items = [{"id": 1, "claim": "C", "label": "REFUTED", "evidence": []}]
        self._write_fever_data(data_dir, items)
        monkeypatch.setattr("benchmarks.fever.BenchmarkSuite.data_dir", property(lambda self: data_dir))

        suite = FEVER()
        examples = suite.load(split="dev")
        assert examples[0].gold_label == "REFUTES"

    def test_evidence_page_extraction(self, tmp_path, monkeypatch):
        """Evidence items extract page titles and sentence indices."""
        data_dir = tmp_path / "fever"
        items = [
            {
                "id": 1,
                "claim": "C",
                "label": "SUPPORTS",
                "evidence": [
                    [[0, 0, "Page_A", 2], [0, 1, "Page_B", 0]],
                ],
            },
        ]
        self._write_fever_data(data_dir, items)
        monkeypatch.setattr("benchmarks.fever.BenchmarkSuite.data_dir", property(lambda self: data_dir))

        suite = FEVER()
        examples = suite.load(split="dev")
        assert "Page_A" in examples[0].metadata["evidence_pages"]
        assert "Page_B" in examples[0].metadata["evidence_pages"]
        assert 2 in examples[0].evidence_sentence_indices

    def test_evidence_text_from_wiki(self, tmp_path, monkeypatch):
        """Evidence text is extracted when wiki pages are available."""
        data_dir = tmp_path / "fever"

        # Create wiki pages
        wiki_dir = data_dir / "wiki-pages" / "wiki-pages"
        wiki_dir.mkdir(parents=True)
        wiki_file = wiki_dir / "wiki-001.jsonl"
        wiki_page = {
            "id": "Sun",
            "lines": "0\tThe Sun is a star.\n1\tIt is very hot.",
        }
        wiki_file.write_text(json.dumps(wiki_page) + "\n")

        items = [
            {
                "id": 1,
                "claim": "The Sun is a star.",
                "label": "SUPPORTS",
                "evidence": [[[0, 0, "Sun", 0]]],
            },
        ]
        self._write_fever_data(data_dir, items)
        monkeypatch.setattr("benchmarks.fever.BenchmarkSuite.data_dir", property(lambda self: data_dir))

        suite = FEVER()
        examples = suite.load(split="dev")
        assert len(examples[0].evidence_sentences) == 1
        assert examples[0].evidence_sentences[0] == "The Sun is a star."

    def test_nei_no_evidence_extracted(self, tmp_path, monkeypatch):
        """NEI claims don't have evidence extracted even with evidence annotations."""
        data_dir = tmp_path / "fever"
        items = [
            {
                "id": 1,
                "claim": "C",
                "label": "NOT ENOUGH INFO",
                "evidence": [[[0, 0, "Page", 0]]],
            },
        ]
        self._write_fever_data(data_dir, items)
        monkeypatch.setattr("benchmarks.fever.BenchmarkSuite.data_dir", property(lambda self: data_dir))

        suite = FEVER()
        examples = suite.load(split="dev")
        assert examples[0].evidence_sentences == []

    def test_missing_data_raises(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "fever"
        data_dir.mkdir()
        monkeypatch.setattr("benchmarks.fever.BenchmarkSuite.data_dir", property(lambda self: data_dir))
        suite = FEVER()
        with pytest.raises(FileNotFoundError):
            suite.load(split="dev")

    def test_hf_format_fallback(self, tmp_path, monkeypatch):
        """Falls back to dev_hf.jsonl format."""
        data_dir = tmp_path / "fever"
        items = [{"id": 1, "claim": "C", "label": "SUPPORTS", "evidence": []}]
        self._write_fever_data(data_dir, items, filename="dev_hf.jsonl")
        monkeypatch.setattr("benchmarks.fever.BenchmarkSuite.data_dir", property(lambda self: data_dir))

        suite = FEVER()
        examples = suite.load(split="dev")
        assert len(examples) == 1

    def test_multiple_claims(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "fever"
        items = [
            {"id": i, "claim": f"Claim {i}", "label": "SUPPORTS", "evidence": []}
            for i in range(5)
        ]
        self._write_fever_data(data_dir, items)
        monkeypatch.setattr("benchmarks.fever.BenchmarkSuite.data_dir", property(lambda self: data_dir))

        suite = FEVER()
        examples = suite.load(split="dev")
        assert len(examples) == 5


# ---------------------------------------------------------------------------
# Wiki sentence loading
# ---------------------------------------------------------------------------


class TestLoadWikiSentences:
    """Tests for FEVER._load_wiki_sentences."""

    def test_load_wiki_pages(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "fever"
        wiki_dir = data_dir / "wiki-pages" / "wiki-pages"
        wiki_dir.mkdir(parents=True)

        wiki_file = wiki_dir / "wiki-001.jsonl"
        pages = [
            {"id": "Page_A", "lines": "0\tFirst sentence.\n1\tSecond sentence."},
            {"id": "Page_B", "lines": "0\tOnly sentence."},
        ]
        with open(wiki_file, "w") as f:
            for page in pages:
                f.write(json.dumps(page) + "\n")

        monkeypatch.setattr("benchmarks.fever.BenchmarkSuite.data_dir", property(lambda self: data_dir))
        suite = FEVER()
        wiki = suite._load_wiki_sentences()
        assert len(wiki) == 2
        assert wiki["Page_A"] == ["First sentence.", "Second sentence."]
        assert wiki["Page_B"] == ["Only sentence."]

    def test_wiki_caching(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "fever"
        wiki_dir = data_dir / "wiki-pages" / "wiki-pages"
        wiki_dir.mkdir(parents=True)
        wiki_file = wiki_dir / "wiki-001.jsonl"
        wiki_file.write_text(json.dumps({"id": "P", "lines": "0\tS."}) + "\n")

        monkeypatch.setattr("benchmarks.fever.BenchmarkSuite.data_dir", property(lambda self: data_dir))
        suite = FEVER()
        first = suite._load_wiki_sentences()
        second = suite._load_wiki_sentences()
        assert first is second

    def test_no_wiki_dir_returns_empty(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "fever"
        data_dir.mkdir()
        monkeypatch.setattr("benchmarks.fever.BenchmarkSuite.data_dir", property(lambda self: data_dir))
        suite = FEVER()
        wiki = suite._load_wiki_sentences()
        assert wiki == {}
