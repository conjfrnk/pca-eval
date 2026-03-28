"""Tests for benchmarks.hagrid module.

Tests HAGRID parsing, label mapping, strategy helper functions, and
entity/number overlap utilities without requiring NLI models.
"""

import json

import pytest

from benchmarks.hagrid import (
    HAGRID,
    _entity_overlap,
    _number_overlap,
    _split_sentences,
)

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------


class TestMapNLILabel:
    """Tests for HAGRID.map_nli_label."""

    def setup_method(self):
        self.suite = HAGRID()

    def test_supports_maps_to_attributable(self):
        assert self.suite.map_nli_label("SUPPORTS") == "ATTRIBUTABLE"

    def test_refutes_maps_to_not_attributable(self):
        assert self.suite.map_nli_label("REFUTES") == "NOT_ATTRIBUTABLE"

    def test_nei_maps_to_not_attributable(self):
        assert self.suite.map_nli_label("NOT_ENOUGH_INFO") == "NOT_ATTRIBUTABLE"

    def test_unknown_maps_to_not_attributable(self):
        assert self.suite.map_nli_label("UNKNOWN") == "NOT_ATTRIBUTABLE"


# ---------------------------------------------------------------------------
# Item parsing
# ---------------------------------------------------------------------------


class TestParseItem:
    """Tests for HAGRID._parse_item."""

    def setup_method(self):
        self.suite = HAGRID()

    def test_basic_attributable_answer(self):
        item = {
            "query": "What is the capital of France?",
            "knowledge": [{"title": "France", "text": "Paris is the capital of France."}],
            "answers": [
                {"answer": "Paris is the capital.", "attributable": True, "quotes": ["Paris is the capital of France."]},
            ],
        }
        examples = self.suite._parse_item(item, 0)
        assert len(examples) == 1
        assert examples[0].gold_label == "ATTRIBUTABLE"
        assert examples[0].claim_or_query == "What is the capital of France?"
        assert examples[0].answer_text == "Paris is the capital."

    def test_not_attributable_answer(self):
        item = {
            "query": "What is X?",
            "knowledge": [{"text": "Some evidence."}],
            "answers": [
                {"answer": "Hallucinated answer.", "attributable": False},
            ],
        }
        examples = self.suite._parse_item(item, 0)
        assert len(examples) == 1
        assert examples[0].gold_label == "NOT_ATTRIBUTABLE"

    def test_multiple_answers(self):
        item = {
            "query": "Question",
            "knowledge": [{"text": "Evidence."}],
            "answers": [
                {"answer": "Answer 1", "attributable": True},
                {"answer": "Answer 2", "attributable": False},
            ],
        }
        examples = self.suite._parse_item(item, 0)
        assert len(examples) == 2
        assert examples[0].gold_label == "ATTRIBUTABLE"
        assert examples[1].gold_label == "NOT_ATTRIBUTABLE"

    def test_string_answers(self):
        item = {
            "query": "Q",
            "knowledge": [{"text": "Evidence."}],
            "answers": ["Simple string answer"],
        }
        examples = self.suite._parse_item(item, 0)
        assert len(examples) == 1
        assert examples[0].answer_text == "Simple string answer"
        assert examples[0].gold_label == "ATTRIBUTABLE"

    def test_no_answers_no_passages(self):
        item = {"query": "Q"}
        examples = self.suite._parse_item(item, 0)
        assert examples == []

    def test_empty_answer_text_skipped(self):
        item = {
            "query": "Q",
            "knowledge": [{"text": "Evidence."}],
            "answers": [{"answer": "", "attributable": True}],
        }
        examples = self.suite._parse_item(item, 0)
        assert examples == []

    def test_quotes_used_as_evidence(self):
        item = {
            "query": "Q",
            "knowledge": [{"text": "Full passage text."}],
            "answers": [
                {
                    "answer": "Answer",
                    "attributable": True,
                    "quotes": ["Specific quote from passage."],
                },
            ],
        }
        examples = self.suite._parse_item(item, 0)
        assert "Specific quote from passage." in examples[0].evidence_sentences

    def test_passages_fallback_when_no_quotes(self):
        item = {
            "query": "Q",
            "knowledge": [{"text": "Passage text."}],
            "answers": [{"answer": "Answer", "attributable": True}],
        }
        examples = self.suite._parse_item(item, 0)
        assert "Passage text." in examples[0].evidence_sentences

    def test_knowledge_dict_with_title(self):
        item = {
            "query": "Q",
            "knowledge": [{"title": "Wikipedia", "text": "Content here."}],
            "answers": [{"answer": "A", "attributable": True}],
        }
        examples = self.suite._parse_item(item, 0)
        assert "Wikipedia: Content here." in examples[0].metadata["passages"]

    def test_passages_key_alternative(self):
        item = {
            "query": "Q",
            "passages": [{"text": "From passages key."}],
            "answers": [{"answer": "A", "attributable": True}],
        }
        examples = self.suite._parse_item(item, 0)
        assert examples[0].evidence_sentences == ["From passages key."]

    def test_documents_key_alternative(self):
        item = {
            "query": "Q",
            "documents": ["String passage directly."],
            "answers": [{"answer": "A", "attributable": True}],
        }
        examples = self.suite._parse_item(item, 0)
        assert "String passage directly." in examples[0].evidence_sentences

    def test_id_generation(self):
        item = {
            "query": "Q",
            "knowledge": [{"text": "E"}],
            "answers": [
                {"answer": "A1", "attributable": True},
                {"answer": "A2", "attributable": False},
            ],
        }
        examples = self.suite._parse_item(item, 10)
        assert examples[0].id == "hagrid_10"
        assert examples[1].id == "hagrid_11"

    def test_attributable_inferred_from_quotes(self):
        """When attributable field is absent, presence of quotes means ATTRIBUTABLE."""
        item = {
            "query": "Q",
            "knowledge": [{"text": "E"}],
            "answers": [{"answer": "A", "quotes": ["Some quote"]}],
        }
        examples = self.suite._parse_item(item, 0)
        assert examples[0].gold_label == "ATTRIBUTABLE"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestLoad:
    """Tests for HAGRID.load with mock data files."""

    def test_load_dev_split(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "hagrid"
        data_dir.mkdir()
        dev_file = data_dir / "dev.jsonl"

        items = [
            {
                "query": "Q1",
                "knowledge": [{"text": "Evidence 1."}],
                "answers": [{"answer": "A1", "attributable": True}],
            },
            {
                "query": "Q2",
                "knowledge": [{"text": "Evidence 2."}],
                "answers": [{"answer": "A2", "attributable": False}],
            },
        ]
        with open(dev_file, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        monkeypatch.setattr("benchmarks.hagrid.BenchmarkSuite.data_dir", property(lambda self: data_dir))
        suite = HAGRID()
        examples = suite.load(split="dev")
        assert len(examples) == 2

    def test_load_missing_file_raises(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "hagrid"
        data_dir.mkdir()
        monkeypatch.setattr("benchmarks.hagrid.BenchmarkSuite.data_dir", property(lambda self: data_dir))
        suite = HAGRID()
        with pytest.raises(FileNotFoundError):
            suite.load(split="nonexistent")


# ---------------------------------------------------------------------------
# Entity overlap
# ---------------------------------------------------------------------------


class TestEntityOverlap:
    """Tests for _entity_overlap utility."""

    def test_full_entity_overlap(self):
        claim = "Dr. Smith found that Paris is a city."
        evidence = "Dr. Smith studied Paris extensively."
        score = _entity_overlap(claim, evidence)
        assert score > 0.5

    def test_no_entities_returns_one(self):
        """No entities in claim = nothing to verify = 1.0."""
        assert _entity_overlap("the sky is blue", "the sky is blue") == 1.0

    def test_number_overlap(self):
        claim = "The population is 1000 people."
        evidence = "About 1000 people live there."
        score = _entity_overlap(claim, evidence)
        assert score > 0.0

    def test_missing_entity(self):
        claim = "Dr. Jones found the result in 2024."
        evidence = "The study was conducted recently."
        score = _entity_overlap(claim, evidence)
        assert score < 1.0

    def test_partial_overlap(self):
        claim = "Paris and London are capitals. Population is 5000."
        evidence = "Paris is a capital city. The population is 5000."
        score = _entity_overlap(claim, evidence)
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Number overlap
# ---------------------------------------------------------------------------


class TestNumberOverlap:
    """Tests for _number_overlap utility."""

    def test_full_number_overlap(self):
        assert _number_overlap("The year was 2024.", "In 2024, events occurred.") == 1.0

    def test_no_numbers_returns_one(self):
        assert _number_overlap("The sky is blue.", "Evidence text.") == 1.0

    def test_missing_numbers(self):
        assert _number_overlap("The count was 42 and 100.", "No numbers here.") == 0.0

    def test_partial_number_overlap(self):
        score = _number_overlap("Values are 10 and 20.", "The value is 10.")
        assert score == 0.5

    def test_decimal_numbers(self):
        assert _number_overlap("The rate is 3.14.", "Pi is approximately 3.14.") == 1.0


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


class TestSplitSentences:
    """Tests for _split_sentences wrapper."""

    def test_basic_split(self):
        text = "First sentence. Second sentence. Third sentence."
        sents = _split_sentences(text)
        assert len(sents) >= 2

    def test_single_sentence(self):
        text = "Just one sentence here."
        sents = _split_sentences(text)
        assert len(sents) == 1

    def test_empty_text(self):
        sents = _split_sentences("")
        assert len(sents) == 1  # Returns [text] for empty
