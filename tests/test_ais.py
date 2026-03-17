"""Tests for AIS (Attributable to Identified Sources) metric module.

Tests dataclasses, statement segmentation, and scoring logic.
Uses mocked NLI evaluator to avoid model loading.
"""

from unittest.mock import MagicMock

import pytest

from benchmarks.ais import AISExample, AISReport, AISResult, AISScorer

# ---------------------------------------------------------------------------
# AISExample dataclass
# ---------------------------------------------------------------------------


class TestAISExample:
    def test_basic_creation(self):
        ex = AISExample(
            id="ex1",
            generated_text="The sky is blue.",
            source_texts=["The sky appears blue due to Rayleigh scattering."],
        )
        assert ex.id == "ex1"
        assert ex.generated_text == "The sky is blue."
        assert len(ex.source_texts) == 1

    def test_default_statements_empty(self):
        ex = AISExample(id="ex1", generated_text="text", source_texts=["src"])
        assert ex.statements == []

    def test_custom_statements(self):
        ex = AISExample(
            id="ex1",
            generated_text="Full text.",
            source_texts=["Source."],
            statements=["Statement one.", "Statement two."],
        )
        assert len(ex.statements) == 2

    def test_multiple_sources(self):
        ex = AISExample(
            id="ex1",
            generated_text="text",
            source_texts=["source1", "source2", "source3"],
        )
        assert len(ex.source_texts) == 3


# ---------------------------------------------------------------------------
# AISResult dataclass
# ---------------------------------------------------------------------------


class TestAISResult:
    def test_basic_creation(self):
        r = AISResult(
            id="ex1",
            num_statements=3,
            num_attributable=2,
            ais_score=2 / 3,
        )
        assert r.num_statements == 3
        assert r.num_attributable == 2
        assert abs(r.ais_score - 2 / 3) < 1e-6

    def test_default_per_statement_empty(self):
        r = AISResult(id="ex1", num_statements=1, num_attributable=1, ais_score=1.0)
        assert r.per_statement == []

    def test_perfect_score(self):
        r = AISResult(id="ex1", num_statements=5, num_attributable=5, ais_score=1.0)
        assert r.ais_score == 1.0

    def test_zero_score(self):
        r = AISResult(id="ex1", num_statements=5, num_attributable=0, ais_score=0.0)
        assert r.ais_score == 0.0


# ---------------------------------------------------------------------------
# AISReport
# ---------------------------------------------------------------------------


class TestAISReport:
    def test_basic_creation(self):
        report = AISReport(
            num_examples=10,
            mean_ais_score=0.85,
            median_ais_score=0.90,
            min_ais_score=0.50,
            max_ais_score=1.0,
            fully_attributable_rate=0.60,
        )
        assert report.num_examples == 10
        assert report.mean_ais_score == 0.85

    def test_default_results_empty(self):
        report = AISReport(
            num_examples=0, mean_ais_score=0, median_ais_score=0,
            min_ais_score=0, max_ais_score=0, fully_attributable_rate=0,
        )
        assert report.results == []

    def test_summary_format(self):
        report = AISReport(
            num_examples=10,
            mean_ais_score=0.85,
            median_ais_score=0.90,
            min_ais_score=0.50,
            max_ais_score=1.0,
            fully_attributable_rate=0.60,
        )
        summary = report.summary()
        assert "AIS" in summary
        assert "10" in summary
        assert "85.0%" in summary
        assert "90.0%" in summary
        assert "50.0%" in summary
        assert "100.0%" in summary
        assert "60.0%" in summary

    def test_summary_contains_all_metrics(self):
        report = AISReport(
            num_examples=5,
            mean_ais_score=0.75,
            median_ais_score=0.80,
            min_ais_score=0.20,
            max_ais_score=1.0,
            fully_attributable_rate=0.40,
        )
        summary = report.summary()
        assert "Mean AIS" in summary
        assert "Median AIS" in summary
        assert "Min AIS" in summary
        assert "Max AIS" in summary
        assert "Fully Attributable" in summary


# ---------------------------------------------------------------------------
# AISScorer.segment_statements
# ---------------------------------------------------------------------------


class TestSegmentStatements:
    @pytest.fixture
    def scorer(self):
        mock_nli = MagicMock()
        return AISScorer(nli=mock_nli)

    def test_single_sentence(self, scorer):
        result = scorer.segment_statements("The model achieves high accuracy.")
        assert len(result) == 1
        assert "accuracy" in result[0]

    def test_multiple_sentences(self, scorer):
        text = "First sentence here. Second sentence follows. Third one too."
        result = scorer.segment_statements(text)
        assert len(result) == 3

    def test_exclamation_and_question(self, scorer):
        text = "This is amazing! What do you think? It works well."
        result = scorer.segment_statements(text)
        assert len(result) == 3

    def test_short_fragments_filtered(self, scorer):
        text = "Ok. The model achieves high accuracy on the benchmark."
        result = scorer.segment_statements(text)
        # "Ok." is <= 10 chars, should be filtered
        assert all(len(s) > 10 for s in result)

    def test_all_short_returns_original(self, scorer):
        text = "Short."
        result = scorer.segment_statements(text)
        assert result == ["Short."]

    def test_empty_string(self, scorer):
        result = scorer.segment_statements("")
        assert result == [""]

    def test_whitespace_stripped(self, scorer):
        text = "  The model works well.  "
        result = scorer.segment_statements(text)
        assert result[0] == "The model works well."

    def test_preserves_sentence_content(self, scorer):
        text = "DeBERTa achieves 95.3% on SciFact. The model uses attention mechanisms."
        result = scorer.segment_statements(text)
        assert any("DeBERTa" in s for s in result)
        assert any("attention" in s for s in result)


# ---------------------------------------------------------------------------
# AISScorer.score_single (with mocked NLI)
# ---------------------------------------------------------------------------


class TestScoreSingle:
    @pytest.fixture
    def mock_nli(self):
        nli = MagicMock()
        return nli

    @pytest.fixture
    def scorer(self, mock_nli):
        return AISScorer(nli=mock_nli, entailment_threshold=0.5)

    def test_fully_attributable(self, scorer, mock_nli):
        mock_nli.classify_claim.return_value = {
            "entailment": 0.95,
            "supporting_sentences": [0],
        }
        example = AISExample(
            id="ex1",
            generated_text="The model achieves high accuracy.",
            source_texts=["The model achieves 95% accuracy on all benchmarks."],
            statements=["The model achieves high accuracy."],
        )
        result = scorer.score_single(example)
        assert result.ais_score == 1.0
        assert result.num_attributable == 1
        assert result.num_statements == 1

    def test_not_attributable(self, scorer, mock_nli):
        mock_nli.classify_claim.return_value = {
            "entailment": 0.1,
            "supporting_sentences": [],
        }
        example = AISExample(
            id="ex1",
            generated_text="The model is bad.",
            source_texts=["The model achieves excellent performance."],
            statements=["The model is bad."],
        )
        result = scorer.score_single(example)
        assert result.ais_score == 0.0
        assert result.num_attributable == 0

    def test_partial_attribution(self, scorer, mock_nli):
        # First call: attributable, second: not
        mock_nli.classify_claim.side_effect = [
            {"entailment": 0.9, "supporting_sentences": [0]},
            {"entailment": 0.2, "supporting_sentences": []},
        ]
        example = AISExample(
            id="ex1",
            generated_text="Stmt1. Stmt2.",
            source_texts=["Source text."],
            statements=["Statement one is supported.", "Statement two is not."],
        )
        result = scorer.score_single(example)
        assert result.ais_score == 0.5
        assert result.num_attributable == 1
        assert result.num_statements == 2

    def test_empty_sources(self, scorer, mock_nli):
        example = AISExample(
            id="ex1",
            generated_text="Some claim.",
            source_texts=[],
            statements=["Some claim."],
        )
        result = scorer.score_single(example)
        assert result.ais_score == 0.0

    def test_no_sources_returns_zero(self, scorer, mock_nli):
        example = AISExample(
            id="ex1",
            generated_text="Some generated text here.",
            source_texts=[],
            statements=["Some generated text here."],
        )
        result = scorer.score_single(example)
        assert result.ais_score == 0.0
        assert result.num_attributable == 0

    def test_per_statement_details(self, scorer, mock_nli):
        mock_nli.classify_claim.return_value = {
            "entailment": 0.8,
            "supporting_sentences": [1],
        }
        example = AISExample(
            id="ex1",
            generated_text="The sky is blue.",
            source_texts=["Light scattering.", "Sky appears blue."],
            statements=["The sky is blue."],
        )
        result = scorer.score_single(example)
        assert len(result.per_statement) == 1
        assert result.per_statement[0]["attributable"] is True
        assert result.per_statement[0]["entailment"] == 0.8
        assert result.per_statement[0]["best_evidence_idx"] == 1

    def test_threshold_boundary_below(self, scorer, mock_nli):
        mock_nli.classify_claim.return_value = {
            "entailment": 0.49,
            "supporting_sentences": [],
        }
        example = AISExample(
            id="ex1",
            generated_text="test",
            source_texts=["src"],
            statements=["Just below threshold statement."],
        )
        result = scorer.score_single(example)
        assert result.num_attributable == 0

    def test_threshold_boundary_at(self, scorer, mock_nli):
        mock_nli.classify_claim.return_value = {
            "entailment": 0.5,
            "supporting_sentences": [0],
        }
        example = AISExample(
            id="ex1",
            generated_text="test",
            source_texts=["src"],
            statements=["Exactly at threshold statement."],
        )
        result = scorer.score_single(example)
        assert result.num_attributable == 1

    def test_uses_pre_segmented_statements(self, scorer, mock_nli):
        mock_nli.classify_claim.return_value = {
            "entailment": 0.9,
            "supporting_sentences": [0],
        }
        example = AISExample(
            id="ex1",
            generated_text="Full text that would segment differently.",
            source_texts=["Source."],
            statements=["Pre-segmented statement."],
        )
        scorer.score_single(example)
        # Should use the provided statement, not segment the generated text
        mock_nli.classify_claim.assert_called_once()
        call_args = mock_nli.classify_claim.call_args
        assert call_args.kwargs["claim"] == "Pre-segmented statement."


# ---------------------------------------------------------------------------
# AISScorer.score_batch
# ---------------------------------------------------------------------------


class TestScoreBatch:
    @pytest.fixture
    def mock_nli(self):
        nli = MagicMock()
        nli.classify_claim.return_value = {
            "entailment": 0.9,
            "supporting_sentences": [0],
        }
        return nli

    @pytest.fixture
    def scorer(self, mock_nli):
        return AISScorer(nli=mock_nli)

    def test_empty_batch(self, scorer):
        report = scorer.score_batch([])
        assert report.num_examples == 0
        assert report.mean_ais_score == 0

    def test_single_example(self, scorer):
        examples = [
            AISExample(
                id="ex1",
                generated_text="Test.",
                source_texts=["Source."],
                statements=["Test statement here."],
            )
        ]
        report = scorer.score_batch(examples)
        assert report.num_examples == 1
        assert report.mean_ais_score == 1.0

    def test_multiple_examples(self, scorer, mock_nli):
        # All return high entailment
        examples = [
            AISExample(
                id=f"ex{i}",
                generated_text=f"Text {i}.",
                source_texts=[f"Source {i}."],
                statements=[f"Statement number {i} here."],
            )
            for i in range(5)
        ]
        report = scorer.score_batch(examples)
        assert report.num_examples == 5
        assert report.mean_ais_score == 1.0
        assert report.fully_attributable_rate == 1.0
        assert report.min_ais_score == 1.0
        assert report.max_ais_score == 1.0

    def test_mixed_scores(self, scorer, mock_nli):
        # Alternate between attributable and not
        mock_nli.classify_claim.side_effect = [
            {"entailment": 0.9, "supporting_sentences": [0]},  # ex1: attributable
            {"entailment": 0.1, "supporting_sentences": []},   # ex2: not attributable
        ]
        examples = [
            AISExample(id="ex1", generated_text="T.", source_texts=["S."],
                       statements=["Statement one is long enough."]),
            AISExample(id="ex2", generated_text="T.", source_texts=["S."],
                       statements=["Statement two is long enough."]),
        ]
        report = scorer.score_batch(examples)
        assert report.num_examples == 2
        assert report.mean_ais_score == 0.5
        assert report.min_ais_score == 0.0
        assert report.max_ais_score == 1.0
        assert report.fully_attributable_rate == 0.5

    def test_report_has_results(self, scorer):
        examples = [
            AISExample(id="ex1", generated_text="T.", source_texts=["S."],
                       statements=["A statement that is long."]),
        ]
        report = scorer.score_batch(examples)
        assert len(report.results) == 1
        assert report.results[0].id == "ex1"


# ---------------------------------------------------------------------------
# AISScorer.score_from_benchmark
# ---------------------------------------------------------------------------


class TestScoreFromBenchmark:
    @pytest.fixture
    def mock_nli(self):
        nli = MagicMock()
        nli.classify_claim.return_value = {
            "entailment": 0.9,
            "supporting_sentences": [0],
        }
        return nli

    @pytest.fixture
    def scorer(self, mock_nli):
        return AISScorer(nli=mock_nli)

    def test_converts_benchmark_examples(self, scorer):
        mock_ex = MagicMock()
        mock_ex.id = "ex1"
        mock_ex.answer_text = "The model works well on tasks."
        mock_ex.evidence_sentences = ["Evidence sentence."]

        report = scorer.score_from_benchmark("test_benchmark", [mock_ex])
        assert report.num_examples == 1

    def test_skips_examples_without_answer(self, scorer):
        mock_ex = MagicMock()
        mock_ex.answer_text = ""
        mock_ex.evidence_sentences = ["Evidence."]

        report = scorer.score_from_benchmark("test", [mock_ex])
        assert report.num_examples == 0

    def test_skips_examples_without_evidence(self, scorer):
        mock_ex = MagicMock()
        mock_ex.answer_text = "Answer text."
        mock_ex.evidence_sentences = []

        report = scorer.score_from_benchmark("test", [mock_ex])
        assert report.num_examples == 0

    def test_skips_examples_missing_attributes(self, scorer):
        mock_ex = MagicMock(spec=[])  # No attributes at all

        report = scorer.score_from_benchmark("test", [mock_ex])
        assert report.num_examples == 0

    def test_empty_examples_list(self, scorer):
        report = scorer.score_from_benchmark("test", [])
        assert report.num_examples == 0
        assert report.mean_ais_score == 0

    def test_mixed_valid_and_invalid(self, scorer):
        valid = MagicMock()
        valid.id = "valid1"
        valid.answer_text = "A valid answer text here."
        valid.evidence_sentences = ["Evidence."]

        invalid = MagicMock()
        invalid.answer_text = ""
        invalid.evidence_sentences = ["Evidence."]

        report = scorer.score_from_benchmark("test", [valid, invalid])
        assert report.num_examples == 1
