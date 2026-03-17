"""Tests for benchmarks.nli module.

Tests NLI utility functions (sentence splitting, vocab overlap, score conversion)
without requiring actual model loading.
"""

import json

import pytest

from benchmarks.nli import (
    _decompose_evidence,
    _minicheck_scores_to_nli,
    compute_vocab_overlap,
    is_minicheck_model,
    is_moritzlaurer_model,
    split_sentences,
)

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


class TestSplitSentences:
    """Tests for split_sentences."""

    def test_basic_split(self):
        text = "First sentence. Second sentence. Third sentence."
        sents = split_sentences(text)
        assert len(sents) >= 2

    def test_single_sentence(self):
        sents = split_sentences("Just one sentence here.")
        assert len(sents) == 1
        assert sents[0] == "Just one sentence here."

    def test_empty_returns_original(self):
        sents = split_sentences("")
        assert sents == [""]

    def test_abbreviation_et_al(self):
        text = "Smith et al. found that the method works. The results are significant."
        sents = split_sentences(text)
        # "et al." should NOT cause a split
        assert any("et al." in s for s in sents)

    def test_abbreviation_eg(self):
        text = "Some methods e.g. BERT work well. Other methods also work."
        sents = split_sentences(text)
        assert any("e.g." in s for s in sents)

    def test_abbreviation_ie(self):
        text = "The model i.e. BERT was used. It performed well."
        sents = split_sentences(text)
        assert any("i.e." in s for s in sents)

    def test_abbreviation_dr(self):
        text = "Dr. Smith conducted the study. The results were positive."
        sents = split_sentences(text)
        assert any("Dr." in s for s in sents)

    def test_abbreviation_fig(self):
        text = "See Fig. 3 for details. The graph shows a trend."
        sents = split_sentences(text)
        assert any("Fig." in s for s in sents)

    def test_abbreviation_us(self):
        text = "The U.S. government funded the research. It was published in 2024."
        sents = split_sentences(text)
        assert any("U.S." in s for s in sents)

    def test_exclamation_mark(self):
        text = "This is great! The results are amazing."
        sents = split_sentences(text)
        assert len(sents) == 2

    def test_question_mark(self):
        text = "Is this correct? The evidence suggests yes."
        sents = split_sentences(text)
        assert len(sents) == 2

    def test_short_fragments_filtered(self):
        """Fragments <= 5 characters are filtered out."""
        text = "A. B. C. This is a real sentence. Another sentence here."
        sents = split_sentences(text)
        # Short fragments should be filtered
        for s in sents:
            assert len(s) > 5

    def test_preserves_content(self):
        text = "The model achieved 95% accuracy. Training took 3 hours."
        sents = split_sentences(text)
        full_text = " ".join(sents)
        assert "95%" in full_text
        assert "3 hours" in full_text


# ---------------------------------------------------------------------------
# Vocab overlap
# ---------------------------------------------------------------------------


class TestComputeVocabOverlap:
    """Tests for compute_vocab_overlap."""

    def test_identical_text(self):
        text = "machine learning model performs classification"
        score = compute_vocab_overlap(text, text)
        assert score == 1.0

    def test_no_overlap(self):
        claim = "quantum computing breakthrough"
        evidence = "recipes cooking kitchen"
        score = compute_vocab_overlap(claim, evidence)
        assert score == 0.0

    def test_partial_overlap(self):
        claim = "deep learning model accuracy"
        evidence = "the deep learning approach showed improvements"
        score = compute_vocab_overlap(claim, evidence)
        assert 0.0 < score < 1.0

    def test_stopwords_excluded(self):
        """Stopwords don't count toward overlap."""
        claim = "the is a an"
        evidence = "completely different text"
        score = compute_vocab_overlap(claim, evidence)
        # All claim tokens are stopwords, so claim_tokens is empty -> returns 1.0
        assert score == 1.0

    def test_case_insensitive(self):
        claim = "BERT Model Performance"
        evidence = "bert model performance evaluation"
        score = compute_vocab_overlap(claim, evidence)
        assert score == 1.0

    def test_empty_claim_returns_one(self):
        """Empty claim (after stopword removal) returns 1.0."""
        score = compute_vocab_overlap("the a an", "some evidence")
        assert score == 1.0


# ---------------------------------------------------------------------------
# MiniCheck score conversion
# ---------------------------------------------------------------------------


class TestMiniCheckScoresToNLI:
    """Tests for _minicheck_scores_to_nli."""

    def test_high_support(self):
        scores = _minicheck_scores_to_nli(0.9)
        assert scores[1] == 0.9  # entailment
        assert scores[0] < 0.1  # contradiction low
        assert sum(scores) == pytest.approx(1.0, abs=0.01)

    def test_low_support(self):
        scores = _minicheck_scores_to_nli(0.1)
        assert scores[1] == 0.1  # entailment
        assert scores[0] > 0.3  # contradiction high
        assert sum(scores) == pytest.approx(1.0, abs=0.01)

    def test_medium_support(self):
        scores = _minicheck_scores_to_nli(0.5)
        assert scores[1] == 0.5  # entailment
        assert scores[0] == 0.05  # small contradiction
        assert sum(scores) == pytest.approx(1.0, abs=0.01)

    def test_zero_support(self):
        scores = _minicheck_scores_to_nli(0.0)
        assert scores[1] == 0.0  # no entailment
        assert scores[0] == pytest.approx(0.7)  # max contradiction

    def test_full_support(self):
        scores = _minicheck_scores_to_nli(1.0)
        assert scores[1] == 1.0  # full entailment
        assert scores[0] == 0.05  # minimal contradiction

    def test_returns_three_values(self):
        scores = _minicheck_scores_to_nli(0.5)
        assert len(scores) == 3

    def test_order_is_contradiction_entailment_neutral(self):
        """Output order matches cross-encoder convention: [con, ent, neu]."""
        scores = _minicheck_scores_to_nli(0.8)
        # Entailment (index 1) should be the support score
        assert scores[1] == 0.8

    def test_boundary_0_3(self):
        """At 0.3 boundary, contradiction transitions."""
        scores = _minicheck_scores_to_nli(0.3)
        assert scores[1] == 0.3
        # Just at the boundary - should still have some contradiction
        assert scores[0] >= 0.0


# ---------------------------------------------------------------------------
# Model type detection
# ---------------------------------------------------------------------------


class TestIsMiniCheckModel:
    """Tests for is_minicheck_model."""

    def test_known_minicheck_model(self):
        assert is_minicheck_model("lytang/MiniCheck-DeBERTa-v3-Large")

    def test_known_factcg_model(self):
        assert is_minicheck_model("yaxili96/FactCG-DeBERTa-v3-Large")

    def test_regular_model_not_minicheck(self):
        assert not is_minicheck_model("cross-encoder/nli-deberta-v3-base")

    def test_local_binary_model(self, tmp_path):
        """Detects binary model from local config.json."""
        config = {"id2label": {"0": "not_supported", "1": "supported"}}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))
        assert is_minicheck_model(str(tmp_path))

    def test_local_ternary_model_not_minicheck(self, tmp_path):
        """Ternary model from local config.json is NOT minicheck."""
        config = {"id2label": {"0": "contradiction", "1": "entailment", "2": "neutral"}}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))
        assert not is_minicheck_model(str(tmp_path))


class TestIsMoritzLaurerModel:
    """Tests for is_moritzlaurer_model."""

    def test_known_model(self):
        assert is_moritzlaurer_model("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")

    def test_unknown_model(self):
        assert not is_moritzlaurer_model("cross-encoder/nli-deberta-v3-base")


# ---------------------------------------------------------------------------
# Evidence decomposition
# ---------------------------------------------------------------------------


class TestDecomposeEvidence:
    """Tests for _decompose_evidence."""

    def test_short_evidence_unchanged(self):
        """Short evidence (<= 20 words) is not decomposed."""
        evidence = ["The sky is blue."]
        result = _decompose_evidence(evidence)
        assert result == ["The sky is blue."]

    def test_long_evidence_decomposed(self):
        """Long evidence is split into sentences."""
        long_text = " ".join(["This is a sentence about topic X."] * 10)
        evidence = [long_text]
        result = _decompose_evidence(evidence)
        assert len(result) > 1

    def test_max_sentences_limit(self):
        """Decomposed evidence is capped at max_sentences."""
        long_text = " ".join([f"Sentence number {i} about the topic." for i in range(100)])
        evidence = [long_text]
        result = _decompose_evidence(evidence, max_sentences=10)
        assert len(result) <= 10

    def test_empty_evidence_returns_original(self):
        evidence = [""]
        result = _decompose_evidence(evidence)
        assert result == [""]

    def test_multiple_evidence_entries(self):
        evidence = [
            "Short entry.",
            " ".join(["Long entry sentence."] * 15),
        ]
        result = _decompose_evidence(evidence)
        # First should stay, second should be decomposed
        assert len(result) >= 2

    def test_preserves_content(self):
        """All content from original evidence is preserved."""
        evidence = ["The model uses BERT. It achieves 95% accuracy on the benchmark."]
        result = _decompose_evidence(evidence, min_length=5)
        full = " ".join(result)
        assert "BERT" in full
        assert "95%" in full
