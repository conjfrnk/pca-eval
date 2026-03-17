"""Tests for benchmark suite label mapping and utility functions.

Tests SciFact, FEVER, and QASPER label mapping, data parsing, and
the _reformulate_answer utility without requiring model loading.
"""

import pytest

from benchmarks.fever import FEVER
from benchmarks.qasper import QASPER, _reformulate_answer
from benchmarks.scifact import SciFact

# ---------------------------------------------------------------------------
# SciFact.map_nli_label
# ---------------------------------------------------------------------------


class TestScifactMapNliLabel:
    @pytest.fixture
    def scifact(self):
        return SciFact()

    def test_supports_maps_to_supports(self, scifact):
        assert scifact.map_nli_label("SUPPORTS") == "SUPPORTS"

    def test_refutes_maps_to_refutes(self, scifact):
        assert scifact.map_nli_label("REFUTES") == "REFUTES"

    def test_nei_maps_to_nei(self, scifact):
        assert scifact.map_nli_label("NOT_ENOUGH_INFO") == "NOT_ENOUGH_INFO"

    def test_unknown_maps_to_nei(self, scifact):
        assert scifact.map_nli_label("UNKNOWN") == "NOT_ENOUGH_INFO"

    def test_entailment_maps_to_nei(self, scifact):
        assert scifact.map_nli_label("entailment") == "NOT_ENOUGH_INFO"


# ---------------------------------------------------------------------------
# SciFact class attributes
# ---------------------------------------------------------------------------


class TestScifactAttributes:
    def test_name(self):
        s = SciFact()
        assert s.name == "scifact"

    def test_labels(self):
        s = SciFact()
        assert "SUPPORTS" in s.labels
        assert "REFUTES" in s.labels
        assert "NOT_ENOUGH_INFO" in s.labels

    def test_description(self):
        s = SciFact()
        assert "scientific" in s.description.lower() or "claim" in s.description.lower()


# ---------------------------------------------------------------------------
# FEVER.map_nli_label
# ---------------------------------------------------------------------------


class TestFeverMapNliLabel:
    @pytest.fixture
    def fever(self):
        return FEVER()

    def test_supports_maps_to_supports(self, fever):
        assert fever.map_nli_label("SUPPORTS") == "SUPPORTS"

    def test_refutes_maps_to_refutes(self, fever):
        assert fever.map_nli_label("REFUTES") == "REFUTES"

    def test_nei_maps_to_nei(self, fever):
        assert fever.map_nli_label("NOT_ENOUGH_INFO") == "NOT_ENOUGH_INFO"

    def test_unknown_maps_to_nei(self, fever):
        assert fever.map_nli_label("UNKNOWN") == "NOT_ENOUGH_INFO"


# ---------------------------------------------------------------------------
# FEVER class attributes
# ---------------------------------------------------------------------------


class TestFeverAttributes:
    def test_name(self):
        f = FEVER()
        assert f.name == "fever"

    def test_labels(self):
        f = FEVER()
        assert "SUPPORTS" in f.labels
        assert "REFUTES" in f.labels
        assert "NOT_ENOUGH_INFO" in f.labels


# ---------------------------------------------------------------------------
# QASPER.map_nli_label
# ---------------------------------------------------------------------------


class TestQasperMapNliLabel:
    @pytest.fixture
    def qasper(self):
        return QASPER()

    def test_supports_maps_to_answerable(self, qasper):
        assert qasper.map_nli_label("SUPPORTS") == "ANSWERABLE"

    def test_refutes_maps_to_answerable(self, qasper):
        assert qasper.map_nli_label("REFUTES") == "ANSWERABLE"

    def test_nei_maps_to_unanswerable(self, qasper):
        assert qasper.map_nli_label("NOT_ENOUGH_INFO") == "UNANSWERABLE"

    def test_unknown_maps_to_unanswerable(self, qasper):
        assert qasper.map_nli_label("UNKNOWN") == "UNANSWERABLE"


# ---------------------------------------------------------------------------
# QASPER class attributes
# ---------------------------------------------------------------------------


class TestQasperAttributes:
    def test_name(self):
        q = QASPER()
        assert q.name == "qasper"

    def test_labels(self):
        q = QASPER()
        assert "ANSWERABLE" in q.labels
        assert "UNANSWERABLE" in q.labels


# ---------------------------------------------------------------------------
# _reformulate_answer
# ---------------------------------------------------------------------------


class TestReformulateAnswer:
    """Tests for declarative answer reformulation."""

    def test_yes_is_declarative(self):
        """'Is X Y?' + Yes → declarative 'X is Y.'"""
        result = _reformulate_answer("Yes", "Is the sky blue?")
        assert "sky" in result
        assert "blue" in result

    def test_no_does_declarative(self):
        """'Does X Y?' + No → 'X does not Y.'"""
        result = _reformulate_answer("No", "Does water freeze at 200C?")
        assert "does not" in result

    def test_true_answer(self):
        result = _reformulate_answer("True", "Is this correct?")
        assert "correct" in result

    def test_false_answer(self):
        result = _reformulate_answer("False", "Is this wrong?")
        assert "not" in result

    def test_yes_case_insensitive(self):
        result = _reformulate_answer("yes", "Is it valid?")
        assert "valid" in result

    def test_short_answer_gets_context(self):
        result = _reformulate_answer("42 percent", "What was the accuracy?")
        assert "42 percent" in result

    def test_single_word_answer(self):
        result = _reformulate_answer("DeBERTa", "Which model was used?")
        assert "DeBERTa" in result

    def test_three_word_answer(self):
        result = _reformulate_answer("deep neural network", "What architecture?")
        assert "deep neural network" in result

    def test_long_answer_unchanged(self):
        answer = "The model uses a transformer architecture with attention mechanisms for encoding."
        result = _reformulate_answer(answer, "What is the approach?")
        assert result == answer

    def test_four_word_answer_unchanged(self):
        answer = "deep neural network architecture"
        result = _reformulate_answer(answer, "What type?")
        assert "deep neural network architecture" in result

    def test_trailing_period_stripped_for_check(self):
        result = _reformulate_answer("Yes.", "Is it true?")
        assert "true" in result.lower() or "it" in result

    def test_whitespace_handling(self):
        result = _reformulate_answer("  Yes  ", "Is it true?")
        assert "true" in result.lower() or "it" in result

    def test_no_with_period(self):
        result = _reformulate_answer("No.", "Is it false?")
        assert "not" in result


# ---------------------------------------------------------------------------
# QASPER._parse_qasper_json
# ---------------------------------------------------------------------------


class TestQasperParseJson:
    @pytest.fixture
    def qasper(self):
        return QASPER()

    def test_basic_answerable(self, qasper):
        data = {
            "paper1": {
                "title": "Test Paper",
                "abstract": "This paper tests things.",
                "full_text": [],
                "qas": [
                    {
                        "question": "What does the paper test?",
                        "question_id": "q1",
                        "answers": [
                            {
                                "answer": {
                                    "free_form_answer": "It tests NLI systems.",
                                    "evidence": ["The paper evaluates NLI."],
                                }
                            }
                        ],
                    }
                ],
            }
        }
        examples = qasper._parse_qasper_json(data)
        assert len(examples) == 1
        assert examples[0].gold_label == "ANSWERABLE"
        assert examples[0].answer_text == "It tests NLI systems."
        assert examples[0].claim_or_query == "What does the paper test?"

    def test_unanswerable(self, qasper):
        data = {
            "paper1": {
                "title": "Test",
                "abstract": "Abstract.",
                "full_text": [],
                "qas": [
                    {
                        "question": "What about X?",
                        "question_id": "q1",
                        "answers": [
                            {"answer": {"unanswerable": True}}
                        ],
                    }
                ],
            }
        }
        examples = qasper._parse_qasper_json(data)
        assert len(examples) == 1
        assert examples[0].gold_label == "UNANSWERABLE"

    def test_extractive_answer(self, qasper):
        data = {
            "paper1": {
                "title": "Test",
                "abstract": "Abstract.",
                "full_text": [],
                "qas": [
                    {
                        "question": "What spans?",
                        "question_id": "q1",
                        "answers": [
                            {
                                "answer": {
                                    "extractive_spans": ["span one", "span two"],
                                    "evidence": [],
                                }
                            }
                        ],
                    }
                ],
            }
        }
        examples = qasper._parse_qasper_json(data)
        assert examples[0].answer_text == "span one span two"

    def test_yes_no_answer(self, qasper):
        data = {
            "paper1": {
                "title": "Test",
                "abstract": "Abstract.",
                "full_text": [],
                "qas": [
                    {
                        "question": "Is it good?",
                        "question_id": "q1",
                        "answers": [
                            {"answer": {"yes_no": True, "evidence": []}}
                        ],
                    }
                ],
            }
        }
        examples = qasper._parse_qasper_json(data)
        assert examples[0].answer_text == "Yes"

    def test_yes_no_false(self, qasper):
        data = {
            "paper1": {
                "title": "Test",
                "abstract": "Abstract.",
                "full_text": [],
                "qas": [
                    {
                        "question": "Is it bad?",
                        "question_id": "q1",
                        "answers": [
                            {"answer": {"yes_no": False, "evidence": []}}
                        ],
                    }
                ],
            }
        }
        examples = qasper._parse_qasper_json(data)
        assert examples[0].answer_text == "No"

    def test_multiple_questions(self, qasper):
        data = {
            "paper1": {
                "title": "Test",
                "abstract": "Abstract.",
                "full_text": [],
                "qas": [
                    {
                        "question": "Q1?",
                        "question_id": "q1",
                        "answers": [
                            {"answer": {"free_form_answer": "A1.", "evidence": []}}
                        ],
                    },
                    {
                        "question": "Q2?",
                        "question_id": "q2",
                        "answers": [
                            {"answer": {"free_form_answer": "A2.", "evidence": []}}
                        ],
                    },
                ],
            }
        }
        examples = qasper._parse_qasper_json(data)
        assert len(examples) == 2

    def test_multiple_papers(self, qasper):
        data = {
            "paper1": {
                "title": "Paper 1",
                "abstract": "Abstract 1.",
                "full_text": [],
                "qas": [
                    {
                        "question": "Q?",
                        "question_id": "q1",
                        "answers": [
                            {"answer": {"free_form_answer": "A.", "evidence": []}}
                        ],
                    }
                ],
            },
            "paper2": {
                "title": "Paper 2",
                "abstract": "Abstract 2.",
                "full_text": [],
                "qas": [
                    {
                        "question": "Q2?",
                        "question_id": "q2",
                        "answers": [
                            {"answer": {"free_form_answer": "A2.", "evidence": []}}
                        ],
                    }
                ],
            },
        }
        examples = qasper._parse_qasper_json(data)
        assert len(examples) == 2
        paper_ids = {ex.source_doc_id for ex in examples}
        assert "paper1" in paper_ids
        assert "paper2" in paper_ids

    def test_evidence_extracted(self, qasper):
        data = {
            "paper1": {
                "title": "Test",
                "abstract": "Abstract.",
                "full_text": [
                    {
                        "section_name": "Methods",
                        "paragraphs": ["We used DeBERTa.", "The model was fine-tuned."],
                    }
                ],
                "qas": [
                    {
                        "question": "What model?",
                        "question_id": "q1",
                        "answers": [
                            {
                                "answer": {
                                    "free_form_answer": "DeBERTa.",
                                    "evidence": ["We used DeBERTa."],
                                }
                            }
                        ],
                    }
                ],
            }
        }
        examples = qasper._parse_qasper_json(data)
        assert "We used DeBERTa." in examples[0].evidence_sentences

    def test_empty_data(self, qasper):
        examples = qasper._parse_qasper_json({})
        assert len(examples) == 0

    def test_paper_with_no_qas(self, qasper):
        data = {
            "paper1": {
                "title": "Test",
                "abstract": "Abstract.",
                "full_text": [],
                "qas": [],
            }
        }
        examples = qasper._parse_qasper_json(data)
        assert len(examples) == 0

    def test_example_id_format(self, qasper):
        data = {
            "paper_abc": {
                "title": "Test",
                "abstract": "Abstract.",
                "full_text": [],
                "qas": [
                    {
                        "question": "Q?",
                        "question_id": "q42",
                        "answers": [
                            {"answer": {"free_form_answer": "A.", "evidence": []}}
                        ],
                    }
                ],
            }
        }
        examples = qasper._parse_qasper_json(data)
        assert examples[0].id == "paper_abc_q42"

    def test_metadata_includes_abstract(self, qasper):
        data = {
            "paper1": {
                "title": "Test",
                "abstract": "The main abstract text.",
                "full_text": [],
                "qas": [
                    {
                        "question": "Q?",
                        "question_id": "q1",
                        "answers": [
                            {"answer": {"free_form_answer": "A.", "evidence": []}}
                        ],
                    }
                ],
            }
        }
        examples = qasper._parse_qasper_json(data)
        assert examples[0].metadata["abstract"] == "The main abstract text."

    def test_full_text_built_from_sections(self, qasper):
        data = {
            "paper1": {
                "title": "Test",
                "abstract": "Abstract here.",
                "full_text": [
                    {
                        "section_name": "Intro",
                        "paragraphs": ["Intro paragraph."],
                    },
                    {
                        "section_name": "Methods",
                        "paragraphs": ["Method paragraph."],
                    },
                ],
                "qas": [
                    {
                        "question": "Q?",
                        "question_id": "q1",
                        "answers": [
                            {"answer": {"free_form_answer": "A.", "evidence": []}}
                        ],
                    }
                ],
            }
        }
        examples = qasper._parse_qasper_json(data)
        assert "Abstract here." in examples[0].full_source_text
        assert "Intro paragraph." in examples[0].full_source_text
        assert "Method paragraph." in examples[0].full_source_text
