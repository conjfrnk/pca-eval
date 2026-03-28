"""Tests for benchmarks.qasper module.

Tests QASPER parsing, label mapping, and answer reformulation
without requiring NLI models or real dataset downloads.
"""

import json

import pytest

from benchmarks.qasper import QASPER, _reformulate_answer

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------


class TestMapNLILabel:
    """Tests for QASPER.map_nli_label."""

    def setup_method(self):
        self.suite = QASPER()

    def test_supports_maps_to_answerable(self):
        assert self.suite.map_nli_label("SUPPORTS") == "ANSWERABLE"

    def test_refutes_maps_to_answerable(self):
        assert self.suite.map_nli_label("REFUTES") == "ANSWERABLE"

    def test_nei_maps_to_unanswerable(self):
        assert self.suite.map_nli_label("NOT_ENOUGH_INFO") == "UNANSWERABLE"

    def test_unknown_maps_to_unanswerable(self):
        assert self.suite.map_nli_label("UNKNOWN") == "UNANSWERABLE"


# ---------------------------------------------------------------------------
# Answer reformulation
# ---------------------------------------------------------------------------


class TestReformulateAnswer:
    """Tests for _reformulate_answer with declarative reformulation."""

    def test_yes_does_declarative(self):
        """'Does X Y?' + Yes → 'X Y.' (drop auxiliary)."""
        result = _reformulate_answer("Yes", "Does the method work?")
        assert result == "the method work."

    def test_yes_is_declarative(self):
        """'Is X Y?' + Yes → 'X is Y.' (declarative)."""
        result = _reformulate_answer("Yes", "Is the model realistic?")
        assert "model" in result
        assert "is" in result.lower()
        assert "realistic" in result

    def test_no_does_declarative(self):
        """'Does X Y?' + No → 'X does not Y.'."""
        result = _reformulate_answer("No", "Does it converge?")
        assert "does not" in result

    def test_no_is_declarative(self):
        """'Is it effective?' + No → declarative negation."""
        result = _reformulate_answer("No", "Is it effective?")
        assert "not" in result

    def test_true_answer(self):
        result = _reformulate_answer("True", "Is the hypothesis valid?")
        assert "valid" in result

    def test_false_answer(self):
        result = _reformulate_answer("False", "Does it converge?")
        assert "not" in result

    def test_short_answer_what_is(self):
        """Short answers with 'What is X?' get merged into declarative."""
        result = _reformulate_answer("BERT", "What is the model used?")
        assert "BERT" in result
        assert "model" in result

    def test_short_answer_what_noun(self):
        """Short answers with 'What X is Y?' get merged into declarative."""
        result = _reformulate_answer("BERT model", "What model was used?")
        assert "BERT model" in result

    def test_long_answer_unchanged(self):
        """Longer answers are returned as-is."""
        answer = "The experiment showed that the BERT model significantly outperformed baselines."
        result = _reformulate_answer(answer, "What were the results?")
        assert result == answer

    def test_medium_answer_with_context(self):
        """5-word answers get question context appended."""
        answer = "The model performs well."
        result = _reformulate_answer(answer, "How does it perform?")
        assert "model performs well" in result

    def test_period_handling(self):
        """Short answer with period gets handled correctly."""
        result = _reformulate_answer("LSTM.", "What architecture?")
        assert "LSTM" in result

    def test_can_modal(self):
        """'Can X Y?' + Yes → 'X can Y.'."""
        result = _reformulate_answer("Yes", "Can BERT handle long sequences?")
        assert "BERT" in result
        assert "can" in result
        assert "handle" in result

    def test_are_with_complement(self):
        """'Are X Y?' + No → 'X are not Y.'."""
        result = _reformulate_answer("No", "Are the results significant?")
        assert "not" in result
        assert "significant" in result


# ---------------------------------------------------------------------------
# QASPER JSON parsing
# ---------------------------------------------------------------------------


class TestParseQasperJson:
    """Tests for QASPER._parse_qasper_json."""

    def setup_method(self):
        self.suite = QASPER()

    def _make_paper(self, **kwargs):
        """Create a minimal QASPER paper dict."""
        paper = {
            "title": kwargs.get("title", "Test Paper"),
            "abstract": kwargs.get("abstract", "This is the abstract."),
            "full_text": kwargs.get("full_text", []),
            "qas": kwargs.get("qas", []),
        }
        return paper

    def test_answerable_extractive(self):
        paper = self._make_paper(qas=[{
            "question": "What method was used?",
            "question_id": "q1",
            "answers": [{
                "answer": {
                    "unanswerable": False,
                    "extractive_spans": ["BERT"],
                    "evidence": ["We used BERT for classification."],
                },
            }],
        }])

        examples = self.suite._parse_qasper_json({"paper1": paper})
        assert len(examples) == 1
        assert examples[0].gold_label == "ANSWERABLE"
        assert examples[0].answer_text == "BERT"
        assert examples[0].claim_or_query == "What method was used?"

    def test_answerable_free_form(self):
        paper = self._make_paper(qas=[{
            "question": "What were the results?",
            "question_id": "q1",
            "answers": [{
                "answer": {
                    "unanswerable": False,
                    "free_form_answer": "The model achieved 95% accuracy.",
                    "evidence": [],
                },
            }],
        }])

        examples = self.suite._parse_qasper_json({"paper1": paper})
        assert examples[0].answer_text == "The model achieved 95% accuracy."

    def test_answerable_yes_no(self):
        paper = self._make_paper(qas=[{
            "question": "Does it work?",
            "question_id": "q1",
            "answers": [{
                "answer": {
                    "unanswerable": False,
                    "yes_no": True,
                    "evidence": [],
                },
            }],
        }])

        examples = self.suite._parse_qasper_json({"paper1": paper})
        assert examples[0].answer_text == "Yes"

    def test_answerable_yes_no_false(self):
        paper = self._make_paper(qas=[{
            "question": "Is it perfect?",
            "question_id": "q1",
            "answers": [{
                "answer": {
                    "unanswerable": False,
                    "yes_no": False,
                    "evidence": [],
                },
            }],
        }])

        examples = self.suite._parse_qasper_json({"paper1": paper})
        assert examples[0].answer_text == "No"

    def test_unanswerable(self):
        paper = self._make_paper(qas=[{
            "question": "What is the meaning of life?",
            "question_id": "q1",
            "answers": [{
                "answer": {
                    "unanswerable": True,
                    "evidence": [],
                },
            }],
        }])

        examples = self.suite._parse_qasper_json({"paper1": paper})
        assert examples[0].gold_label == "UNANSWERABLE"

    def test_multiple_questions(self):
        paper = self._make_paper(qas=[
            {
                "question": "Q1?",
                "question_id": "q1",
                "answers": [{"answer": {"unanswerable": False, "free_form_answer": "A1", "evidence": []}}],
            },
            {
                "question": "Q2?",
                "question_id": "q2",
                "answers": [{"answer": {"unanswerable": True, "evidence": []}}],
            },
        ])

        examples = self.suite._parse_qasper_json({"paper1": paper})
        assert len(examples) == 2

    def test_evidence_extraction(self):
        paper = self._make_paper(
            full_text=[
                {"section_name": "Methods", "paragraphs": ["We used BERT.", "Training was done."]},
            ],
            qas=[{
                "question": "Q?",
                "question_id": "q1",
                "answers": [{
                    "answer": {
                        "unanswerable": False,
                        "free_form_answer": "BERT",
                        "evidence": ["We used BERT."],
                    },
                }],
            }],
        )

        examples = self.suite._parse_qasper_json({"paper1": paper})
        assert "We used BERT." in examples[0].evidence_sentences

    def test_multiple_papers(self):
        paper1 = self._make_paper(qas=[{
            "question": "Q1?", "question_id": "q1",
            "answers": [{"answer": {"unanswerable": True, "evidence": []}}],
        }])
        paper2 = self._make_paper(qas=[{
            "question": "Q2?", "question_id": "q1",
            "answers": [{"answer": {"unanswerable": False, "free_form_answer": "A", "evidence": []}}],
        }])

        examples = self.suite._parse_qasper_json({"p1": paper1, "p2": paper2})
        assert len(examples) == 2

    def test_id_format(self):
        paper = self._make_paper(qas=[{
            "question": "Q?", "question_id": "q42",
            "answers": [{"answer": {"unanswerable": True, "evidence": []}}],
        }])
        examples = self.suite._parse_qasper_json({"paper_abc": paper})
        assert examples[0].id == "paper_abc_q42"

    def test_full_text_in_source(self):
        paper = self._make_paper(
            abstract="Abstract text.",
            full_text=[
                {"section_name": "Intro", "paragraphs": ["Paragraph 1.", "Paragraph 2."]},
            ],
            qas=[{
                "question": "Q?", "question_id": "q1",
                "answers": [{"answer": {"unanswerable": True, "evidence": []}}],
            }],
        )
        examples = self.suite._parse_qasper_json({"p1": paper})
        assert "Abstract text." in examples[0].full_source_text
        assert "Paragraph 1." in examples[0].full_source_text


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


class TestLoad:
    """Tests for QASPER.load."""

    def test_load_json_format(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "qasper"
        data_dir.mkdir()

        paper = {
            "title": "Paper",
            "abstract": "Abstract.",
            "full_text": [],
            "qas": [{
                "question": "Q?",
                "question_id": "q1",
                "answers": [{"answer": {"unanswerable": True, "evidence": []}}],
            }],
        }
        data_file = data_dir / "qasper_dev.json"
        data_file.write_text(json.dumps({"paper1": paper}))

        monkeypatch.setattr("benchmarks.qasper.BenchmarkSuite.data_dir", property(lambda self: data_dir))
        suite = QASPER()
        examples = suite.load(split="dev")
        assert len(examples) == 1

    def test_missing_data_raises(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "qasper"
        data_dir.mkdir()
        monkeypatch.setattr("benchmarks.qasper.BenchmarkSuite.data_dir", property(lambda self: data_dir))
        suite = QASPER()
        with pytest.raises(FileNotFoundError):
            suite.load(split="dev")
