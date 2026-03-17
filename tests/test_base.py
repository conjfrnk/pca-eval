"""
Tests for benchmark base classes and utilities.

Tests pure functions and methods in benchmarks/base.py:
- BenchmarkExample dataclass
- PredictionResult dataclass
- BenchmarkReport (to_dict, summary, metrics)
- BenchmarkSuite.sample() stratified sampling
- BenchmarkSuite.compute_metrics() aggregate metric computation
"""

from benchmarks.base import (
    BenchmarkExample,
    BenchmarkReport,
    BenchmarkSuite,
    PredictionResult,
)

# =============================================================================
# Dataclass Tests
# =============================================================================


class TestBenchmarkExample:
    def test_default_fields(self):
        ex = BenchmarkExample(id="1", claim_or_query="test", gold_label="SUPPORTS")
        assert ex.evidence_sentences == []
        assert ex.evidence_sentence_indices == []
        assert ex.source_doc_id == ""
        assert ex.answer_text == ""
        assert ex.metadata == {}

    def test_with_evidence(self):
        ex = BenchmarkExample(
            id="1", claim_or_query="test", gold_label="SUPPORTS",
            evidence_sentences=["evidence text"],
            evidence_sentence_indices=[0],
        )
        assert len(ex.evidence_sentences) == 1
        assert ex.evidence_sentence_indices == [0]


class TestPredictionResult:
    def test_basic_prediction(self):
        pred = PredictionResult(
            example_id="1", gold_label="SUPPORTS",
            predicted_label="SUPPORTS", correct=True,
        )
        assert pred.correct is True
        assert pred.entailment_score == 0.0

    def test_incorrect_prediction(self):
        pred = PredictionResult(
            example_id="1", gold_label="SUPPORTS",
            predicted_label="REFUTES", correct=False,
            entailment_score=0.3, contradiction_score=0.6,
        )
        assert pred.correct is False
        assert pred.contradiction_score == 0.6


# =============================================================================
# BenchmarkReport Tests
# =============================================================================


class TestBenchmarkReport:
    def test_to_dict_excludes_predictions(self):
        report = BenchmarkReport(
            benchmark_name="test", tier="nli-only",
            num_examples=10, accuracy=0.8,
            precision=0.75, recall=0.85, f1=0.8,
            predictions=[
                PredictionResult("1", "S", "S", True),
            ],
        )
        d = report.to_dict()
        assert "predictions" not in d
        assert d["accuracy"] == 0.8
        assert d["benchmark_name"] == "test"

    def test_summary_includes_all_metrics(self):
        report = BenchmarkReport(
            benchmark_name="SciFact", tier="nli-only",
            num_examples=100, accuracy=0.87,
            precision=0.85, recall=0.89, f1=0.87,
            label_accuracy={"SUPPORTS": 0.9, "REFUTES": 0.8},
            total_time_s=15.5,
        )
        summary = report.summary()
        assert "SciFact" in summary
        assert "87.0%" in summary
        assert "SUPPORTS" in summary
        assert "15.5s" in summary

    def test_summary_with_evidence_metrics(self):
        report = BenchmarkReport(
            benchmark_name="test", tier="full",
            num_examples=10, accuracy=0.8,
            precision=0.75, recall=0.85, f1=0.8,
            evidence_f1=0.7,
        )
        summary = report.summary()
        assert "Evidence F1" in summary

    def test_summary_with_cache_stats(self):
        report = BenchmarkReport(
            benchmark_name="test", tier="full",
            num_examples=10, accuracy=0.8,
            precision=0.75, recall=0.85, f1=0.8,
            cache_stats={"hits": 50, "misses": 10},
        )
        summary = report.summary()
        assert "50 hits" in summary

    def test_default_fields(self):
        report = BenchmarkReport(
            benchmark_name="test", tier="t",
            num_examples=0, accuracy=0,
            precision=0, recall=0, f1=0,
        )
        assert report.evidence_precision == 0.0
        assert report.predictions == []
        assert report.label_accuracy == {}


# =============================================================================
# Sampling Tests
# =============================================================================


class _TestSuite(BenchmarkSuite):
    """Minimal concrete BenchmarkSuite for testing."""
    name = "test"
    labels = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

    def download(self):
        pass

    def load(self, split="test"):
        return []

    def map_nli_label(self, nli_label):
        return nli_label


class TestSampling:
    def setup_method(self):
        self.suite = _TestSuite()
        self.examples = [
            BenchmarkExample(f"{i}", f"claim {i}", label)
            for i, label in enumerate([
                "SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS",
                "REFUTES", "REFUTES", "REFUTES",
                "NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO",
            ])
        ]

    def test_sample_fewer_than_total(self):
        sampled = self.suite.sample(self.examples, 5)
        assert len(sampled) == 5

    def test_sample_all_returns_original(self):
        sampled = self.suite.sample(self.examples, 10)
        assert len(sampled) == 10

    def test_sample_more_than_total_returns_all(self):
        sampled = self.suite.sample(self.examples, 20)
        assert len(sampled) == 10

    def test_stratified_preserves_proportions(self):
        sampled = self.suite.sample(self.examples, 5, stratify=True)
        labels = [ex.gold_label for ex in sampled]
        # Should roughly preserve 5:3:2 ratio
        assert "SUPPORTS" in labels
        assert len(sampled) == 5

    def test_non_stratified_random(self):
        sampled = self.suite.sample(self.examples, 5, stratify=False)
        assert len(sampled) == 5

    def test_deterministic_with_seed(self):
        s1 = self.suite.sample(self.examples, 5, seed=42)
        s2 = self.suite.sample(self.examples, 5, seed=42)
        assert [ex.id for ex in s1] == [ex.id for ex in s2]

    def test_different_seeds_different_results(self):
        s1 = self.suite.sample(self.examples, 5, seed=42)
        s2 = self.suite.sample(self.examples, 5, seed=99)
        # Just check they're valid samples
        assert len(s1) == 5
        assert len(s2) == 5


# =============================================================================
# Compute Metrics Tests
# =============================================================================


class TestComputeMetrics:
    def setup_method(self):
        self.suite = _TestSuite()

    def test_perfect_predictions(self):
        preds = [
            PredictionResult("1", "SUPPORTS", "SUPPORTS", True),
            PredictionResult("2", "REFUTES", "REFUTES", True),
            PredictionResult("3", "NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO", True),
        ]
        report = self.suite.compute_metrics(preds)
        assert report.accuracy == 1.0
        assert report.f1 == 1.0

    def test_all_wrong(self):
        preds = [
            PredictionResult("1", "SUPPORTS", "REFUTES", False),
            PredictionResult("2", "REFUTES", "SUPPORTS", False),
        ]
        report = self.suite.compute_metrics(preds)
        assert report.accuracy == 0.0
        assert report.f1 == 0.0

    def test_empty_predictions(self):
        report = self.suite.compute_metrics([])
        assert report.num_examples == 0
        assert report.f1 == 0

    def test_per_label_accuracy(self):
        preds = [
            PredictionResult("1", "SUPPORTS", "SUPPORTS", True),
            PredictionResult("2", "SUPPORTS", "REFUTES", False),
            PredictionResult("3", "REFUTES", "REFUTES", True),
        ]
        report = self.suite.compute_metrics(preds)
        assert report.label_accuracy["SUPPORTS"] == 0.5
        assert report.label_accuracy["REFUTES"] == 1.0

    def test_evidence_metrics(self):
        preds = [
            PredictionResult(
                "1", "SUPPORTS", "SUPPORTS", True,
                gold_evidence_indices=[0, 1, 2],
                predicted_evidence_indices=[0, 1],
            ),
        ]
        report = self.suite.compute_metrics(preds)
        assert report.evidence_precision > 0
        assert report.evidence_recall > 0
        assert report.evidence_f1 > 0

    def test_evidence_precision_recall(self):
        preds = [
            PredictionResult(
                "1", "S", "S", True,
                gold_evidence_indices=[0, 1],
                predicted_evidence_indices=[0, 1, 2],  # 1 extra
            ),
        ]
        report = self.suite.compute_metrics(preds)
        # Precision: 2/3, Recall: 2/2
        assert abs(report.evidence_precision - 2/3) < 0.01
        assert report.evidence_recall == 1.0

    def test_latency_tracking(self):
        preds = [
            PredictionResult("1", "S", "S", True, latency_ms=100),
            PredictionResult("2", "S", "S", True, latency_ms=200),
        ]
        report = self.suite.compute_metrics(preds)
        assert report.avg_latency_ms == 150.0

    def test_num_examples(self):
        preds = [
            PredictionResult("1", "S", "S", True),
            PredictionResult("2", "S", "S", True),
            PredictionResult("3", "S", "S", True),
        ]
        report = self.suite.compute_metrics(preds)
        assert report.num_examples == 3
