"""
Base class for academic benchmark suites.

Provides tiered execution, result collection, and reporting.
"""

import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class BenchmarkExample:
    """Single example from a benchmark dataset."""
    id: str
    claim_or_query: str
    gold_label: str                              # Dataset-specific label
    evidence_sentences: list[str] = field(default_factory=list)   # Gold evidence text
    evidence_sentence_indices: list[int] = field(default_factory=list)
    source_doc_id: str = ""
    source_doc_title: str = ""
    full_source_text: str = ""                   # Full document/abstract text
    answer_text: str = ""                        # For QA benchmarks
    metadata: dict = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of running a single benchmark example through the pipeline."""
    example_id: str
    gold_label: str
    predicted_label: str
    correct: bool
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    neutral_score: float = 0.0
    predicted_evidence_indices: list[int] = field(default_factory=list)
    gold_evidence_indices: list[int] = field(default_factory=list)
    latency_ms: int = 0
    tier: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Aggregate results for a benchmark run."""
    benchmark_name: str
    tier: str
    num_examples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    label_accuracy: dict = field(default_factory=dict)  # Per-label accuracy
    evidence_precision: float = 0.0
    evidence_recall: float = 0.0
    evidence_f1: float = 0.0
    avg_latency_ms: float = 0.0
    total_time_s: float = 0.0
    cache_stats: dict = field(default_factory=dict)
    predictions: list[PredictionResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("predictions")
        return d

    def summary(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"  {self.benchmark_name} | tier={self.tier} | n={self.num_examples}",
            f"{'=' * 60}",
            f"  Accuracy:    {self.accuracy:.1%}",
            f"  Precision:   {self.precision:.1%}",
            f"  Recall:      {self.recall:.1%}",
            f"  F1:          {self.f1:.1%}",
        ]
        if self.label_accuracy:
            lines.append("  Per-label accuracy:")
            for label, acc in sorted(self.label_accuracy.items()):
                lines.append(f"    {label:20s}: {acc:.1%}")
        if self.evidence_f1 > 0:
            lines.append(f"  Evidence F1: {self.evidence_f1:.1%}")
        if self.avg_latency_ms > 0:
            lines.append(f"  Avg latency: {self.avg_latency_ms:.0f}ms")
        lines.append(f"  Total time:  {self.total_time_s:.1f}s")
        if self.cache_stats:
            lines.append(f"  Cache:       {self.cache_stats.get('hits', 0)} hits, {self.cache_stats.get('misses', 0)} misses")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    def save(self, path: Path | None = None) -> Path:
        RESULTS_DIR.mkdir(exist_ok=True)
        if path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = RESULTS_DIR / f"{self.benchmark_name}_{self.tier}_{timestamp}.json"
        data = self.to_dict()
        data["predictions"] = [asdict(p) for p in self.predictions]
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Results saved to {path}")
        return path


class BenchmarkSuite(ABC):
    """
    Base class for academic benchmark suites.

    Subclasses implement dataset loading and label mapping.
    The base class handles sampling, reporting, and tiered execution.
    """

    name: str = ""
    description: str = ""
    labels: list[str] = []          # e.g. ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
    source_url: str = ""

    @abstractmethod
    def download(self) -> None:
        """Download dataset files to DATA_DIR / self.name."""

    @abstractmethod
    def load(self, split: str = "test") -> list[BenchmarkExample]:
        """Load and return benchmark examples."""

    @abstractmethod
    def map_nli_label(self, nli_label: str) -> str:
        """Map NLI output label to this benchmark's label scheme."""

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / self.name

    def is_downloaded(self) -> bool:
        return self.data_dir.exists() and any(self.data_dir.iterdir())

    def sample(
        self,
        examples: list[BenchmarkExample],
        n: int,
        seed: int = 42,
        stratify: bool = True,
    ) -> list[BenchmarkExample]:
        """
        Sample n examples, optionally stratified by label.

        Stratified sampling ensures each label is proportionally represented.
        """
        if n >= len(examples):
            return examples

        rng = random.Random(seed)

        if not stratify:
            return rng.sample(examples, n)

        by_label: dict[str, list[BenchmarkExample]] = {}
        for ex in examples:
            by_label.setdefault(ex.gold_label, []).append(ex)

        sampled = []
        for _label, group in by_label.items():
            label_n = max(1, round(n * len(group) / len(examples)))
            sampled.extend(rng.sample(group, min(label_n, len(group))))

        # If rounding caused overshoot/undershoot, adjust
        if len(sampled) > n:
            sampled = rng.sample(sampled, n)
        elif len(sampled) < n:
            remaining = [ex for ex in examples if ex not in sampled]
            sampled.extend(rng.sample(remaining, n - len(sampled)))

        return sampled

    def compute_metrics(self, predictions: list[PredictionResult]) -> BenchmarkReport:
        """Compute aggregate metrics from predictions."""
        if not predictions:
            return BenchmarkReport(
                benchmark_name=self.name, tier="", num_examples=0,
                accuracy=0, precision=0, recall=0, f1=0,
            )

        correct = sum(1 for p in predictions if p.correct)
        accuracy = correct / len(predictions)

        # Per-label metrics
        label_correct: dict[str, int] = {}
        label_total: dict[str, int] = {}
        for p in predictions:
            label_total[p.gold_label] = label_total.get(p.gold_label, 0) + 1
            if p.correct:
                label_correct[p.gold_label] = label_correct.get(p.gold_label, 0) + 1

        label_accuracy = {
            label: label_correct.get(label, 0) / total
            for label, total in label_total.items()
        }

        # Macro-averaged precision/recall/F1 across labels
        precisions, recalls = [], []
        for label in self.labels:
            tp = sum(1 for p in predictions if p.predicted_label == label and p.gold_label == label)
            fp = sum(1 for p in predictions if p.predicted_label == label and p.gold_label != label)
            fn = sum(1 for p in predictions if p.predicted_label != label and p.gold_label == label)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)

        macro_prec = sum(precisions) / len(precisions) if precisions else 0.0
        macro_rec = sum(recalls) / len(recalls) if recalls else 0.0
        macro_f1 = 2 * macro_prec * macro_rec / (macro_prec + macro_rec) if (macro_prec + macro_rec) > 0 else 0.0

        # Evidence metrics (for benchmarks that have evidence annotations)
        ev_preds_with_evidence = [p for p in predictions if p.gold_evidence_indices]
        ev_prec, ev_rec, ev_f1 = 0.0, 0.0, 0.0
        if ev_preds_with_evidence:
            ev_precisions, ev_recalls = [], []
            for p in ev_preds_with_evidence:
                gold = set(p.gold_evidence_indices)
                pred = set(p.predicted_evidence_indices)
                if pred:
                    ev_precisions.append(len(gold & pred) / len(pred))
                if gold:
                    ev_recalls.append(len(gold & pred) / len(gold))
            ev_prec = sum(ev_precisions) / len(ev_precisions) if ev_precisions else 0.0
            ev_rec = sum(ev_recalls) / len(ev_recalls) if ev_recalls else 0.0
            ev_f1 = 2 * ev_prec * ev_rec / (ev_prec + ev_rec) if (ev_prec + ev_rec) > 0 else 0.0

        latencies = [p.latency_ms for p in predictions if p.latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        return BenchmarkReport(
            benchmark_name=self.name,
            tier="",
            num_examples=len(predictions),
            accuracy=accuracy,
            precision=macro_prec,
            recall=macro_rec,
            f1=macro_f1,
            label_accuracy=label_accuracy,
            evidence_precision=ev_prec,
            evidence_recall=ev_rec,
            evidence_f1=ev_f1,
            avg_latency_ms=avg_latency,
            predictions=predictions,
        )
