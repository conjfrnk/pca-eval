"""
AIS (Attributable to Identified Sources) metric.

AIS is a framework for measuring whether generated text is actually
supported by cited sources. Unlike SciFact/FEVER (which are datasets),
AIS is a metric that can be applied to outputs from any system.

Implements the academic AIS framework:
every claim should be attributable to an identified source.

Reference: Rashkin et al. (2022) "Measuring Attribution in Natural
Language Generation Models"

AIS Score = fraction of generated statements that are attributable
to at least one provided source.

We implement AIS as a metric layer that can be applied on top of
any benchmark's outputs or on proof-carrying answer outputs.
"""

import logging
import re
from dataclasses import dataclass, field

from .nli import NLIEvaluator

logger = logging.getLogger(__name__)


@dataclass
class AISExample:
    """A single example for AIS scoring."""
    id: str
    generated_text: str          # The text to check
    source_texts: list[str]      # The sources it should be attributable to
    # Optional: pre-segmented statements
    statements: list[str] = field(default_factory=list)


@dataclass
class AISResult:
    """AIS scoring result for a single example."""
    id: str
    num_statements: int
    num_attributable: int
    ais_score: float                          # fraction attributable
    per_statement: list[dict] = field(default_factory=list)


@dataclass
class AISReport:
    """Aggregate AIS scoring report."""
    num_examples: int
    mean_ais_score: float
    median_ais_score: float
    min_ais_score: float
    max_ais_score: float
    fully_attributable_rate: float            # fraction with AIS=1.0
    results: list[AISResult] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{'=' * 60}\n"
            f"  AIS (Attributable to Identified Sources)\n"
            f"{'=' * 60}\n"
            f"  Examples:               {self.num_examples}\n"
            f"  Mean AIS:               {self.mean_ais_score:.1%}\n"
            f"  Median AIS:             {self.median_ais_score:.1%}\n"
            f"  Min AIS:                {self.min_ais_score:.1%}\n"
            f"  Max AIS:                {self.max_ais_score:.1%}\n"
            f"  Fully Attributable:     {self.fully_attributable_rate:.1%}\n"
            f"{'=' * 60}"
        )


class AISScorer:
    """
    AIS metric scorer using NLI.

    Segments generated text into statements, then checks each statement
    for entailment against provided sources using the same DeBERTa model
    used for verification.
    """

    def __init__(
        self,
        nli: NLIEvaluator | None = None,
        entailment_threshold: float = 0.5,
    ):
        self.nli = nli or NLIEvaluator()
        self.threshold = entailment_threshold

    def segment_statements(self, text: str) -> list[str]:
        """
        Segment text into atomic statements.

        Simple sentence-level segmentation. More sophisticated claim
        decomposition approaches exist, but sentence-level segmentation
        is sufficient for benchmarking.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        statements = [s.strip() for s in sentences if len(s.strip()) > 10]
        return statements if statements else [text.strip()]

    def score_single(self, example: AISExample) -> AISResult:
        """Compute AIS score for a single example."""
        statements = example.statements or self.segment_statements(example.generated_text)

        if not statements or not example.source_texts:
            return AISResult(
                id=example.id,
                num_statements=len(statements),
                num_attributable=0,
                ais_score=0.0,
            )

        per_statement = []
        num_attributable = 0

        for stmt in statements:
            result = self.nli.classify_claim(
                claim=stmt,
                evidence_sentences=example.source_texts,
                entailment_threshold=self.threshold,
            )

            is_attributable = result["entailment"] >= self.threshold
            if is_attributable:
                num_attributable += 1

            per_statement.append({
                "statement": stmt,
                "attributable": is_attributable,
                "entailment": result["entailment"],
                "best_evidence_idx": (
                    result["supporting_sentences"][0]
                    if result["supporting_sentences"]
                    else -1
                ),
            })

        ais_score = num_attributable / len(statements) if statements else 0.0

        return AISResult(
            id=example.id,
            num_statements=len(statements),
            num_attributable=num_attributable,
            ais_score=ais_score,
            per_statement=per_statement,
        )

    def score_batch(self, examples: list[AISExample]) -> AISReport:
        """Compute AIS scores across a batch of examples."""
        results = [self.score_single(ex) for ex in examples]

        scores = [r.ais_score for r in results]
        if not scores:
            return AISReport(
                num_examples=0, mean_ais_score=0, median_ais_score=0,
                min_ais_score=0, max_ais_score=0, fully_attributable_rate=0,
            )

        sorted_scores = sorted(scores)
        median = sorted_scores[len(sorted_scores) // 2]

        return AISReport(
            num_examples=len(results),
            mean_ais_score=sum(scores) / len(scores),
            median_ais_score=median,
            min_ais_score=min(scores),
            max_ais_score=max(scores),
            fully_attributable_rate=sum(1 for s in scores if s >= 1.0) / len(scores),
            results=results,
        )

    def score_from_benchmark(
        self,
        benchmark_name: str,
        examples: list,
    ) -> AISReport:
        """
        Apply AIS metric to examples from another benchmark.

        Converts benchmark examples into AIS format and scores.
        Works with any benchmark that has answer_text + evidence_sentences.
        """
        ais_examples = []
        for ex in examples:
            if hasattr(ex, "answer_text") and ex.answer_text and hasattr(ex, "evidence_sentences") and ex.evidence_sentences:
                ais_examples.append(AISExample(
                    id=ex.id,
                    generated_text=ex.answer_text,
                    source_texts=ex.evidence_sentences,
                ))

        if not ais_examples:
            logger.warning(f"No examples with answer_text and evidence from {benchmark_name}")
            return AISReport(
                num_examples=0, mean_ais_score=0, median_ais_score=0,
                min_ais_score=0, max_ais_score=0, fully_attributable_rate=0,
            )

        logger.info(f"Running AIS on {len(ais_examples)} examples from {benchmark_name}")
        return self.score_batch(ais_examples)
