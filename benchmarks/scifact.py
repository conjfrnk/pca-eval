"""
SciFact benchmark suite.

SciFact: 1,409 expert-annotated scientific claims verified against
5,183 abstracts from S2ORC. This is a key verification benchmark
because it directly tests the NLI verification pipeline on the exact
task the verification pipeline performs.

Labels: SUPPORT, CONTRADICT, NOT_ENOUGH_INFO

Published baselines (from Wadden et al. 2022, Table 2):
    MultiVerS (Wadden 2022):         72.5 F1  (abstract-level)
    MultiVerS (Wadden 2022):         67.2 F1  (sentence-level)

What we measure:
    - Label accuracy (SUPPORT/CONTRADICT/NEI classification)
    - Rationale sentence selection (evidence F1)
    - Combined: correct label + correct rationale (the SciFact metric)
"""

import json
import logging
import time

from .base import BenchmarkExample, BenchmarkSuite, PredictionResult
from .nli import NLIEvaluator

logger = logging.getLogger(__name__)


class SciFact(BenchmarkSuite):
    name = "scifact"
    description = "Claim verification against scientific abstracts"
    labels = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
    source_url = "https://github.com/allenai/scifact"

    def __init__(self):
        self._corpus: dict[str, dict] = {}

    def download(self) -> None:
        from .download import download_scifact
        download_scifact()

    def _load_corpus(self) -> dict[str, dict]:
        """Load corpus of abstracts. Returns {doc_id: {title, abstract_sentences}}."""
        if self._corpus:
            return self._corpus

        corpus_path = self.data_dir / "corpus.jsonl"
        if not corpus_path.exists():
            # Try HF format
            corpus_path = self.data_dir / "corpus_hf.jsonl"

        if not corpus_path.exists():
            raise FileNotFoundError(
                f"SciFact corpus not found at {self.data_dir}. "
                "Run: python -m benchmarks.download scifact"
            )

        corpus = {}
        with open(corpus_path) as f:
            for line in f:
                doc = json.loads(line)
                doc_id = str(doc.get("doc_id", doc.get("id", "")))
                abstract = doc.get("abstract", [])
                if isinstance(abstract, str):
                    abstract = [abstract]
                corpus[doc_id] = {
                    "title": doc.get("title", ""),
                    "abstract": abstract,
                }

        self._corpus = corpus
        logger.info(f"Loaded SciFact corpus: {len(corpus)} abstracts")
        return corpus

    def load(self, split: str = "dev") -> list[BenchmarkExample]:
        """
        Load SciFact claims with gold evidence.

        Args:
            split: "dev" (default, 300 claims with labels) or "train" (809 claims)
        """
        corpus = self._load_corpus()

        claims_path = self.data_dir / f"claims_{split}.jsonl"
        if not claims_path.exists():
            claims_path = self.data_dir / f"claims_{split}_hf.jsonl"
        if not claims_path.exists():
            raise FileNotFoundError(f"SciFact claims not found: {claims_path}")

        examples = []
        with open(claims_path) as f:
            for line in f:
                claim_data = json.loads(line)
                claim_id = str(claim_data.get("id", ""))
                claim_text = claim_data.get("claim", "")
                evidence = claim_data.get("evidence", {})

                if not evidence:
                    # No evidence = NOT_ENOUGH_INFO
                    examples.append(BenchmarkExample(
                        id=claim_id,
                        claim_or_query=claim_text,
                        gold_label="NOT_ENOUGH_INFO",
                        evidence_sentences=[],
                        evidence_sentence_indices=[],
                    ))
                    continue

                # Process each evidence document
                for doc_id_str, doc_evidence_list in evidence.items():
                    doc = corpus.get(doc_id_str, {})
                    abstract_sents = doc.get("abstract", [])
                    full_text = " ".join(abstract_sents)

                    for annotation in doc_evidence_list:
                        sent_indices = annotation.get("sentences", [])
                        label = annotation.get("label", "")

                        # Map SciFact labels to our format
                        if label == "SUPPORT":
                            mapped_label = "SUPPORTS"
                        elif label == "CONTRADICT":
                            mapped_label = "REFUTES"
                        else:
                            mapped_label = "NOT_ENOUGH_INFO"

                        evidence_sents = [
                            abstract_sents[i] for i in sent_indices
                            if i < len(abstract_sents)
                        ]

                        examples.append(BenchmarkExample(
                            id=f"{claim_id}_{doc_id_str}",
                            claim_or_query=claim_text,
                            gold_label=mapped_label,
                            evidence_sentences=evidence_sents,
                            evidence_sentence_indices=sent_indices,
                            source_doc_id=doc_id_str,
                            source_doc_title=doc.get("title", ""),
                            full_source_text=full_text,
                            metadata={"all_abstract_sentences": abstract_sents},
                        ))

        logger.info(f"Loaded SciFact {split}: {len(examples)} examples")
        label_dist = {}
        for ex in examples:
            label_dist[ex.gold_label] = label_dist.get(ex.gold_label, 0) + 1
        logger.info(f"  Label distribution: {label_dist}")
        return examples

    def map_nli_label(self, nli_label: str) -> str:
        mapping = {
            "SUPPORTS": "SUPPORTS",
            "REFUTES": "REFUTES",
            "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
        }
        return mapping.get(nli_label, "NOT_ENOUGH_INFO")

    def run_nli_only(
        self,
        examples: list[BenchmarkExample],
        nli: NLIEvaluator,
        entailment_threshold: float = 0.5,
        contradiction_threshold: float = 0.5,
        use_rerank: bool = False,
        use_confidence_margin: bool = False,
        use_minicheck_fallback: bool = False,
        fallback_model: str | None = None,
        use_passage_scoring: bool = False,
        **kwargs,
    ) -> list[PredictionResult]:
        """
        Run NLI-only (Tier 1, $0 cost).

        For each claim with gold evidence, run the DeBERTa model to classify.
        For claims without evidence (NEI), skip NLI and predict NEI.
        """
        predictions = []

        for ex in examples:
            start = time.time()

            if not ex.evidence_sentences and not ex.metadata.get("all_abstract_sentences"):
                # No evidence at all - predict NEI
                predictions.append(PredictionResult(
                    example_id=ex.id,
                    gold_label=ex.gold_label,
                    predicted_label="NOT_ENOUGH_INFO",
                    correct=ex.gold_label == "NOT_ENOUGH_INFO",
                    tier="nli-only",
                ))
                continue

            # Use gold evidence sentences for oracle mode
            evidence = ex.evidence_sentences
            if not evidence:
                evidence = ex.metadata.get("all_abstract_sentences", [])

            result = nli.classify_claim(
                claim=ex.claim_or_query,
                evidence_sentences=evidence,
                entailment_threshold=entailment_threshold,
                contradiction_threshold=contradiction_threshold,
                use_rerank=use_rerank,
                use_confidence_margin=use_confidence_margin,
                use_minicheck_fallback=use_minicheck_fallback,
                fallback_model=fallback_model,
                use_passage_scoring=use_passage_scoring,
            )

            pred_label = self.map_nli_label(result["label"])
            elapsed_ms = int((time.time() - start) * 1000)

            predictions.append(PredictionResult(
                example_id=ex.id,
                gold_label=ex.gold_label,
                predicted_label=pred_label,
                correct=pred_label == ex.gold_label,
                entailment_score=result["entailment"],
                contradiction_score=result["contradiction"],
                neutral_score=result["neutral"],
                predicted_evidence_indices=result["supporting_sentences"],
                gold_evidence_indices=ex.evidence_sentence_indices,
                latency_ms=elapsed_ms,
                tier="nli-only",
            ))

        return predictions

    def run_nli_abstract_retrieval(
        self,
        examples: list[BenchmarkExample],
        nli: NLIEvaluator,
        entailment_threshold: float = 0.5,
        contradiction_threshold: float = 0.5,
        use_rerank: bool = False,
        use_context_window: bool = False,
        use_confidence_margin: bool = False,
        use_minicheck_fallback: bool = False,
        minicheck_fallback_threshold: float = 0.5,
        fallback_model: str | None = None,
    ) -> list[PredictionResult]:
        """
        Run NLI against ALL abstract sentences (still $0).

        Instead of using gold evidence sentences, runs NLI against every
        sentence in the abstract. Tests both NLI accuracy and implicit
        rationale selection. Closer to real-world usage.
        """
        predictions = []

        for ex in examples:
            start = time.time()

            all_sents = ex.metadata.get("all_abstract_sentences", [])
            if not all_sents and ex.evidence_sentences:
                all_sents = ex.evidence_sentences

            if not all_sents:
                predictions.append(PredictionResult(
                    example_id=ex.id,
                    gold_label=ex.gold_label,
                    predicted_label="NOT_ENOUGH_INFO",
                    correct=ex.gold_label == "NOT_ENOUGH_INFO",
                    tier="nli-abstract",
                ))
                continue

            result = nli.classify_claim(
                claim=ex.claim_or_query,
                evidence_sentences=all_sents,
                entailment_threshold=entailment_threshold,
                contradiction_threshold=contradiction_threshold,
                use_context_window=use_context_window,
                all_sentences=all_sents,
                use_rerank=use_rerank,
                use_confidence_margin=use_confidence_margin,
                use_minicheck_fallback=use_minicheck_fallback,
                minicheck_fallback_threshold=minicheck_fallback_threshold,
                fallback_model=fallback_model,
            )

            pred_label = self.map_nli_label(result["label"])
            elapsed_ms = int((time.time() - start) * 1000)

            predictions.append(PredictionResult(
                example_id=ex.id,
                gold_label=ex.gold_label,
                predicted_label=pred_label,
                correct=pred_label == ex.gold_label,
                entailment_score=result["entailment"],
                contradiction_score=result["contradiction"],
                neutral_score=result["neutral"],
                predicted_evidence_indices=result["supporting_sentences"],
                gold_evidence_indices=ex.evidence_sentence_indices,
                latency_ms=elapsed_ms,
                tier="nli-abstract",
            ))

        return predictions
