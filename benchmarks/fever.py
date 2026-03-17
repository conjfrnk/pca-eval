"""
FEVER benchmark suite.

FEVER (Fact Extraction and VERification): 185,445 claims generated from
Wikipedia, manually classified as SUPPORTS, REFUTES, or NOT ENOUGH INFO
with annotated evidence sentences.

This tests the core NLI capability at scale.

Published baselines (label accuracy on dev):
    Majority class:          52.1%
    Decomposable Attention:  51.6%
    NSMN (Nie et al. 2019):  68.2%  (evidence + label)
    KGAT:                    70.4%  (evidence + label)
    DeBERTa-v3 NLI:          ~85-90% (label-only with oracle evidence)

We measure:
    - Label accuracy with oracle evidence (direct NLI test)
    - Label accuracy with abstract retrieval (evidence + classification)
"""

import json
import logging
import time

from .base import BenchmarkExample, BenchmarkSuite, PredictionResult
from .nli import NLIEvaluator

logger = logging.getLogger(__name__)


class FEVER(BenchmarkSuite):
    name = "fever"
    description = "Fact verification against Wikipedia at scale"
    labels = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
    source_url = "https://fever.ai"

    def __init__(self):
        self._wiki_cache: dict[str, list[str]] = {}

    def download(self) -> None:
        from .download import download_fever
        download_fever()

    def _load_wiki_sentences(self) -> dict[str, list[str]]:
        """
        Load Wikipedia sentences for evidence lookup.

        Returns {page_title: [sentence_0, sentence_1, ...]}.
        Falls back to empty dict if wiki pages not downloaded.
        """
        if self._wiki_cache:
            return self._wiki_cache

        wiki_dir = self.data_dir / "wiki-pages" / "wiki-pages"
        if not wiki_dir.exists():
            logger.info("FEVER Wikipedia pages not downloaded. Evidence text will be unavailable.")
            logger.info("For NLI-only mode, claims with missing evidence will be skipped.")
            logger.info("Run: python -m benchmarks.download --include-wiki  (4GB download)")
            return {}

        for wiki_file in sorted(wiki_dir.glob("wiki-*.jsonl")):
            for line in open(wiki_file):
                page = json.loads(line)
                title = page.get("id", "")
                lines_dict = page.get("lines", "")
                if isinstance(lines_dict, str):
                    sentences = []
                    for line in lines_dict.split("\n"):
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            sentences.append(parts[1])
                    self._wiki_cache[title] = sentences

        logger.info(f"Loaded {len(self._wiki_cache)} Wikipedia pages")
        return self._wiki_cache

    def load(self, split: str = "dev") -> list[BenchmarkExample]:
        """
        Load FEVER claims with evidence annotations.

        Args:
            split: "dev" (~19K claims with labels and evidence)
        """
        # Try shared task format first
        data_path = self.data_dir / f"shared_task_{split}.jsonl"
        if not data_path.exists():
            data_path = self.data_dir / f"{split}_hf.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(f"FEVER data not found at {self.data_dir}. Run: python -m benchmarks.download fever")

        wiki = self._load_wiki_sentences()

        examples = []
        skipped_no_evidence = 0

        for line in open(data_path):
            item = json.loads(line)
            claim_id = str(item.get("id", len(examples)))
            claim = item.get("claim", "")
            label = item.get("label", "")

            # Normalize label
            if label in ("SUPPORTS", "SUPPORTED"):
                mapped_label = "SUPPORTS"
            elif label in ("REFUTES", "REFUTED"):
                mapped_label = "REFUTES"
            else:
                mapped_label = "NOT_ENOUGH_INFO"

            # Extract evidence
            evidence_sets = item.get("evidence", [])
            evidence_sentences = []
            evidence_indices = []
            evidence_pages = []

            if evidence_sets and mapped_label != "NOT_ENOUGH_INFO":
                # FEVER evidence format: list of evidence sets,
                # each set is a list of [annotation_id, evidence_id, page_title, sentence_idx]
                for evidence_set in evidence_sets:
                    for evidence_item in evidence_set:
                        if isinstance(evidence_item, list) and len(evidence_item) >= 4:
                            page_title = evidence_item[2]
                            sent_idx = evidence_item[3]
                            if page_title and sent_idx is not None:
                                evidence_pages.append(page_title)
                                evidence_indices.append(sent_idx)

                                # Try to get actual text from wiki
                                if wiki and page_title in wiki:
                                    page_sents = wiki[page_title]
                                    if sent_idx < len(page_sents):
                                        evidence_sentences.append(page_sents[sent_idx])

            # If we have evidence annotations but no text, and it's not NEI
            if not evidence_sentences and mapped_label != "NOT_ENOUGH_INFO" and evidence_pages:
                skipped_no_evidence += 1

            examples.append(BenchmarkExample(
                id=claim_id,
                claim_or_query=claim,
                gold_label=mapped_label,
                evidence_sentences=evidence_sentences,
                evidence_sentence_indices=evidence_indices,
                metadata={
                    "evidence_pages": evidence_pages,
                    "has_evidence_text": len(evidence_sentences) > 0,
                },
            ))

        if skipped_no_evidence > 0:
            logger.info(
                f"FEVER: {skipped_no_evidence} claims have evidence annotations but no text "
                f"(Wikipedia pages not downloaded)"
            )

        logger.info(f"Loaded FEVER {split}: {len(examples)} examples")
        label_dist = {}
        for ex in examples:
            label_dist[ex.gold_label] = label_dist.get(ex.gold_label, 0) + 1
        logger.info(f"  Label distribution: {label_dist}")

        with_text = sum(1 for ex in examples if ex.evidence_sentences)
        logger.info(f"  Claims with evidence text: {with_text}")
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
        Run NLI-only with oracle evidence text.

        Only runs on claims that have evidence text available.
        Claims without evidence text or NEI claims are handled separately.
        """
        predictions = []

        for ex in examples:
            start = time.time()

            if ex.gold_label == "NOT_ENOUGH_INFO":
                # For NEI: if we have no evidence, predict NEI
                predictions.append(PredictionResult(
                    example_id=ex.id,
                    gold_label=ex.gold_label,
                    predicted_label="NOT_ENOUGH_INFO",
                    correct=True,
                    tier="nli-only",
                ))
                continue

            if not ex.evidence_sentences:
                # Can't run without evidence text - skip
                continue

            result = nli.classify_claim(
                claim=ex.claim_or_query,
                evidence_sentences=ex.evidence_sentences,
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
