"""
AttributionBench benchmark suite.

AttributionBench (OSU NLP Group, ACL 2024 Findings): A comprehensive
benchmark for attribution evaluation, testing whether claims are fully
supported by cited evidence passages.

Each example has:
    - A claim (one or more sentences from a generated response)
    - Reference passages cited as evidence
    - Binary label: attributable or not attributable

Published baselines (OOD macro-F1, Li et al. 2024 Table 3):
    Fine-tuned GPT-3.5:   81.9%
    GPT-4 zero-shot w/ CoT: 78.9%
    GPT-4 zero-shot w/o CoT: 78.0%

What we measure:
    - Attribution accuracy: is the claim supported by the cited references?
    - Macro-F1 across attributable/not-attributable classes
    - Separate evaluation on in-distribution (ID) and out-of-distribution (OOD) test sets

The OOD test set includes HAGRID examples, allowing direct comparison
with our HAGRID results.

Source: https://osu-nlp-group.github.io/AttributionBench/
Paper: https://aclanthology.org/2024.findings-acl.886.pdf
Data: https://huggingface.co/datasets/osunlp/AttributionBench
"""

import json
import logging

from .base import BenchmarkExample, BenchmarkSuite, PredictionResult
from .nli import NLIEvaluator, split_sentences

logger = logging.getLogger(__name__)


class AttributionBench(BenchmarkSuite):
    name = "attribution_bench"
    description = "Attribution evaluation for claims against cited evidence"
    labels = ["ATTRIBUTABLE", "NOT_ATTRIBUTABLE"]
    source_url = "https://huggingface.co/datasets/osunlp/AttributionBench"

    def download(self) -> None:
        from .download import download_attribution_bench
        download_attribution_bench()

    def load(self, split: str = "test") -> list[BenchmarkExample]:
        """
        Load AttributionBench examples.

        Splits:
            test     - Combined ID + OOD test sets (default)
            test_id  - In-distribution test set only
            test_ood - Out-of-distribution test set only (includes HAGRID)
            train    - Training set
            dev      - Development set
        """
        if split == "all_test":
            # Load both ID and OOD test sets
            examples = []
            for sub_split in ("test", "test_ood"):
                examples.extend(self._load_split(sub_split))
            return examples
        return self._load_split(split)

    def _load_split(self, split: str) -> list[BenchmarkExample]:
        """Load a single split file."""
        data_path = self.data_dir / f"{split}.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(
                f"AttributionBench data not found at {data_path}. "
                "Run: python -m benchmarks.download attribution_bench"
            )

        examples = []
        with open(data_path) as f:
            for line in f:
                item = json.loads(line)
                ex = self._parse_item(item, len(examples), split)
                if ex is not None:
                    examples.append(ex)

        logger.info(f"Loaded AttributionBench {split}: {len(examples)} examples")
        label_dist: dict[str, int] = {}
        for ex in examples:
            label_dist[ex.gold_label] = label_dist.get(ex.gold_label, 0) + 1
        logger.info(f"  Label distribution: {label_dist}")
        return examples

    def _parse_item(
        self, item: dict, idx: int, split: str
    ) -> BenchmarkExample | None:
        """Parse a single AttributionBench item."""
        # Get claim text
        claim = item.get("claim", item.get("claim_raw_string", ""))
        if not claim:
            return None

        # Get references (evidence passages)
        references = item.get("references", [])
        if isinstance(references, str):
            references = [references]

        # Flatten references into evidence sentences
        evidence: list[str] = []
        for ref in references:
            if isinstance(ref, str):
                evidence.append(ref)
            elif isinstance(ref, dict):
                evidence.append(ref.get("text", ref.get("content", str(ref))))

        if not evidence:
            return None

        # Get label
        label_raw = item.get("attribution_label", item.get("label", ""))
        if isinstance(label_raw, str):
            label_raw = label_raw.strip().lower()
        gold_label = self._normalize_label(label_raw)
        if gold_label is None:
            return None

        # Get source dataset info
        src_dataset = item.get("src_dataset", "")
        question = item.get("question", "")
        response = item.get("response", "")

        return BenchmarkExample(
            id=f"attrbench_{split}_{idx}",
            claim_or_query=question if question else claim,
            gold_label=gold_label,
            evidence_sentences=evidence,
            answer_text=claim,
            full_source_text="\n\n".join(evidence),
            metadata={
                "src_dataset": src_dataset,
                "response": response,
                "split": split,
                "references": references,
            },
        )

    def _normalize_label(self, label: str | int) -> str | None:
        """Normalize label to ATTRIBUTABLE/NOT_ATTRIBUTABLE."""
        if isinstance(label, int):
            return "ATTRIBUTABLE" if label == 1 else "NOT_ATTRIBUTABLE"
        label_str = str(label).strip().lower()
        if label_str in ("attributable", "attributed", "1", "true", "yes", "supported"):
            return "ATTRIBUTABLE"
        if label_str in (
            "not attributable", "not_attributable", "not attributed",
            "0", "false", "no", "not supported", "extrapolatory",
            "contradictory",
        ):
            return "NOT_ATTRIBUTABLE"
        logger.warning(f"Unknown AttributionBench label: {label}")
        return None

    def map_nli_label(self, nli_label: str) -> str:
        if nli_label in ("SUPPORTS",):
            return "ATTRIBUTABLE"
        return "NOT_ATTRIBUTABLE"

    def run_nli_only(
        self,
        examples: list[BenchmarkExample],
        nli: NLIEvaluator,
        entailment_threshold: float = 0.5,
        use_rerank: bool = False,
        use_confidence_margin: bool = False,
        use_minicheck_fallback: bool = False,
        fallback_model: str | None = None,
        use_passage_scoring: bool = False,
        attribution_strategy: str = "whole",
        decompose_evidence: bool = False,
        **kwargs,
    ) -> list[PredictionResult]:
        """
        Run NLI-only attribution check on AttributionBench examples.

        Default strategy is 'whole' since AttributionBench claims are
        typically 1-3 sentences and evidence is already pre-selected.

        attribution_strategy controls scoring:
            "whole"     - Check entire claim against concatenated evidence (default)
            "majority"  - >50% of claim sentences must be supported
        """
        import time

        predictions = []

        for ex in examples:
            start = time.time()

            if not ex.answer_text or not ex.evidence_sentences:
                continue

            concat_evidence = " ".join(ex.evidence_sentences)

            if attribution_strategy == "whole":
                result = nli.classify_claim(
                    claim=ex.answer_text,
                    evidence_sentences=[concat_evidence],
                    entailment_threshold=entailment_threshold,
                    use_rerank=use_rerank,
                    use_confidence_margin=use_confidence_margin,
                    use_minicheck_fallback=use_minicheck_fallback,
                    fallback_model=fallback_model,
                )
                max_ent = result["entailment"]
                max_con = result["contradiction"]
                pred_label = (
                    "ATTRIBUTABLE" if max_ent >= entailment_threshold
                    else "NOT_ATTRIBUTABLE"
                )
                details = {"strategy": "whole", "entailment": max_ent}

            else:
                # Sentence-level majority
                pred_label, max_ent, max_con, details = self._sentence_majority(
                    ex, nli, entailment_threshold,
                    use_minicheck_fallback, fallback_model,
                )

            elapsed_ms = int((time.time() - start) * 1000)

            predictions.append(PredictionResult(
                example_id=ex.id,
                gold_label=ex.gold_label,
                predicted_label=pred_label,
                correct=pred_label == ex.gold_label,
                entailment_score=max_ent,
                contradiction_score=max_con,
                latency_ms=elapsed_ms,
                tier="nli-only",
                details=details,
            ))

        return predictions

    def _sentence_majority(
        self, ex: BenchmarkExample, nli: NLIEvaluator,
        threshold: float, use_minicheck_fallback: bool,
        fallback_model: str | None,
    ) -> tuple[str, float, float, dict]:
        """Majority of claim sentences must be supported."""
        claim_sents = split_sentences(ex.answer_text)

        ent_scores = []
        for sent in claim_sents:
            result = nli.classify_claim(
                claim=sent,
                evidence_sentences=ex.evidence_sentences,
                entailment_threshold=threshold,
                use_minicheck_fallback=use_minicheck_fallback,
                fallback_model=fallback_model,
            )
            ent_scores.append(result["entailment"])

        if not ent_scores:
            return "NOT_ATTRIBUTABLE", 0.0, 0.0, {"strategy": "majority"}

        supported = sum(1 for e in ent_scores if e >= threshold)
        ratio = supported / len(ent_scores)
        max_ent = max(ent_scores)
        pred_label = "ATTRIBUTABLE" if ratio >= 0.5 else "NOT_ATTRIBUTABLE"
        return pred_label, max_ent, 0.0, {
            "strategy": "majority",
            "supported_ratio": ratio,
            "num_sents": len(ent_scores),
        }
