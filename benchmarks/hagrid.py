"""
HAGRID benchmark suite.

HAGRID (Human-Annotated Generative Retrieval-Augmented Inferences Dataset):
Measures whether generated answers in RAG systems are properly attributed
to their source passages. This directly measures proof-carrying answer quality.

Each example has:
    - A query
    - Retrieved passages
    - A generated answer
    - Human annotations of which answer sentences are attributable to which passages

Published baselines:
    Various RAG systems score 40-70% attribution accuracy.
    Proof-carrying answers should score significantly higher.

What we measure:
    - Attribution accuracy: is each answer sentence supported by cited passages?
    - Over-attribution: does the system claim support that doesn't exist?
    - Under-attribution: does the system miss valid support?
"""

import json
import logging
import time

from .base import BenchmarkExample, BenchmarkSuite, PredictionResult
from .nli import NLIEvaluator, split_sentences

logger = logging.getLogger(__name__)


class HAGRID(BenchmarkSuite):
    name = "hagrid"
    description = "Hallucination detection in RAG systems"
    labels = ["ATTRIBUTABLE", "NOT_ATTRIBUTABLE"]
    source_url = "https://huggingface.co/datasets/miracl/hagrid"

    def download(self) -> None:
        from .download import download_hagrid
        download_hagrid()

    def load(self, split: str = "dev") -> list[BenchmarkExample]:
        """
        Load HAGRID examples.

        HAGRID format varies; we handle both the original and HuggingFace formats.
        """
        data_path = self.data_dir / f"{split}.jsonl"
        if not data_path.exists():
            data_path = self.data_dir / "test.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(f"HAGRID data not found at {self.data_dir}. Run: python -m benchmarks.download hagrid")

        examples = []

        with open(data_path) as f:
            for line in f:
                item = json.loads(line)
                examples.extend(self._parse_item(item, len(examples)))

        logger.info(f"Loaded HAGRID: {len(examples)} examples")
        label_dist = {}
        for ex in examples:
            label_dist[ex.gold_label] = label_dist.get(ex.gold_label, 0) + 1
        logger.info(f"  Label distribution: {label_dist}")
        return examples

    def _parse_item(self, item: dict, base_idx: int) -> list[BenchmarkExample]:
        """Parse a single HAGRID item into one or more examples."""
        examples = []
        query = item.get("query", "")
        answers = item.get("answers", [])
        quotes = item.get("quotes", [])

        # Get passages (knowledge/documents)
        passages = []
        for k in ("knowledge", "passages", "documents"):
            if k in item:
                raw = item[k]
                if isinstance(raw, list):
                    for p in raw:
                        if isinstance(p, dict):
                            text = p.get("text", p.get("content", p.get("passage", "")))
                            title = p.get("title", "")
                            passages.append(f"{title}: {text}" if title else text)
                        elif isinstance(p, str):
                            passages.append(p)
                break

        if not passages and not answers:
            return examples

        # Process each answer
        for i, answer_data in enumerate(answers):
            if isinstance(answer_data, dict):
                answer_text = answer_data.get("answer", answer_data.get("text", ""))
                # Check attribution annotations
                attributable = answer_data.get("attributable", None)
                answer_quotes = answer_data.get("quotes", [])

                if attributable is not None:
                    gold_label = "ATTRIBUTABLE" if attributable else "NOT_ATTRIBUTABLE"
                elif answer_quotes:
                    gold_label = "ATTRIBUTABLE"
                else:
                    gold_label = "NOT_ATTRIBUTABLE"

                evidence = []
                if answer_quotes:
                    for q in answer_quotes:
                        if isinstance(q, str):
                            evidence.append(q)
                        elif isinstance(q, dict):
                            evidence.append(q.get("text", q.get("quote", str(q))))
                elif quotes:
                    for q in quotes:
                        if isinstance(q, str):
                            evidence.append(q)
                        elif isinstance(q, dict):
                            evidence.append(q.get("text", q.get("quote", str(q))))
            elif isinstance(answer_data, str):
                answer_text = answer_data
                gold_label = "ATTRIBUTABLE" if passages else "NOT_ATTRIBUTABLE"
                evidence = []
            else:
                continue

            if not answer_text:
                continue

            # Use passages as evidence if no specific quotes
            if not evidence:
                evidence = passages

            examples.append(BenchmarkExample(
                id=f"hagrid_{base_idx + i}",
                claim_or_query=query,
                gold_label=gold_label,
                evidence_sentences=evidence,
                answer_text=answer_text,
                full_source_text="\n\n".join(passages),
                metadata={
                    "passages": passages,
                    "num_passages": len(passages),
                },
            ))

        return examples

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
        attribution_strategy: str = "majority",
        **kwargs,
    ) -> list[PredictionResult]:
        """
        Run NLI-only attribution check.

        For each answer, check if it's entailed by the provided passages.

        attribution_strategy controls how scores are computed and aggregated:
            "majority"          - >50% of answer sentences must be supported (default)
            "whole"             - Check entire answer against concatenated evidence
            "max-score"         - Use max entailment score across sentences
            "any-sentence"      - Attributable if ANY sentence is strongly supported
            "mean-score"        - Use mean entailment score across sentences
            "weighted"          - Length-weighted mean across sentences
        """
        predictions = []

        for ex in examples:
            start = time.time()

            if not ex.answer_text or not ex.evidence_sentences:
                continue

            if attribution_strategy == "whole":
                pred_label, max_ent, max_con, details = _strategy_whole(
                    ex, nli, entailment_threshold, use_confidence_margin,
                    use_minicheck_fallback, fallback_model,
                )

            else:
                # Sentence-level strategies (majority, max-score, any-sentence, mean-score, weighted)
                pred_label, max_ent, max_con, details = _strategy_sentence_level(
                    ex, nli, entailment_threshold, attribution_strategy,
                    use_rerank, use_confidence_margin,
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


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _strategy_whole(
    ex, nli, threshold, use_confidence_margin, use_minicheck_fallback, fallback_model,
):
    """Check entire answer as one claim against concatenated evidence."""
    concat_evidence = " ".join(ex.evidence_sentences)
    result = nli.classify_claim(
        claim=ex.answer_text,
        evidence_sentences=[concat_evidence],
        entailment_threshold=threshold,
        use_rerank=False,
        use_confidence_margin=use_confidence_margin,
        use_minicheck_fallback=use_minicheck_fallback,
        fallback_model=fallback_model,
    )
    max_ent = result["entailment"]
    max_con = result["contradiction"]
    pred_label = "ATTRIBUTABLE" if max_ent >= threshold else "NOT_ATTRIBUTABLE"
    return pred_label, max_ent, max_con, {"strategy": "whole", "entailment": max_ent}


def _strategy_sentence_level(
    ex, nli, threshold, strategy_name,
    use_rerank, use_confidence_margin,
    use_minicheck_fallback, fallback_model,
):
    """Sentence-level strategies: majority, max-score, any-sentence, mean-score, weighted."""
    answer_sentences = _split_sentences(ex.answer_text)

    sentence_results = []
    for sent in answer_sentences:
        result = nli.classify_claim(
            claim=sent,
            evidence_sentences=ex.evidence_sentences,
            entailment_threshold=threshold,
            use_rerank=use_rerank,
            use_confidence_margin=use_confidence_margin,
            use_minicheck_fallback=use_minicheck_fallback,
            fallback_model=fallback_model,
        )
        sentence_results.append(result)

    if not sentence_results:
        return "NOT_ATTRIBUTABLE", 0.0, 0.0, {"strategy": strategy_name}

    ent_scores = [r["entailment"] for r in sentence_results]
    max_ent = max(ent_scores)
    max_con = max(r["contradiction"] for r in sentence_results)

    if strategy_name == "max-score":
        pred_label = "ATTRIBUTABLE" if max_ent >= threshold else "NOT_ATTRIBUTABLE"
        details = {"strategy": "max-score", "max_ent": max_ent}

    elif strategy_name == "any-sentence":
        high_threshold = max(threshold, 0.7)
        any_strong = any(e >= high_threshold for e in ent_scores)
        pred_label = "ATTRIBUTABLE" if any_strong else "NOT_ATTRIBUTABLE"
        details = {"strategy": "any-sentence", "any_strong": any_strong}

    elif strategy_name == "mean-score":
        mean_ent = sum(ent_scores) / len(ent_scores)
        pred_label = "ATTRIBUTABLE" if mean_ent >= threshold else "NOT_ATTRIBUTABLE"
        details = {"strategy": "mean-score", "mean_ent": mean_ent}

    elif strategy_name == "weighted":
        weights = [len(s.split()) for s in answer_sentences]
        total_w = sum(weights) or 1
        weighted_ent = sum(e * w for e, w in zip(ent_scores, weights, strict=False)) / total_w
        pred_label = "ATTRIBUTABLE" if weighted_ent >= threshold else "NOT_ATTRIBUTABLE"
        details = {"strategy": "weighted", "weighted_ent": weighted_ent}

    else:  # "majority"
        supported_count = sum(1 for e in ent_scores if e >= threshold)
        attribution_rate = supported_count / len(ent_scores)
        pred_label = "ATTRIBUTABLE" if attribution_rate >= 0.5 else "NOT_ATTRIBUTABLE"
        details = {
            "strategy": "majority",
            "attribution_rate": attribution_rate,
            "sentences_checked": len(sentence_results),
            "sentences_supported": supported_count,
        }

    return pred_label, max_ent, max_con, details


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using abbreviation-aware splitter."""
    return split_sentences(text)
