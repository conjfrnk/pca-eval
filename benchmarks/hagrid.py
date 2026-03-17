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
import re

from .base import BenchmarkExample, BenchmarkSuite, PredictionResult
from .nli import NLIEvaluator, compute_vocab_overlap, split_sentences

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

        for line in open(data_path):
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
            "majority"          - >50% of answer sentences must be supported
            "whole"             - Check entire answer against concatenated evidence
            "max-score"         - Use max entailment score across sentences
            "any-sentence"      - Attributable if ANY sentence is strongly supported
            "mean-score"        - Use mean entailment score across sentences
            "weighted"          - Length-weighted mean across sentences
            "passage-individual"- Score each passage independently, take max
            "decomposed"        - Break evidence into sentences for fine-grained matching
            "hybrid"            - Combine NLI score with lexical overlap
            "cascade"           - Quick whole-answer check, then fine-grained if uncertain
            "best-passage"      - Score answer against each passage; use best passage
            "multi-granularity" - Combine passage-level and sentence-level scores
            "weakest-link"      - Min entailment across answer sentences (all must pass)
            "all-supported"     - All answer sentences must exceed threshold
            "weakest-link-hybrid" - Min entailment + lexical overlap combined
        """
        predictions = []
        import time as _time

        for ex in examples:
            start = _time.time()

            if not ex.answer_text or not ex.evidence_sentences:
                continue

            passages = ex.metadata.get("passages", ex.evidence_sentences)

            if attribution_strategy == "whole":
                pred_label, max_ent, max_con, details = _strategy_whole(
                    ex, nli, entailment_threshold, use_confidence_margin,
                    use_minicheck_fallback, fallback_model,
                )

            elif attribution_strategy == "passage-individual":
                pred_label, max_ent, max_con, details = _strategy_passage_individual(
                    ex, passages, nli, entailment_threshold, use_confidence_margin,
                    use_minicheck_fallback, fallback_model,
                )

            elif attribution_strategy == "decomposed":
                pred_label, max_ent, max_con, details = _strategy_decomposed(
                    ex, passages, nli, entailment_threshold,
                    use_minicheck_fallback, fallback_model,
                )

            elif attribution_strategy == "hybrid":
                pred_label, max_ent, max_con, details = _strategy_hybrid(
                    ex, nli, entailment_threshold, use_confidence_margin,
                    use_minicheck_fallback, fallback_model,
                )

            elif attribution_strategy == "cascade":
                pred_label, max_ent, max_con, details = _strategy_cascade(
                    ex, passages, nli, entailment_threshold, use_confidence_margin,
                    use_minicheck_fallback, fallback_model,
                )

            elif attribution_strategy == "best-passage":
                pred_label, max_ent, max_con, details = _strategy_best_passage(
                    ex, passages, nli, entailment_threshold,
                    use_minicheck_fallback, fallback_model,
                )

            elif attribution_strategy == "multi-granularity":
                pred_label, max_ent, max_con, details = _strategy_multi_granularity(
                    ex, passages, nli, entailment_threshold,
                    use_minicheck_fallback, fallback_model,
                )

            elif attribution_strategy in ("weakest-link", "all-supported", "weakest-link-hybrid"):
                pred_label, max_ent, max_con, details = _strategy_weakest_link(
                    ex, nli, entailment_threshold, attribution_strategy,
                    use_minicheck_fallback, fallback_model,
                )

            elif attribution_strategy in ("query-whole", "query-sentences", "query-per-quote"):
                pred_label, max_ent, max_con, details = _strategy_query_aware(
                    ex, nli, entailment_threshold, attribution_strategy,
                    use_minicheck_fallback, fallback_model,
                )

            elif attribution_strategy == "multi-signal":
                pred_label, max_ent, max_con, details = _strategy_multi_signal(
                    ex, nli, entailment_threshold,
                    use_minicheck_fallback, fallback_model,
                )

            elif attribution_strategy == "learned-weights":
                pred_label, max_ent, max_con, details = _strategy_learned_weights(
                    ex, nli, entailment_threshold,
                    use_minicheck_fallback, fallback_model,
                )

            elif attribution_strategy == "xgboost":
                pred_label, max_ent, max_con, details = _strategy_xgboost(
                    ex, nli, entailment_threshold,
                    use_minicheck_fallback, fallback_model,
                )

            else:
                # Sentence-level strategies (majority, max-score, any-sentence, mean-score, weighted)
                pred_label, max_ent, max_con, details = _strategy_sentence_level(
                    ex, nli, entailment_threshold, attribution_strategy,
                    use_rerank, use_confidence_margin,
                    use_minicheck_fallback, fallback_model,
                )

            elapsed_ms = int((_time.time() - start) * 1000)

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


def _strategy_passage_individual(
    ex, passages, nli, threshold, use_confidence_margin,
    use_minicheck_fallback, fallback_model,
):
    """Score answer against each passage independently, take max entailment."""
    best_ent = 0.0
    best_con = 0.0
    for passage in passages:
        result = nli.classify_claim(
            claim=ex.answer_text,
            evidence_sentences=[passage],
            entailment_threshold=threshold,
            use_confidence_margin=use_confidence_margin,
            use_minicheck_fallback=use_minicheck_fallback,
            fallback_model=fallback_model,
        )
        if result["entailment"] > best_ent:
            best_ent = result["entailment"]
        if result["contradiction"] > best_con:
            best_con = result["contradiction"]
    pred_label = "ATTRIBUTABLE" if best_ent >= threshold else "NOT_ATTRIBUTABLE"
    return pred_label, best_ent, best_con, {
        "strategy": "passage-individual",
        "best_ent": best_ent,
        "num_passages": len(passages),
    }


def _strategy_decomposed(
    ex, passages, nli, threshold, use_minicheck_fallback, fallback_model,
):
    """Break evidence passages into sentences, score answer sentences individually."""
    # Decompose evidence into individual sentences
    evidence_sents = []
    for passage in passages:
        evidence_sents.extend(_split_sentences(passage))

    if not evidence_sents:
        evidence_sents = passages

    answer_sents = _split_sentences(ex.answer_text)

    # Score each answer sentence against all evidence sentences
    ent_scores = []
    for a_sent in answer_sents:
        result = nli.classify_claim(
            claim=a_sent,
            evidence_sentences=evidence_sents,
            entailment_threshold=threshold,
            use_minicheck_fallback=use_minicheck_fallback,
            fallback_model=fallback_model,
        )
        ent_scores.append(result["entailment"])

    if not ent_scores:
        return "NOT_ATTRIBUTABLE", 0.0, 0.0, {"strategy": "decomposed"}

    mean_ent = sum(ent_scores) / len(ent_scores)
    max_ent = max(ent_scores)
    # Use a blend: if most sentences are well-supported, it's attributable
    supported = sum(1 for e in ent_scores if e >= threshold)
    ratio = supported / len(ent_scores)
    pred_label = "ATTRIBUTABLE" if ratio >= 0.4 else "NOT_ATTRIBUTABLE"
    return pred_label, max_ent, 0.0, {
        "strategy": "decomposed",
        "mean_ent": mean_ent,
        "supported_ratio": ratio,
        "evidence_sents": len(evidence_sents),
        "answer_sents": len(answer_sents),
    }


def _strategy_hybrid(
    ex, nli, threshold, use_confidence_margin,
    use_minicheck_fallback, fallback_model,
):
    """Combine NLI entailment with lexical overlap for robust scoring."""
    # NLI score (whole answer against concatenated evidence)
    concat_evidence = " ".join(ex.evidence_sentences)
    result = nli.classify_claim(
        claim=ex.answer_text,
        evidence_sentences=[concat_evidence],
        entailment_threshold=threshold,
        use_confidence_margin=use_confidence_margin,
        use_minicheck_fallback=use_minicheck_fallback,
        fallback_model=fallback_model,
    )
    nli_ent = result["entailment"]
    max_con = result["contradiction"]

    # Lexical overlap score
    overlap = compute_vocab_overlap(ex.answer_text, concat_evidence)

    # Combined score: NLI is primary, overlap is a boost/penalty
    # High overlap + moderate NLI -> likely attributable
    # Low overlap + high NLI -> might be paraphrasing, still trust NLI
    combined = nli_ent * 0.7 + overlap * 0.3
    pred_label = "ATTRIBUTABLE" if combined >= threshold else "NOT_ATTRIBUTABLE"

    return pred_label, nli_ent, max_con, {
        "strategy": "hybrid",
        "nli_ent": nli_ent,
        "overlap": overlap,
        "combined": combined,
    }


def _strategy_cascade(
    ex, passages, nli, threshold, use_confidence_margin,
    use_minicheck_fallback, fallback_model,
):
    """Quick whole-answer check first; if uncertain, do fine-grained decomposed."""
    # Stage 1: Quick whole-answer check
    concat_evidence = " ".join(ex.evidence_sentences)
    result = nli.classify_claim(
        claim=ex.answer_text,
        evidence_sentences=[concat_evidence],
        entailment_threshold=threshold,
        use_confidence_margin=use_confidence_margin,
        use_minicheck_fallback=use_minicheck_fallback,
        fallback_model=fallback_model,
    )
    whole_ent = result["entailment"]
    max_con = result["contradiction"]

    # If highly confident, return immediately
    if whole_ent >= 0.8:
        return "ATTRIBUTABLE", whole_ent, max_con, {"strategy": "cascade", "stage": "whole-confident"}
    if whole_ent <= 0.15:
        return "NOT_ATTRIBUTABLE", whole_ent, max_con, {"strategy": "cascade", "stage": "whole-reject"}

    # Stage 2: Fine-grained passage-level check
    best_passage_ent = 0.0
    for passage in passages:
        r = nli.classify_claim(
            claim=ex.answer_text,
            evidence_sentences=[passage],
            entailment_threshold=threshold,
            use_minicheck_fallback=use_minicheck_fallback,
            fallback_model=fallback_model,
        )
        if r["entailment"] > best_passage_ent:
            best_passage_ent = r["entailment"]

    # Combine whole and passage-level scores
    combined_ent = max(whole_ent, best_passage_ent)
    pred_label = "ATTRIBUTABLE" if combined_ent >= threshold else "NOT_ATTRIBUTABLE"

    return pred_label, combined_ent, max_con, {
        "strategy": "cascade",
        "stage": "passage-refined",
        "whole_ent": whole_ent,
        "best_passage_ent": best_passage_ent,
    }


def _strategy_best_passage(
    ex, passages, nli, threshold, use_minicheck_fallback, fallback_model,
):
    """Score answer sentences against each passage; aggregate by best passage."""
    answer_sents = _split_sentences(ex.answer_text)

    best_passage_score = 0.0
    best_passage_details = {}
    for p_idx, passage in enumerate(passages):
        passage_sents = _split_sentences(passage)
        if not passage_sents:
            passage_sents = [passage]

        # Score each answer sentence against this passage's sentences
        sent_ents = []
        for a_sent in answer_sents:
            r = nli.classify_claim(
                claim=a_sent,
                evidence_sentences=passage_sents,
                entailment_threshold=threshold,
                use_minicheck_fallback=use_minicheck_fallback,
                fallback_model=fallback_model,
            )
            sent_ents.append(r["entailment"])

        if sent_ents:
            passage_score = sum(sent_ents) / len(sent_ents)
            if passage_score > best_passage_score:
                best_passage_score = passage_score
                best_passage_details = {
                    "passage_idx": p_idx,
                    "sent_scores": sent_ents,
                }

    pred_label = "ATTRIBUTABLE" if best_passage_score >= threshold else "NOT_ATTRIBUTABLE"
    return pred_label, best_passage_score, 0.0, {
        "strategy": "best-passage",
        "best_passage_score": best_passage_score,
        **best_passage_details,
    }


def _strategy_multi_granularity(
    ex, passages, nli, threshold, use_minicheck_fallback, fallback_model,
):
    """Combine passage-level and sentence-level NLI scores."""
    # Passage-level: whole answer against each passage
    passage_ents = []
    for passage in passages:
        r = nli.classify_claim(
            claim=ex.answer_text,
            evidence_sentences=[passage],
            entailment_threshold=threshold,
            use_minicheck_fallback=use_minicheck_fallback,
            fallback_model=fallback_model,
        )
        passage_ents.append(r["entailment"])

    # Sentence-level: answer sentences against all evidence
    answer_sents = _split_sentences(ex.answer_text)
    sent_ents = []
    for a_sent in answer_sents:
        r = nli.classify_claim(
            claim=a_sent,
            evidence_sentences=ex.evidence_sentences,
            entailment_threshold=threshold,
            use_minicheck_fallback=use_minicheck_fallback,
            fallback_model=fallback_model,
        )
        sent_ents.append(r["entailment"])

    max_passage = max(passage_ents) if passage_ents else 0.0
    mean_sent = sum(sent_ents) / len(sent_ents) if sent_ents else 0.0
    max_sent = max(sent_ents) if sent_ents else 0.0

    # Multi-granularity combination: passage-level max + sentence-level mean
    combined = max_passage * 0.5 + mean_sent * 0.3 + max_sent * 0.2
    pred_label = "ATTRIBUTABLE" if combined >= threshold else "NOT_ATTRIBUTABLE"

    return pred_label, combined, 0.0, {
        "strategy": "multi-granularity",
        "max_passage": max_passage,
        "mean_sent": mean_sent,
        "max_sent": max_sent,
        "combined": combined,
    }


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


def _strategy_multi_signal(
    ex, nli, threshold,
    use_minicheck_fallback, fallback_model,
):
    """
    Multi-signal attribution: combine NLI with lexical and factual signals.

    Addresses the core HAGRID challenge: NOT_ATTRIBUTABLE answers contain
    hallucinated facts (names, dates, numbers) that aren't in the evidence.
    Pure NLI models miss these because they focus on semantic similarity.

    Signals:
    1. NLI entailment (max across evidence, per answer sentence)
    2. Content word overlap (per sentence)
    3. Named entity / proper noun overlap
    4. Numerical fact coverage
    5. Min-aggregation (weakest link) to catch ANY unsupported sentence
    """
    answer_sents = _split_sentences(ex.answer_text)
    evidence_text = " ".join(ex.evidence_sentences)

    sent_scores = []
    for a_sent in answer_sents:
        # Signal 1: NLI entailment
        result = nli.classify_claim(
            claim=a_sent,
            evidence_sentences=ex.evidence_sentences,
            entailment_threshold=threshold,
            use_minicheck_fallback=use_minicheck_fallback,
            fallback_model=fallback_model,
        )
        nli_ent = result["entailment"]

        # Signal 2: Content word overlap
        overlap = compute_vocab_overlap(a_sent, evidence_text)

        # Signal 3: Named entity / proper noun overlap
        # Extract capitalized multi-word phrases and single capitalized words
        entity_score = _entity_overlap(a_sent, evidence_text)

        # Signal 4: Numerical fact coverage
        num_score = _number_overlap(a_sent, evidence_text)

        # Combined score
        combined = (
            nli_ent * 0.45
            + overlap * 0.20
            + entity_score * 0.20
            + num_score * 0.15
        )

        sent_scores.append({
            "nli_ent": nli_ent,
            "overlap": overlap,
            "entity": entity_score,
            "numeric": num_score,
            "combined": combined,
        })

    if not sent_scores:
        return "NOT_ATTRIBUTABLE", 0.0, 0.0, {"strategy": "multi-signal"}

    # Weakest-link aggregation
    combined_scores = [s["combined"] for s in sent_scores]
    min_combined = min(combined_scores)
    mean_combined = sum(combined_scores) / len(combined_scores)

    # Use a blended threshold: min must be above a lower threshold AND mean above main threshold
    min_passes = min_combined >= (threshold * 0.6)
    mean_passes = mean_combined >= threshold
    pred_label = "ATTRIBUTABLE" if (min_passes and mean_passes) else "NOT_ATTRIBUTABLE"

    return pred_label, min_combined, 0.0, {
        "strategy": "multi-signal",
        "min_combined": min_combined,
        "mean_combined": mean_combined,
        "num_sents": len(sent_scores),
        "sent_details": sent_scores,
    }


def _strategy_learned_weights(
    ex, nli, threshold,
    use_minicheck_fallback, fallback_model,
):
    """
    Learned-weights attribution: use trained logistic regression coefficients.

    Extracts the same signals as multi-signal (NLI, overlap, entity, number)
    but uses optimized weights trained on HAGRID training data.
    Falls back to multi-signal if no trained weights are available.
    """
    raise NotImplementedError(
        "The learned-weights aggregation strategy requires trained models not included "
        "in this public release. See the paper for methodology details."
    )


def _get_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """Extract n-grams from text, lowercased and stripped of punctuation."""
    words = [w.strip(".,;:!?()\"'").lower() for w in text.split()]
    words = [w for w in words if w]
    if len(words) < n:
        return set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _bigram_overlap(claim: str, evidence: str) -> float:
    """Fraction of claim bigrams found in evidence."""
    claim_bi = _get_ngrams(claim, 2)
    if not claim_bi:
        return 1.0
    evidence_bi = _get_ngrams(evidence, 2)
    return len(claim_bi & evidence_bi) / len(claim_bi)


def _trigram_overlap(claim: str, evidence: str) -> float:
    """Fraction of claim trigrams found in evidence."""
    claim_tri = _get_ngrams(claim, 3)
    if not claim_tri:
        return 1.0
    evidence_tri = _get_ngrams(evidence, 3)
    return len(claim_tri & evidence_tri) / len(claim_tri)


def _reverse_coverage(answer: str, evidence: str) -> float:
    """What fraction of evidence content words appear in the answer."""
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "have", "has", "had", "do", "does", "did", "will", "would",
                 "could", "should", "may", "might", "shall", "can", "of", "in",
                 "to", "for", "with", "on", "at", "from", "by", "that", "this",
                 "it", "and", "or", "but", "not", "no", "if", "as", "than",
                 "its", "their", "our", "your"}
    evidence_tokens = set(evidence.lower().split()) - stopwords
    answer_tokens = set(answer.lower().split()) - stopwords
    if not evidence_tokens:
        return 1.0
    return len(evidence_tokens & answer_tokens) / len(evidence_tokens)


def _strategy_xgboost(
    ex, nli, threshold,
    use_minicheck_fallback, fallback_model,
):
    """
    XGBoost aggregation: trained gradient-boosted tree on comprehensive features.

    Extracts NLI + lexical signal features at both sentence-level and
    whole-answer level, then uses a trained XGBoost model to predict
    ATTRIBUTABLE vs NOT_ATTRIBUTABLE.
    """
    raise NotImplementedError(
        "The XGBoost aggregation strategy requires trained models not included "
        "in this public release. See the paper for methodology details."
    )


def _entity_overlap(claim: str, evidence: str) -> float:
    """Compute overlap of proper nouns/named entities between claim and evidence."""
    # Extract likely entities: capitalized words that aren't sentence-starters
    claim_entities = set()
    evidence_entities = set()

    for text, entity_set in [(claim, claim_entities), (evidence, evidence_entities)]:
        words = text.split()
        for i, word in enumerate(words):
            # Skip sentence-initial words
            if i > 0 and word and word[0].isupper():
                clean = word.strip(".,;:!?()\"'")
                if clean and len(clean) > 1:
                    entity_set.add(clean.lower())
            # Also capture numbers as entities
            clean = word.strip(".,;:!?()\"'")
            if clean and any(c.isdigit() for c in clean):
                entity_set.add(clean.lower())

    if not claim_entities:
        return 1.0  # No entities to verify

    overlap = len(claim_entities & evidence_entities)
    return overlap / len(claim_entities)


def _number_overlap(claim: str, evidence: str) -> float:
    """Check if numbers mentioned in the claim appear in the evidence."""
    # Extract all numbers (including years, quantities, percentages)
    claim_numbers = set(re.findall(r'\b\d[\d,.]*\b', claim))
    if not claim_numbers:
        return 1.0  # No numbers to verify

    evidence_numbers = set(re.findall(r'\b\d[\d,.]*\b', evidence))
    overlap = len(claim_numbers & evidence_numbers)
    return overlap / len(claim_numbers)


def _strategy_query_aware(
    ex, nli, threshold, variant,
    use_minicheck_fallback, fallback_model,
):
    """
    Query-aware scoring: include the query as context in the NLI input.

    In HAGRID, answers respond to a query. Including the query in the premise
    helps the NLI model understand what information the evidence conveys.

    Variants:
        "query-whole"     - Query+evidence vs answer (whole)
        "query-sentences" - Query+evidence vs each answer sentence, weakest-link
        "query-per-quote" - Query+each quote vs answer, take max across quotes
    """
    query = ex.claim_or_query or ""

    if variant == "query-whole":
        # Concatenate query and evidence as premise, answer as hypothesis
        concat_evidence = " ".join(ex.evidence_sentences)
        premise = f"Question: {query} Evidence: {concat_evidence}"
        # Use predict_batch directly for custom pair construction
        preds = nli.predict_batch([(premise, ex.answer_text)])
        ent = preds[0].entailment
        con = preds[0].contradiction
        pred_label = "ATTRIBUTABLE" if ent >= threshold else "NOT_ATTRIBUTABLE"
        return pred_label, ent, con, {
            "strategy": "query-whole",
            "entailment": ent,
        }

    elif variant == "query-per-quote":
        # Score answer against each quote separately (with query context)
        passages = ex.metadata.get("passages", ex.evidence_sentences)
        best_ent = 0.0
        best_con = 0.0
        for passage in passages:
            premise = f"Question: {query} Evidence: {passage}"
            preds = nli.predict_batch([(premise, ex.answer_text)])
            if preds[0].entailment > best_ent:
                best_ent = preds[0].entailment
                best_con = preds[0].contradiction
        pred_label = "ATTRIBUTABLE" if best_ent >= threshold else "NOT_ATTRIBUTABLE"
        return pred_label, best_ent, best_con, {
            "strategy": "query-per-quote",
            "best_ent": best_ent,
            "num_passages": len(passages),
        }

    else:  # query-sentences: weakest-link with query context
        answer_sents = _split_sentences(ex.answer_text)
        concat_evidence = " ".join(ex.evidence_sentences)
        premise = f"Question: {query} Evidence: {concat_evidence}"

        pairs = [(premise, a_sent) for a_sent in answer_sents]
        preds = nli.predict_batch(pairs)
        sent_ents = [p.entailment for p in preds]

        if not sent_ents:
            return "NOT_ATTRIBUTABLE", 0.0, 0.0, {"strategy": "query-sentences"}

        min_ent = min(sent_ents)
        mean_ent = sum(sent_ents) / len(sent_ents)
        pred_label = "ATTRIBUTABLE" if min_ent >= threshold else "NOT_ATTRIBUTABLE"
        return pred_label, min_ent, 0.0, {
            "strategy": "query-sentences",
            "min_ent": min_ent,
            "mean_ent": mean_ent,
            "num_sents": len(sent_ents),
            "sent_scores": sent_ents,
        }


def _strategy_weakest_link(
    ex, nli, threshold, variant,
    use_minicheck_fallback, fallback_model,
):
    """
    Weakest-link strategies: ATTRIBUTABLE only if ALL answer sentences are supported.

    Matches HAGRID's annotation scheme where an answer is NOT_ATTRIBUTABLE if
    ANY sentence is not supported by the evidence.

    Variants:
        "weakest-link"        - Use min entailment score across answer sentences.
        "all-supported"       - Strict: all sentences must exceed threshold.
        "weakest-link-hybrid" - Min entailment + lexical overlap boost.
    """
    answer_sents = _split_sentences(ex.answer_text)

    # Score each answer sentence against all evidence
    sent_ents = []
    sent_overlaps = []
    for a_sent in answer_sents:
        result = nli.classify_claim(
            claim=a_sent,
            evidence_sentences=ex.evidence_sentences,
            entailment_threshold=threshold,
            use_minicheck_fallback=use_minicheck_fallback,
            fallback_model=fallback_model,
        )
        sent_ents.append(result["entailment"])
        if variant == "weakest-link-hybrid":
            overlap = compute_vocab_overlap(a_sent, " ".join(ex.evidence_sentences))
            sent_overlaps.append(overlap)

    if not sent_ents:
        return "NOT_ATTRIBUTABLE", 0.0, 0.0, {"strategy": variant}

    min_ent = min(sent_ents)
    mean_ent = sum(sent_ents) / len(sent_ents)
    max_ent = max(sent_ents)

    if variant == "all-supported":
        # Strict: every sentence must pass threshold
        all_pass = all(e >= threshold for e in sent_ents)
        pred_label = "ATTRIBUTABLE" if all_pass else "NOT_ATTRIBUTABLE"
        details = {
            "strategy": "all-supported",
            "min_ent": min_ent,
            "mean_ent": mean_ent,
            "all_pass": all_pass,
            "num_sents": len(sent_ents),
            "num_passing": sum(1 for e in sent_ents if e >= threshold),
        }
    elif variant == "weakest-link-hybrid":
        # Combine min entailment with min lexical overlap
        min_overlap = min(sent_overlaps) if sent_overlaps else 0.0
        # Boosted score: NLI dominant, overlap provides safety net
        combined_scores = [
            e * 0.7 + o * 0.3 for e, o in zip(sent_ents, sent_overlaps, strict=False)
        ]
        min_combined = min(combined_scores) if combined_scores else 0.0
        pred_label = "ATTRIBUTABLE" if min_combined >= threshold else "NOT_ATTRIBUTABLE"
        details = {
            "strategy": "weakest-link-hybrid",
            "min_ent": min_ent,
            "min_overlap": min_overlap,
            "min_combined": min_combined,
            "num_sents": len(sent_ents),
        }
    else:
        # weakest-link: use min entailment directly
        pred_label = "ATTRIBUTABLE" if min_ent >= threshold else "NOT_ATTRIBUTABLE"
        details = {
            "strategy": "weakest-link",
            "min_ent": min_ent,
            "mean_ent": mean_ent,
            "max_ent": max_ent,
            "num_sents": len(sent_ents),
            "sent_scores": sent_ents,
        }

    return pred_label, min_ent, 0.0, details


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using abbreviation-aware splitter."""
    return split_sentences(text)
