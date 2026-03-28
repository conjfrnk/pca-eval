"""
Standalone NLI evaluator using DeBERTa cross-encoder.

Wraps DeBERTa cross-encoder models for NLI-based verification benchmarking.

Supports multiple models including MiniCheck (binary fact-checking model).

No API calls. Runs entirely on CPU. This is the free tier.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cache import ResponseCache

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# Only set offline mode if the default model is already cached locally.
# First-time users need to download the model from HuggingFace Hub.
_hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
_default_cached = (_hf_cache / "models--cross-encoder--nli-deberta-v3-base" / "snapshots").exists()
if _default_cached:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger(__name__)

# Singleton models
_model = None
_model_name = None
_fallback_evaluators: dict[str, "NLIEvaluator"] = {}

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"

# Models with binary output (supported/not-supported) instead of 3-way NLI
MINICHECK_MODELS = {
    "lytang/MiniCheck-DeBERTa-v3-Large",
    "yaxili96/FactCG-DeBERTa-v3-Large",
}

MINICHECK_MODEL = "lytang/MiniCheck-DeBERTa-v3-Large"

# Models that use entailment=0, neutral=1, contradiction=2 label ordering
# (reverse of cross-encoder's contradiction=0, entailment=1, neutral=2)
MORITZLAURER_MODELS = {
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
}

NLI_LABELS = ["contradiction", "entailment", "neutral"]
LABEL_ORDER = {"contradiction": 0, "entailment": 1, "neutral": 2}


def is_minicheck_model(model_name: str) -> bool:
    """Check if a model uses MiniCheck's binary output format (2-class)."""
    if model_name in MINICHECK_MODELS:
        return True
    # Detect binary (2-class) models from local config
    config_path = Path(model_name) / "config.json"
    if config_path.is_file():
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        id2label = cfg.get("id2label", {})
        if len(id2label) == 2:
            return True
    return False


def is_moritzlaurer_model(model_name: str) -> bool:
    """Check if a model uses MoritzLaurer's label ordering (ent=0, neu=1, con=2)."""
    return model_name in MORITZLAURER_MODELS


def _resolve_model_path(model_name: str) -> str:
    """Resolve model name to local cache path if available."""
    # Check HuggingFace cache for local copy
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    safe_name = "models--" + model_name.replace("/", "--")
    model_cache = cache_dir / safe_name / "snapshots"

    if model_cache.exists():
        snapshots = list(model_cache.iterdir())
        if snapshots:
            local_path = str(snapshots[0])
            logger.info(f"Using cached model at {local_path}")
            return local_path

    return model_name


def load_model(model_name: str = DEFAULT_MODEL):
    """Load NLI model (singleton, ~1GB, takes ~10s first time)."""
    global _model, _model_name
    if _model is None or _model_name != model_name:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required but not installed. "
                "Install it with: pip install sentence-transformers"
            ) from None
        # Skip HF cache resolution for local directory paths
        if Path(model_name).is_dir():
            resolved = model_name
        else:
            resolved = _resolve_model_path(model_name)
        logger.info(f"Loading NLI model: {resolved}")
        print(f"Loading NLI model ({resolved})... ", end="", flush=True)
        _model = CrossEncoder(resolved, max_length=512)
        print("done.", flush=True)
        _model_name = model_name
        logger.info("NLI model loaded successfully")
        if model_name == DEFAULT_MODEL:
            logger.info(
                "Using default pre-trained model. For paper-reported results, "
                "provide a fine-tuned model via --model-path."
            )
    return _model


def _minicheck_scores_to_nli(support_score: float) -> list[float]:
    """Convert MiniCheck binary support score (0-1) to 3-way NLI probabilities."""
    entailment = float(support_score)
    if support_score < 0.3:
        contradiction = float(1.0 - support_score) * 0.7
    else:
        contradiction = max(0.0, (0.3 - support_score) * 0.5) if support_score < 0.5 else 0.05
    neutral = max(0.0, 1.0 - entailment - contradiction)
    return [contradiction, entailment, neutral]


def _get_fallback_evaluator(
    model_name: str = MINICHECK_MODEL,
    cache: "ResponseCache | None" = None,
) -> "NLIEvaluator":
    """Get or create singleton fallback evaluator for the given model."""
    global _fallback_evaluators
    if model_name not in _fallback_evaluators:
        logger.info(f"Loading fallback model: {model_name}...")
        _fallback_evaluators[model_name] = NLIEvaluator(
            model_name=model_name,
            cache=cache or ResponseCache(),
        )
    return _fallback_evaluators[model_name]


@dataclass
class NLIPrediction:
    """Single NLI prediction with probabilities."""
    premise: str
    hypothesis: str
    label: str                  # "entailment", "contradiction", "neutral"
    entailment: float
    contradiction: float
    neutral: float



class NLIEvaluator:
    """
    Batch NLI evaluation using cross-encoder models.

    Supports standard 3-way NLI models and MiniCheck binary models.
    Handles caching so repeated runs with different thresholds are instant.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache: ResponseCache | None = None,
        batch_size: int = 64,
        model_path: str | None = None,
        decompose_evidence: bool = False,
    ):
        self.model_name = model_path if model_path else model_name
        self.cache = cache or ResponseCache()
        self.batch_size = batch_size
        self.decompose_evidence = decompose_evidence
        self._model = None
        self._is_minicheck = is_minicheck_model(self.model_name)
        self._is_moritzlaurer = is_moritzlaurer_model(self.model_name)

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.model_name)
        return self._model

    def predict_single(self, premise: str, hypothesis: str) -> NLIPrediction:
        """Run NLI on a single premise-hypothesis pair."""
        results = self.predict_batch([(premise, hypothesis)])
        return results[0]

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[NLIPrediction]:
        """
        Run NLI on a batch of (premise, hypothesis) pairs.

        Uses cache to avoid redundant computation.
        Handles both 3-way NLI models and MiniCheck binary models.
        """
        results: list[NLIPrediction | None] = [None] * len(pairs)
        uncached_indices: list[int] = []
        uncached_pairs: list[tuple[str, str]] = []

        # Check cache (keyed by model name to avoid cross-model contamination)
        for i, (premise, hypothesis) in enumerate(pairs):
            cached = self.cache.get_nli(premise, hypothesis, model=self.model_name)
            if cached is not None:
                probs = cached
                label = NLI_LABELS[int(np.argmax(probs))]
                results[i] = NLIPrediction(
                    premise=premise, hypothesis=hypothesis, label=label,
                    contradiction=probs[0], entailment=probs[1], neutral=probs[2],
                )
            else:
                uncached_indices.append(i)
                uncached_pairs.append((premise[:512], hypothesis[:512]))

        # Run model on uncached pairs
        if uncached_pairs:
            for batch_start in range(0, len(uncached_pairs), self.batch_size):
                batch = uncached_pairs[batch_start:batch_start + self.batch_size]
                batch_indices = uncached_indices[batch_start:batch_start + self.batch_size]

                raw_scores = self.model.predict(batch, show_progress_bar=False)
                raw_scores = np.atleast_2d(raw_scores)

                for _j, (scores, idx) in enumerate(zip(raw_scores, batch_indices, strict=False)):
                    scores = np.asarray(scores).flatten()

                    if self._is_minicheck:
                        # MiniCheck outputs [not_supported, supported] logits
                        # Apply softmax to get calibrated probabilities
                        exp_scores = np.exp(scores - np.max(scores))
                        mc_probs = exp_scores / np.sum(exp_scores)
                        support_score = float(mc_probs[1]) if len(mc_probs) > 1 else float(mc_probs[0])
                        probs_list = _minicheck_scores_to_nli(support_score)
                    else:
                        # Standard 3-way NLI softmax
                        if self._is_moritzlaurer:
                            # MoritzLaurer outputs [entailment, neutral, contradiction]
                            # Remap to cross-encoder order [contradiction, entailment, neutral]
                            scores = np.array([scores[2], scores[0], scores[1]])
                        shifted = scores - np.max(scores)
                        probs = np.exp(shifted) / np.sum(np.exp(shifted))
                        probs_list = [float(probs[0]), float(probs[1]), float(probs[2])]

                    label = NLI_LABELS[int(np.argmax(probs_list))]

                    premise, hypothesis = pairs[idx]
                    results[idx] = NLIPrediction(
                        premise=premise, hypothesis=hypothesis, label=label,
                        contradiction=probs_list[0], entailment=probs_list[1], neutral=probs_list[2],
                    )

                    # Cache the result (keyed by model name)
                    self.cache.put_nli(premise, hypothesis, probs_list, model=self.model_name)

        return results

    def classify_claim(
        self,
        claim: str,
        evidence_sentences: list[str],
        entailment_threshold: float = 0.5,
        contradiction_threshold: float = 0.5,
        use_context_window: bool = False,
        all_sentences: list[str] | None = None,
        use_rerank: bool = False,
        use_confidence_margin: bool = False,
        use_minicheck_fallback: bool = False,
        minicheck_fallback_threshold: float = 0.5,
        fallback_model: str | None = None,
        decompose_evidence: bool = False,
        use_passage_scoring: bool = False,
    ) -> dict:
        """
        Classify a claim against multiple evidence sentences.

        Aggregation strategy: max entailment across
        spans, max contradiction across spans.

        Args:
            claim: The claim to classify.
            evidence_sentences: Evidence sentences to score against.
            entailment_threshold: Threshold for SUPPORTS label.
            contradiction_threshold: Threshold for REFUTES label.
            use_context_window: If True, include +/-1 adjacent sentences as context.
                Requires all_sentences to be provided.
            all_sentences: Full list of sentences (e.g., full abstract) for context window.
            use_rerank: If True, re-score top-k sentences with coverage-based selection.
            use_confidence_margin: If True, use (entailment - contradiction) as signal.
            use_minicheck_fallback: If True, fall back to a secondary model when
                primary model scores below minicheck_fallback_threshold.
            minicheck_fallback_threshold: Entailment threshold below which
                fallback is triggered (default 0.5).
            fallback_model: Model name for fallback (default: MiniCheck).
            decompose_evidence: If True, split long evidence strings into
                individual sentences before scoring. Helps with long passages
                that would be truncated at 512 tokens.
            use_passage_scoring: If True, also score the full concatenated
                evidence as a passage alongside individual sentences. Takes
                the max of both: individual scoring catches specific supporting
                sentences while passage scoring captures cross-sentence context.

        Returns:
            Dict with 'label', 'entailment', 'contradiction', 'neutral',
            'per_sentence' predictions, and 'supporting_sentences' indices.
        """
        if not evidence_sentences:
            return {
                "label": "NOT_ENOUGH_INFO",
                "entailment": 0.0,
                "contradiction": 0.0,
                "neutral": 1.0,
                "per_sentence": [],
                "supporting_sentences": [],
                "minicheck_used": False,
            }

        # Decompose long evidence into individual sentences
        scoring_sentences = evidence_sentences
        if decompose_evidence or self.decompose_evidence:
            scoring_sentences = _decompose_evidence(evidence_sentences)

        # Context-enhanced evidence
        if use_context_window and all_sentences and len(all_sentences) > 1:
            scoring_sentences = _build_context_windows(scoring_sentences, all_sentences)

        pairs = [(sent, claim) for sent in scoring_sentences]

        # Passage-level scoring: also score the full concatenated evidence.
        # Individual sentences catch specific support; the full passage captures
        # cross-sentence context that single sentences may miss.
        if use_passage_scoring and len(scoring_sentences) > 1:
            full_passage = " ".join(s[:170] for s in scoring_sentences)[:512]
            pairs.append((full_passage, claim))

        predictions = self.predict_batch(pairs)

        # Separate passage prediction from per-sentence predictions
        passage_pred = None
        if use_passage_scoring and len(scoring_sentences) > 1:
            passage_pred = predictions[-1]
            predictions = predictions[:-1]

        max_ent = max(p.entailment for p in predictions)
        max_con = max(p.contradiction for p in predictions)

        if passage_pred and passage_pred.entailment > max_ent:
            max_ent = passage_pred.entailment
        if passage_pred and passage_pred.contradiction > max_con:
            max_con = passage_pred.contradiction

        # Coverage-based rerank: select sentences that maximize claim coverage
        if use_rerank and len(predictions) >= 2:
            rerank_score = self._rerank_coverage(predictions, claim, k=3)
            if rerank_score > max_ent:
                max_ent = rerank_score

        # Fallback: when primary model is uncertain, get a second opinion
        minicheck_used = False
        fb_model = fallback_model or MINICHECK_MODEL
        if use_minicheck_fallback and max_ent < minicheck_fallback_threshold and self.model_name != fb_model:
            mc_eval = _get_fallback_evaluator(fb_model, self.cache)
            mc_pairs = [(sent, claim) for sent in scoring_sentences]
            mc_preds = mc_eval.predict_batch(mc_pairs)
            mc_max_ent = max(p.entailment for p in mc_preds)
            mc_max_con = max(p.contradiction for p in mc_preds)

            # Take the better score from either model
            if mc_max_ent > max_ent:
                max_ent = mc_max_ent
                minicheck_used = True
            if mc_max_con > max_con:
                max_con = mc_max_con

        # Reduce contradiction if strong entailment exists
        if max_ent > 0.6 and max_con > 0.3:
            max_con *= 0.5

        neutral = max(0.0, 1.0 - max_ent - max_con)

        # Phase 2c: Confidence margin scoring
        if use_confidence_margin:
            label = _classify_with_margin(
                max_ent, max_con, neutral,
                entailment_threshold, contradiction_threshold,
            )
        else:
            if max_ent >= entailment_threshold and max_ent > max_con:
                label = "SUPPORTS"
            elif max_con >= contradiction_threshold and max_con > max_ent:
                label = "REFUTES"
            else:
                label = "NOT_ENOUGH_INFO"

        supporting = [i for i, p in enumerate(predictions) if p.entailment > 0.3]

        return {
            "label": label,
            "entailment": max_ent,
            "contradiction": max_con,
            "neutral": neutral,
            "per_sentence": [
                {"index": i, "entailment": p.entailment, "contradiction": p.contradiction, "neutral": p.neutral}
                for i, p in enumerate(predictions)
            ],
            "supporting_sentences": supporting,
            "minicheck_used": minicheck_used,
        }

    def _rerank_coverage(
        self,
        predictions: list[NLIPrediction],
        claim: str,
        k: int = 3,
    ) -> float:
        """
        Coverage-based rerank: select sentences that maximize claim token coverage.

        Instead of picking the top-k by entailment score (which may select
        redundant sentences), greedily select sentences that cover the most
        uncovered claim tokens. This captures complementary information from
        different evidence spans.

        Falls back to standard top-k if the claim has no content tokens.
        """
        # Extract claim content tokens (skip stopwords)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "have", "has", "had", "do", "does", "did", "will", "would",
                     "could", "should", "may", "might", "of", "in", "to", "for",
                     "with", "on", "at", "from", "by", "that", "this", "it",
                     "and", "or", "but", "not", "no", "if", "as", "than"}
        claim_tokens = {
            w.lower().strip(".,;:!?()\"'")
            for w in claim.split()
            if w.lower().strip(".,;:!?()\"'") not in stopwords and len(w) > 1
        }

        if not claim_tokens:
            # Fall back to top-k by score
            sorted_preds = sorted(predictions, key=lambda p: p.entailment, reverse=True)
            top_k = sorted_preds[:k]
            concatenated = " ".join(p.premise for p in top_k)
            result = self.predict_single(concatenated, claim)
            return result.entailment

        # Greedy coverage selection: pick the sentence that covers the most
        # uncovered claim tokens, repeating up to k times
        selected: list[NLIPrediction] = []
        remaining = list(predictions)
        uncovered = set(claim_tokens)

        for _ in range(min(k, len(remaining))):
            best_pred = None
            best_coverage = -1
            best_idx = -1

            for idx, pred in enumerate(remaining):
                pred_tokens = {
                    w.lower().strip(".,;:!?()\"'")
                    for w in pred.premise.split()
                }
                coverage = len(uncovered & pred_tokens)
                # Tie-break by entailment score
                if coverage > best_coverage or (
                    coverage == best_coverage and pred.entailment > (best_pred.entailment if best_pred else 0)
                ):
                    best_coverage = coverage
                    best_pred = pred
                    best_idx = idx

            if best_pred is None:
                break

            selected.append(best_pred)
            remaining.pop(best_idx)
            pred_tokens = {
                w.lower().strip(".,;:!?()\"'")
                for w in best_pred.premise.split()
            }
            uncovered -= pred_tokens

            if not uncovered:
                break

        if not selected:
            return max(p.entailment for p in predictions)

        concatenated = " ".join(p.premise[:170] for p in selected)
        result = self.predict_single(concatenated[:512], claim)
        return result.entailment


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences with abbreviation awareness.

    The naive regex ``(?<=[.!?])\\s+(?=[A-Z])`` incorrectly splits on
    common abbreviations (``et al.``, ``Dr.``, ``Fig.``, ``U.S.``, ``e.g.``).
    This function protects known abbreviation patterns before splitting, then
    restores the original text.

    Used by benchmark evaluation and evidence decomposition.
    """
    import re

    PH = '\x00'  # sentinel for protected periods
    protected = text

    # Protect the most common false-boundary patterns in scientific text
    protected = re.sub(r'\bet\s+al\.', f'et al{PH}', protected)
    protected = re.sub(r'\be\.g\.', f'e{PH}g{PH}', protected)
    protected = re.sub(r'\bi\.e\.', f'i{PH}e{PH}', protected)
    protected = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Jr|Sr|Rev)\.', rf'\1{PH}', protected)
    protected = re.sub(r'\b(Fig|Eq|Ref|Sec|Vol|No|Ch|Pt)\.', rf'\1{PH}', protected)
    protected = re.sub(r'\b([A-Z])\.([A-Z])\.', rf'\1{PH}\2{PH}', protected)

    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)

    result = []
    for part in parts:
        restored = part.replace(PH, '.').strip()
        if restored and len(restored) > 5:
            result.append(restored)

    return result if result else [text]


def _decompose_evidence(
    evidence_sentences: list[str],
    min_length: int = 20,
    max_sentences: int = 50,
) -> list[str]:
    """
    Split long evidence passages into individual sentences.

    NLI models are trained on short premise-hypothesis pairs (typically 1-2
    sentences). Long passages get truncated at 512 tokens, losing information.
    Decomposing into individual sentences gives the model what it was trained on.

    Only decomposes entries longer than min_length words to avoid over-splitting
    short evidence that's already sentence-level.
    """
    decomposed = []
    for text in evidence_sentences:
        word_count = len(text.split())
        if word_count > min_length:
            for part in split_sentences(text):
                if len(part) > 10:
                    decomposed.append(part)
        else:
            decomposed.append(text)

    if len(decomposed) > max_sentences:
        decomposed = decomposed[:max_sentences]

    return decomposed if decomposed else evidence_sentences


def _build_context_windows(
    evidence_sentences: list[str],
    all_sentences: list[str],
) -> list[str]:
    """
    Build context-enhanced evidence by including +/-1 adjacent sentences.

    For each evidence sentence, find its position in all_sentences and
    include the previous and next sentence as context.
    """
    # Build lookup: sentence text -> index in all_sentences
    sent_to_idx: dict[str, int] = {}
    for i, s in enumerate(all_sentences):
        stripped = s.strip()
        if stripped not in sent_to_idx:
            sent_to_idx[stripped] = i

    enhanced = []
    for sent in evidence_sentences:
        idx = sent_to_idx.get(sent.strip())
        if idx is not None:
            parts = []
            if idx > 0:
                parts.append(all_sentences[idx - 1].strip())
            parts.append(all_sentences[idx].strip())
            if idx < len(all_sentences) - 1:
                parts.append(all_sentences[idx + 1].strip())
            enhanced.append(" ".join(parts))
        else:
            enhanced.append(sent)

    return enhanced


def _classify_with_margin(
    max_ent: float,
    max_con: float,
    neutral: float,
    entailment_threshold: float,
    contradiction_threshold: float,
) -> str:
    """
    Classify using confidence margin: (entailment - contradiction).

    A claim with ent=0.45, con=0.05 (margin=0.40) is much more likely
    SUPPORTS than one with ent=0.45, con=0.35 (margin=0.10).
    """
    margin = max_ent - max_con

    # Margin-adjusted thresholds: we shift the entailment threshold down
    # when the margin is strongly positive (high confidence in support)
    margin_bonus = max(0.0, margin - 0.1) * 0.3  # Up to ~0.15 bonus for very clear margins

    effective_ent_threshold = entailment_threshold - margin_bonus
    effective_con_threshold = contradiction_threshold + margin_bonus

    if max_ent >= effective_ent_threshold and margin > 0.05:
        return "SUPPORTS"
    elif max_con >= effective_con_threshold and margin < -0.05:
        return "REFUTES"
    else:
        return "NOT_ENOUGH_INFO"


def calibrate_temperature(
    evaluator: NLIEvaluator,
    examples: list,
    label_map: dict[str, str] | None = None,
) -> float:
    """
    Learn optimal temperature T for softmax calibration via grid search.

    Runs on a set of examples, testing T values from 0.5 to 3.0,
    and returns the T that maximizes F1.

    Args:
        evaluator: The NLI evaluator to calibrate.
        examples: List of BenchmarkExample objects with gold_label and evidence_sentences.
        label_map: Optional mapping from NLI labels to benchmark labels.

    Returns:
        Optimal temperature value.
    """
    if label_map is None:
        label_map = {"SUPPORTS": "SUPPORTS", "REFUTES": "REFUTES", "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO"}

    temperatures = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0]
    best_t = 1.0
    best_correct = 0

    # Collect raw logits for all examples first (via cache)
    for t in temperatures:
        correct = 0
        total = 0
        for ex in examples:
            if not ex.evidence_sentences:
                continue
            result = evaluator.classify_claim(
                claim=ex.claim_or_query,
                evidence_sentences=ex.evidence_sentences,
            )
            # Temperature-scale the scores
            ent = result["entailment"] ** (1.0 / t)
            con = result["contradiction"] ** (1.0 / t)
            neu = result["neutral"] ** (1.0 / t)
            total_score = ent + con + neu
            if total_score > 0:
                ent /= total_score
                con /= total_score

            if ent >= 0.5 and ent > con:
                pred = "SUPPORTS"
            elif con >= 0.5 and con > ent:
                pred = "REFUTES"
            else:
                pred = "NOT_ENOUGH_INFO"

            pred = label_map.get(pred, pred)
            if pred == ex.gold_label:
                correct += 1
            total += 1

        if correct > best_correct:
            best_correct = correct
            best_t = t

    logger.info(f"Calibration: best temperature T={best_t} ({best_correct}/{total} correct)")
    return best_t


def find_optimal_thresholds(
    evaluator: NLIEvaluator,
    examples: list,
    label_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """
    Find per-label optimal thresholds via grid search.

    Returns dict with 'entailment_threshold' and 'contradiction_threshold'.
    """
    if label_map is None:
        label_map = {"SUPPORTS": "SUPPORTS", "REFUTES": "REFUTES", "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO"}

    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    best_f1 = 0.0
    best_ent_t = 0.5
    best_con_t = 0.5

    for ent_t in thresholds:
        for con_t in thresholds:
            correct = 0
            total = 0
            for ex in examples:
                if not ex.evidence_sentences:
                    continue
                result = evaluator.classify_claim(
                    claim=ex.claim_or_query,
                    evidence_sentences=ex.evidence_sentences,
                    entailment_threshold=ent_t,
                    contradiction_threshold=con_t,
                )
                pred = label_map.get(result["label"], result["label"])
                if pred == ex.gold_label:
                    correct += 1
                total += 1

            acc = correct / total if total > 0 else 0.0
            if acc > best_f1:
                best_f1 = acc
                best_ent_t = ent_t
                best_con_t = con_t

    logger.info(
        f"Optimal thresholds: entailment={best_ent_t}, contradiction={best_con_t} "
        f"(accuracy={best_f1:.1%})"
    )
    return {"entailment_threshold": best_ent_t, "contradiction_threshold": best_con_t}


def compute_vocab_overlap(claim: str, evidence: str) -> float:
    """
    Compute token-level vocabulary overlap between claim and evidence.

    Returns overlap ratio (0-1). Low overlap (< 0.2) indicates vocabulary
    mismatch that may cause NLI under-performance.
    """
    claim_tokens = set(claim.lower().split())
    evidence_tokens = set(evidence.lower().split())
    # Remove stopwords for more meaningful overlap
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                 "have", "has", "had", "do", "does", "did", "will", "would", "could",
                 "should", "may", "might", "shall", "can", "of", "in", "to", "for",
                 "with", "on", "at", "from", "by", "that", "this", "it", "and", "or",
                 "but", "not", "no", "if", "as", "than", "its", "their", "our", "your"}
    claim_tokens -= stopwords
    evidence_tokens -= stopwords
    if not claim_tokens:
        return 1.0
    overlap = len(claim_tokens & evidence_tokens)
    return overlap / len(claim_tokens)
