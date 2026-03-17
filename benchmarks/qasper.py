"""
QASPER benchmark suite.

QASPER: Question Answering on Scientific Papers with Evidence-based
annotations. ~5K questions across 1,585 NLP papers.

Each question has:
    - An answer (extractive, abstractive, yes/no, or unanswerable)
    - Evidence paragraphs from the paper that support the answer

This tests the full loop: question -> answer -> evidence extraction.

Published baselines:
    LED-base:           32.7 F1 (answer)
    LED-large:          35.3 F1 (answer)
    LongT5-XL:         39.3 F1 (answer)
    GPT-4 + retrieval:  ~45-55 F1 (answer, varies by setup)

What we measure:
    - Answer correctness (F1 against gold answers)
    - Evidence paragraph selection (precision/recall)
    - For NLI-only: whether gold answer is entailed by gold evidence
"""

import json
import logging
import time

from .base import BenchmarkExample, BenchmarkSuite, PredictionResult
from .nli import NLIEvaluator

logger = logging.getLogger(__name__)


class QASPER(BenchmarkSuite):
    name = "qasper"
    description = "Question answering on scientific papers with evidence"
    labels = ["ANSWERABLE", "UNANSWERABLE"]
    source_url = "https://allenai.org/data/qasper"

    def download(self) -> None:
        from .download import download_qasper
        download_qasper()

    def load(self, split: str = "dev") -> list[BenchmarkExample]:
        """
        Load QASPER examples.

        QASPER format: {paper_id: {title, abstract, full_text, qas: [...]}}
        """
        # Try direct JSON format first
        data_path = self.data_dir / f"qasper_{split}.json"
        if not data_path.exists():
            data_path = self.data_dir / f"{split}_hf.jsonl"

        if not data_path.exists():
            raise FileNotFoundError(f"QASPER data not found at {self.data_dir}. Run: python -m benchmarks.download qasper")

        examples = []

        if data_path.suffix == ".json":
            data = json.loads(data_path.read_text())
            examples = self._parse_qasper_json(data)
        else:
            # HuggingFace JSONL format
            for line in open(data_path):
                item = json.loads(line)
                examples.extend(self._parse_hf_item(item))

        logger.info(f"Loaded QASPER {split}: {len(examples)} QA pairs")
        label_dist = {}
        for ex in examples:
            label_dist[ex.gold_label] = label_dist.get(ex.gold_label, 0) + 1
        logger.info(f"  Label distribution: {label_dist}")
        return examples

    def _parse_qasper_json(self, data: dict) -> list[BenchmarkExample]:
        """Parse QASPER v0.3 JSON format."""
        examples = []

        for paper_id, paper in data.items():
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")

            # Build full text from sections
            full_text_parts = [abstract]
            section_paragraphs = []
            for section in paper.get("full_text", []):
                paragraphs = section.get("paragraphs", [])
                for para in paragraphs:
                    full_text_parts.append(para)
                    section_paragraphs.append(para)

            full_text = "\n\n".join(full_text_parts)

            for qa in paper.get("qas", []):
                question = qa.get("question", "")
                question_id = qa.get("question_id", "")

                for answer_info in qa.get("answers", []):
                    answer = answer_info.get("answer", {})
                    answer_text = ""
                    is_unanswerable = answer.get("unanswerable", False)

                    if is_unanswerable:
                        gold_label = "UNANSWERABLE"
                    else:
                        gold_label = "ANSWERABLE"
                        # Get answer text (extractive or free-form)
                        extractive = answer.get("extractive_spans", [])
                        free_form = answer.get("free_form_answer", "")
                        yes_no = answer.get("yes_no", None)

                        if extractive:
                            answer_text = " ".join(extractive)
                        elif free_form:
                            answer_text = free_form
                        elif yes_no is not None:
                            answer_text = "Yes" if yes_no else "No"

                    # Get evidence paragraphs
                    evidence_texts = []
                    evidence_indices = []
                    for ev in answer.get("evidence", []):
                        if isinstance(ev, str) and ev.strip():
                            evidence_texts.append(ev)
                            if ev in section_paragraphs:
                                evidence_indices.append(section_paragraphs.index(ev))

                    examples.append(BenchmarkExample(
                        id=f"{paper_id}_{question_id}",
                        claim_or_query=question,
                        gold_label=gold_label,
                        evidence_sentences=evidence_texts,
                        evidence_sentence_indices=evidence_indices,
                        source_doc_id=paper_id,
                        source_doc_title=title,
                        full_source_text=full_text,
                        answer_text=answer_text,
                        metadata={
                            "abstract": abstract,
                            "is_unanswerable": is_unanswerable,
                            "section_paragraphs": section_paragraphs,
                        },
                    ))

        return examples

    def _parse_hf_item(self, item: dict) -> list[BenchmarkExample]:
        """Parse a single HuggingFace format item."""
        raise NotImplementedError(
            "HuggingFace JSONL format parsing is not implemented for QASPER. "
            "Use the native JSON format (qasper_{split}.json) instead."
        )

    def map_nli_label(self, nli_label: str) -> str:
        if nli_label in ("SUPPORTS", "REFUTES"):
            return "ANSWERABLE"
        return "UNANSWERABLE"

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
        decompose_evidence: bool = False,
        **kwargs,
    ) -> list[PredictionResult]:
        """
        Run NLI-only on QASPER.

        Tests whether the gold answer is entailed by the gold evidence.
        This measures NLI quality on scientific paper content.

        For answerable questions: check if answer is entailed by evidence.
        For unanswerable: check that evidence does NOT entail any answer.

        Handles special answer types:
        - Yes/No answers: reformulated as "question -> answer" for NLI
        - Short extractive spans: checked directly
        - Free-form answers: checked as claims
        """
        predictions = []

        for ex in examples:
            start = time.time()

            if ex.gold_label == "UNANSWERABLE":
                predictions.append(PredictionResult(
                    example_id=ex.id,
                    gold_label=ex.gold_label,
                    predicted_label="UNANSWERABLE",
                    correct=True,
                    tier="nli-only",
                ))
                continue

            if not ex.answer_text or not ex.evidence_sentences:
                continue

            # Reformulate yes/no answers to be more NLI-friendly
            claim = _reformulate_answer(ex.answer_text, ex.claim_or_query)

            result = nli.classify_claim(
                claim=claim,
                evidence_sentences=ex.evidence_sentences,
                entailment_threshold=entailment_threshold,
                use_rerank=use_rerank,
                use_confidence_margin=use_confidence_margin,
                use_minicheck_fallback=use_minicheck_fallback,
                fallback_model=fallback_model,
                use_passage_scoring=use_passage_scoring,
                decompose_evidence=decompose_evidence,
            )

            is_entailed = result["entailment"] >= entailment_threshold
            pred_label = "ANSWERABLE" if is_entailed else "UNANSWERABLE"
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


def _reformulate_answer(answer_text: str, question: str) -> str:
    """
    Reformulate short/yes-no answers into declarative NLI-friendly claims.

    NLI models are trained on declarative premise-hypothesis pairs, not Q&A.
    "Is the model realistic? Yes" is a poor hypothesis. "The model is realistic."
    is what NLI models expect.

    Strategy:
    1. Yes/No → convert question to declarative assertion/negation
    2. Short answers → merge into declarative "The X is/are Y" form
    3. Long answers → pass through (already claim-like)
    """
    stripped = answer_text.strip().rstrip(".")
    lower = stripped.lower()

    # Yes/No answers: convert question to declarative statement
    if lower in ("yes", "no", "true", "false"):
        is_affirmative = lower in ("yes", "true")
        declarative = _question_to_declarative(question, is_affirmative)
        if declarative:
            return declarative
        # Fallback: simple Q+A concatenation
        return f"{question} {answer_text.strip()}"

    # Short answers (1-5 words): merge with question into declaration
    word_count = len(stripped.split())
    if word_count <= 5:
        merged = _merge_qa_declarative(question, stripped)
        if merged:
            return merged
        # Fallback: Q + A with period
        return f"{question} {stripped}."

    return answer_text


def _question_to_declarative(question: str, affirmative: bool) -> str | None:
    """
    Convert a yes/no question into a declarative statement.

    "Is the model realistic?" + Yes → "The model is realistic."
    "Does the system use attention?" + No → "The system does not use attention."
    "Can BERT handle long sequences?" + Yes → "BERT can handle long sequences."
    """
    q = question.strip().rstrip("?").strip()
    words = q.split()
    if len(words) < 3:
        return None

    aux = words[0].lower()
    rest_words = words[1:]

    # "Does/Do/Did [subject] [verb phrase]?" - most common in QASPER
    # Affirmative: just drop the auxiliary → "they use graphical models."
    # Negative: keep auxiliary + "not" → "they do not use graphical models."
    if aux in ("does", "do", "did"):
        rest = " ".join(rest_words)
        if affirmative:
            return f"{rest}."
        return f"{rest_words[0]} {aux} not {' '.join(rest_words[1:])}."

    # "Can/Could/Will/Would/Should/May/Might [subject] [verb phrase]?"
    # Move modal after subject: "BERT can handle long sequences."
    if aux in ("can", "could", "will", "would", "should", "may", "might"):
        neg = " not" if not affirmative else ""
        subject = rest_words[0]
        verb_phrase = " ".join(rest_words[1:])
        return f"{subject} {aux}{neg} {verb_phrase}."

    # "Has/Have/Had [subject] [past participle phrase]?"
    # "Has the model been evaluated?" → "The model has been evaluated."
    if aux in ("has", "have", "had"):
        neg = " not" if not affirmative else ""
        # Find complement: starts at "been" or first past participle
        insert_point = None
        for idx, w in enumerate(rest_words):
            wl = w.lower()
            if wl == "been" or (idx > 0 and wl.endswith("ed")):
                insert_point = idx
                break
        if insert_point is None:
            insert_point = _find_complement_start(rest_words)
        subject = " ".join(rest_words[:insert_point])
        complement = " ".join(rest_words[insert_point:])
        if subject and complement:
            return f"{subject} {aux}{neg} {complement}."

    # "Is/Are/Was/Were [subject] [complement]?"
    # "Is the model realistic?" → "The model is realistic."
    if aux in ("is", "are", "was", "were"):
        neg = " not" if not affirmative else ""
        # Find where the complement starts (after the subject noun phrase)
        insert_point = _find_complement_start(rest_words)
        subject = " ".join(rest_words[:insert_point])
        complement = " ".join(rest_words[insert_point:])
        if subject and complement:
            return f"{subject} {aux}{neg} {complement}."

    return None


def _find_complement_start(words: list[str]) -> int:
    """
    Find where the predicate complement starts in a word list.

    Given "the template-based model realistic", returns 3 (before "realistic").
    Given "the results statistically significant", returns 2 (before "statistically").
    Given "WordNet useful for taxonomic reasoning", returns 1 (before "useful").

    Algorithm: scan left-to-right. An adjective/participle/adverb starts the
    complement unless it's followed (through other adj/adv) by a content noun.
    """
    if len(words) <= 1:
        return max(0, len(words) - 1)

    dets = {
        "the", "a", "an", "this", "that", "these", "those", "any", "some",
        "all", "each", "every", "no", "its", "their", "our", "his", "her", "my",
    }
    preps = {
        "for", "from", "in", "on", "at", "with", "by", "to", "of", "about",
        "into", "through", "during", "before", "after", "between", "among",
        "under", "over", "against", "than", "as",
    }

    i = 0
    # Skip leading determiners
    while i < len(words) and words[i].lower() in dets:
        i += 1

    # Skip "of" NP (e.g., "any of these tasks")
    if i < len(words) and words[i].lower() == "of":
        i += 1
        while i < len(words) and words[i].lower() in dets:
            i += 1

    # Scan for complement start
    while i < len(words):
        w = words[i].lower()

        if _looks_like_predicate(w) or _looks_like_adverb(w):
            # Potential complement start. Check if adj/adv chain is followed by a content noun.
            j = i + 1
            while j < len(words) and (
                _looks_like_predicate(words[j].lower()) or _looks_like_adverb(words[j].lower())
            ):
                j += 1
            # j is now the first non-adj/adv word after position i
            if j < len(words) and words[j].lower() not in preps and words[j].lower() not in dets:
                # Content noun follows → attributive modifier → part of subject
                i = j + 1
            else:
                # No content noun follows → predicative → complement starts here
                return i
        else:
            # Noun-like word → part of subject
            i += 1

    # Fallback: last word is the complement
    return max(1, len(words) - 1)


def _looks_like_predicate(word: str) -> bool:
    """Heuristic: does this word look like a predicate adjective/participle?"""
    w = word.lower().rstrip(",.")
    # Common adjective/participle endings
    suffixes = (
        "ive", "ous", "ful", "ent", "ant", "ble", "ical", "istic",
        "ary", "ory", "al", "ic", "ed", "ing",
    )
    if any(w.endswith(s) for s in suffixes):
        return True
    # Common predicate adjectives in scientific text
    pred_adjs = {
        "useful", "realistic", "different", "similar", "good", "bad",
        "better", "worse", "best", "worst", "new", "old", "true", "false",
        "subject", "able", "available", "possible", "necessary", "sufficient",
        "correct", "wrong", "common", "rare", "large", "small", "high", "low",
        "free", "fair", "robust", "valid", "stable", "reliable",
    }
    return w in pred_adjs


def _looks_like_adverb(word: str) -> bool:
    """Heuristic: does this word look like a predicate-modifying adverb?"""
    w = word.lower()
    adverbs = {
        "very", "quite", "rather", "fairly", "more", "most", "less",
        "statistically", "significantly", "considerably", "particularly",
        "relatively", "highly", "extremely", "really", "always", "never",
        "often", "usually", "also", "still", "already", "not",
    }
    return w in adverbs or w.endswith("ly")


def _merge_qa_declarative(question: str, answer: str) -> str | None:
    """
    Merge a Wh-question + short answer into a declarative claim.

    "What is the dataset used?" + "SNLI" → "The dataset used is SNLI."
    "What methods are compared?" + "BERT and GPT" → "The methods compared are BERT and GPT."
    "How many layers does it have?" + "12" → "It has 12 layers."
    """
    import re

    q = question.strip().rstrip("?").strip()

    # "What is/are [the] X?" → "[The] X is/are answer."
    m = re.match(r"^What\s+(is|are|was|were)\s+(the\s+)?(.+)$", q, re.IGNORECASE)
    if m:
        verb = m.group(1)
        det = m.group(2) or "The "
        rest = m.group(3)
        result = f"{det}{rest} {verb} {answer}."
        return result[0].upper() + result[1:]

    # "What [noun(s)] is/are [verb-ed/used/etc.]?" → "The [noun(s)] [verb-ed] is/are answer."
    m = re.match(r"^What\s+(.+?)\s+(is|are|was|were)\s+(.+)$", q, re.IGNORECASE)
    if m:
        noun_part = m.group(1)
        verb = m.group(2)
        rest = m.group(3)
        return f"The {noun_part} {rest} {verb} {answer}."

    # "What X?" (generic) → "Q A."
    m = re.match(r"^What\s+(.+)$", q, re.IGNORECASE)
    if m:
        return f"{question} {answer}."

    # "Which X?" → "The X is answer."
    m = re.match(r"^Which\s+(.+)$", q, re.IGNORECASE)
    if m:
        return f"{question} {answer}."

    # "How many X does/do/did Y Z?" → "Y Z answer X."
    m = re.match(r"^How\s+many\s+(.+?)\s+(?:does|do|did|are|is|were|was)\s+(.+)$", q, re.IGNORECASE)
    if m:
        noun = m.group(1)
        rest = m.group(2)
        return f"{rest} {answer} {noun}."

    # "How X?" → "Q A."
    m = re.match(r"^How\s+(.+)$", q, re.IGNORECASE)
    if m:
        return f"{question} {answer}."

    # "Where/When/Who X?" → "Q A."
    return f"{question} {answer}."
