"""
FActScore benchmark suite.

FActScore (Min et al., EMNLP 2023): Fine-grained Atomic Evaluation of Factual
Precision in Long Form Text Generation. Evaluates whether individual atomic
facts in LLM-generated biographies are supported by Wikipedia.

Labels: SUPPORTED, NOT_SUPPORTED (binary)

Published baselines (from Min et al. 2023, Table 1, human-evaluated):
    InstructGPT (text-davinci-003): 58.4% FActScore
    ChatGPT:                        62.5% FActScore
    PerplexityAI:                   63.7% FActScore

What we measure:
    - Binary accuracy: can our NLI verifier correctly classify each atomic
      fact as supported or not-supported by Wikipedia evidence?
    - This demonstrates NLI verification generalizes beyond claim verification
      to atomic fact checking for biography generation.

Data source:
    Human-annotated atomic facts from the FActScore GitHub repository.
    Wikipedia evidence fetched via MediaWiki API for each topic entity.
"""

import json
import logging
import re
import time
import zipfile
from collections import Counter
from math import log

import requests

from .base import BenchmarkExample, BenchmarkSuite, PredictionResult
from .nli import NLIEvaluator

logger = logging.getLogger(__name__)

# FActScore annotated data: Google Drive zip (data.zip from Min et al.)
FACTSCORE_DATA_ZIP_ID = "155exEdKs7R21gZF4G-x54-XN3qswBcPo"
FACTSCORE_LABELED_FILES = [
    "InstructGPT.jsonl",
    "ChatGPT.jsonl",
    "PerplexityAI.jsonl",
]

# Wikipedia API for fetching article text
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


class FActScore(BenchmarkSuite):
    name = "factscore"
    description = "Atomic fact verification for LLM-generated biographies"
    labels = ["SUPPORTED", "NOT_SUPPORTED"]
    source_url = "https://github.com/shmsw25/FActScore"

    def __init__(self) -> None:
        self._wiki_cache: dict[str, str] = {}

    def download(self) -> None:
        """Download FActScore labeled data from Google Drive (data.zip)."""
        out_dir = self.data_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Check if all files already exist
        all_exist = all((out_dir / f).exists() for f in FACTSCORE_LABELED_FILES)
        if all_exist:
            for filename in FACTSCORE_LABELED_FILES:
                count = sum(1 for _ in open(out_dir / filename))
                logger.info(f"FActScore {filename} already downloaded: {count} entities")
            return

        # Download data.zip from Google Drive
        try:
            import gdown
        except ImportError:
            raise RuntimeError(
                "gdown is required to download FActScore data. "
                "Install it: pip install gdown"
            ) from None

        zip_path = out_dir / "data.zip"
        if not zip_path.exists():
            url = f"https://drive.google.com/uc?id={FACTSCORE_DATA_ZIP_ID}"
            logger.info("Downloading FActScore data.zip from Google Drive...")
            gdown.download(url, str(zip_path), quiet=False)

        # Extract labeled JSONL files
        logger.info("Extracting labeled data...")
        with zipfile.ZipFile(zip_path) as zf:
            for filename in FACTSCORE_LABELED_FILES:
                member = f"data/labeled/{filename}"
                if member in zf.namelist():
                    data = zf.read(member)
                    (out_dir / filename).write_bytes(data)
                    count = data.count(b"\n")
                    logger.info(f"Extracted {filename}: {count} entities")
                else:
                    logger.error(f"File {member} not found in data.zip")

        # Clean up zip
        zip_path.unlink(missing_ok=True)

    def _fetch_wikipedia_text(self, title: str) -> str:
        """Fetch plain text of a Wikipedia article via MediaWiki API."""
        if title in self._wiki_cache:
            return self._wiki_cache[title]

        try:
            params = {
                "action": "query",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
                "exlimit": 1,
                "format": "json",
            }
            headers = {
                "User-Agent": "PCAEval/1.0 (https://github.com/conjfrnk/pca-eval)",
            }
            resp = requests.get(WIKIPEDIA_API, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if page_id == "-1":
                    logger.warning(f"Wikipedia article not found: {title}")
                    self._wiki_cache[title] = ""
                    return ""
                text = page.get("extract", "")
                self._wiki_cache[title] = text
                return text
        except Exception as e:
            logger.warning(f"Failed to fetch Wikipedia for '{title}': {e}")
            self._wiki_cache[title] = ""
            return ""

        self._wiki_cache[title] = ""
        return ""

    def _save_wiki_cache(self) -> None:
        """Save fetched Wikipedia articles to disk for reuse."""
        cache_path = self.data_dir / "wiki_cache.json"
        cache_path.write_text(json.dumps(self._wiki_cache, indent=2))
        logger.info(f"Saved Wikipedia cache ({len(self._wiki_cache)} articles)")

    def _load_wiki_cache(self) -> None:
        """Load previously fetched Wikipedia articles."""
        cache_path = self.data_dir / "wiki_cache.json"
        if cache_path.exists():
            self._wiki_cache = json.loads(cache_path.read_text())
            logger.info(f"Loaded Wikipedia cache ({len(self._wiki_cache)} articles)")

    def load(self, split: str = "all") -> list[BenchmarkExample]:
        """
        Load FActScore labeled data.

        Each example is a single atomic fact with its gold label.
        Evidence is the Wikipedia article for the topic entity.

        Args:
            split: Which model's annotations to load.
                   "InstructGPT", "ChatGPT", "PerplexityAI", or "all" (default).
        """
        if split == "all":
            files = FACTSCORE_LABELED_FILES
        else:
            matching = [f for f in FACTSCORE_LABELED_FILES if split.lower() in f.lower()]
            if not matching:
                raise ValueError(
                    f"Unknown split '{split}'. Use: InstructGPT, ChatGPT, PerplexityAI, or all"
                )
            files = matching

        # Load wiki cache
        self._load_wiki_cache()

        examples = []
        topics_to_fetch = set()
        raw_data = []

        # First pass: parse all annotations and collect topics
        for filename in files:
            path = self.data_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"FActScore data not found: {path}. "
                    "Run: python -m benchmarks.download factscore"
                )

            model_name = filename.replace(".jsonl", "")
            for line in open(path):
                item = json.loads(line)
                topic = item.get("topic", "")
                annotations = item.get("annotations")
                if annotations is None:
                    continue

                for sent_data in annotations:
                    # Labels are on human-atomic-facts, not model-atomic-facts
                    facts = sent_data.get("human-atomic-facts") or []
                    for fact_data in facts:
                        text = fact_data.get("text", "")
                        label = fact_data.get("label", "")
                        if not text or not label:
                            continue
                        raw_data.append((topic, text, label, model_name))
                        topics_to_fetch.add(topic)

        # Fetch Wikipedia articles for all topics
        new_fetches = 0
        for topic in sorted(topics_to_fetch):
            if topic not in self._wiki_cache:
                self._fetch_wikipedia_text(topic)
                new_fetches += 1
                if new_fetches % 10 == 0:
                    print(f"\r  Fetching Wikipedia: {new_fetches}/{len(topics_to_fetch) - len(self._wiki_cache) + new_fetches}", end="", flush=True)
                    time.sleep(0.1)  # Rate limiting

        if new_fetches > 0:
            print()
            self._save_wiki_cache()
            logger.info(f"Fetched {new_fetches} new Wikipedia articles")

        # Second pass: create examples with evidence
        example_id = 0
        skipped_irrelevant = 0
        skipped_no_wiki = 0

        for topic, fact_text, label, model_name in raw_data:
            # Map FActScore labels to our binary scheme
            if label == "IR":
                skipped_irrelevant += 1
                continue

            if label == "S":
                gold_label = "SUPPORTED"
            elif label == "NS":
                gold_label = "NOT_SUPPORTED"
            else:
                logger.warning(f"Unknown FActScore label: {label}")
                continue

            wiki_text = self._wiki_cache.get(topic, "")
            if not wiki_text:
                skipped_no_wiki += 1
                continue

            # Truncate wiki text to first ~5000 chars (intro + early sections)
            # to keep evidence manageable for NLI
            evidence_text = wiki_text[:5000]

            examples.append(BenchmarkExample(
                id=str(example_id),
                claim_or_query=fact_text,
                gold_label=gold_label,
                evidence_sentences=[evidence_text],
                source_doc_id=topic,
                source_doc_title=topic,
                full_source_text=wiki_text,
                metadata={
                    "source_model": model_name,
                    "original_label": label,
                },
            ))
            example_id += 1

        if skipped_irrelevant > 0:
            logger.info(f"Skipped {skipped_irrelevant} irrelevant facts")
        if skipped_no_wiki > 0:
            logger.info(f"Skipped {skipped_no_wiki} facts with no Wikipedia article")

        logger.info(
            f"Loaded {len(examples)} FActScore examples "
            f"({sum(1 for e in examples if e.gold_label == 'SUPPORTED')} supported, "
            f"{sum(1 for e in examples if e.gold_label == 'NOT_SUPPORTED')} not supported)"
        )

        return examples

    def map_nli_label(self, nli_label: str) -> str:
        """Map NLI labels to FActScore binary labels."""
        if nli_label == "entailment":
            return "SUPPORTED"
        return "NOT_SUPPORTED"

    @staticmethod
    def _make_evidence_chunks(text: str, chunk_size: int = 1000, overlap: int = 200, max_chunks: int = 20) -> list[str]:
        """Split text into overlapping chunks for decomposed NLI scoring."""
        chunks = []
        start = 0
        while start < len(text) and len(chunks) < max_chunks:
            end = start + chunk_size
            chunk = text[start:end].strip()
            if len(chunk) > 50:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks if chunks else [text[:chunk_size]]

    @staticmethod
    def _make_sentence_chunks(text: str, max_sentences: int = 50, group_size: int = 3) -> list[str]:
        """Split text into sentence groups for NLI scoring.

        Groups adjacent sentences (default 3) to provide local context while
        keeping inputs within DeBERTa's training distribution.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        if not sentences:
            return [text[:2000]]

        chunks = []
        for i in range(0, min(len(sentences), max_sentences), group_size):
            group = " ".join(sentences[i:i + group_size])
            if len(group) > 30:
                chunks.append(group)

        return chunks if chunks else [text[:2000]]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercased tokenizer for relevance scoring."""
        return re.findall(r'\b[a-z0-9]+\b', text.lower())

    @classmethod
    def _bm25_score(cls, query_tokens: list[str], chunk_tokens: list[str],
                    doc_freq: Counter, n_docs: int, avg_dl: float,
                    k1: float = 1.5, b: float = 0.75) -> float:
        """BM25 relevance score between a query and a chunk."""
        score = 0.0
        dl = len(chunk_tokens)
        chunk_tf = Counter(chunk_tokens)
        for term in query_tokens:
            if term not in chunk_tf:
                continue
            tf = chunk_tf[term]
            df = doc_freq.get(term, 0)
            idf = log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            score += idf * tf_norm
        return score

    @classmethod
    def _make_relevant_chunks(cls, text: str, query: str,
                              chunk_size: int = 1000, overlap: int = 200,
                              max_chunks: int = 30, top_k: int = 5) -> list[str]:
        """Select the most relevant chunks using BM25 scoring.

        First splits text into overlapping chunks (more than default to cover
        full article), then ranks by BM25 relevance to the query, and returns
        the top-k most relevant.
        """
        # Generate candidate chunks
        all_chunks = []
        start = 0
        while start < len(text) and len(all_chunks) < max_chunks:
            end = start + chunk_size
            chunk = text[start:end].strip()
            if len(chunk) > 50:
                all_chunks.append(chunk)
            start += chunk_size - overlap

        if not all_chunks:
            return [text[:chunk_size]]

        if len(all_chunks) <= top_k:
            return all_chunks

        # Tokenize
        query_tokens = cls._tokenize(query)
        chunk_token_lists = [cls._tokenize(c) for c in all_chunks]

        # Compute document frequencies
        doc_freq: Counter = Counter()
        for tokens in chunk_token_lists:
            for term in set(tokens):
                doc_freq[term] += 1

        n_docs = len(all_chunks)
        avg_dl = sum(len(t) for t in chunk_token_lists) / n_docs

        # Score and rank
        scores = [
            cls._bm25_score(query_tokens, chunk_tokens, doc_freq, n_docs, avg_dl)
            for chunk_tokens in chunk_token_lists
        ]

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [all_chunks[i] for i in ranked[:top_k]]

    @classmethod
    def _make_relevant_sentences(cls, text: str, query: str,
                                 max_sentences: int = 50, group_size: int = 3,
                                 top_k: int = 8) -> list[str]:
        """Combine sentence-level splitting with BM25 relevance ranking.

        Splits into sentence groups, then ranks by BM25 relevance to the query,
        returning only the top-k most relevant groups.
        """
        all_chunks = cls._make_sentence_chunks(text, max_sentences=max_sentences, group_size=group_size)
        if len(all_chunks) <= top_k:
            return all_chunks

        query_tokens = cls._tokenize(query)
        chunk_token_lists = [cls._tokenize(c) for c in all_chunks]

        doc_freq: Counter = Counter()
        for tokens in chunk_token_lists:
            for term in set(tokens):
                doc_freq[term] += 1

        n_docs = len(all_chunks)
        avg_dl = sum(len(t) for t in chunk_token_lists) / n_docs

        scores = [
            cls._bm25_score(query_tokens, ct, doc_freq, n_docs, avg_dl)
            for ct in chunk_token_lists
        ]

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [all_chunks[i] for i in ranked[:top_k]]

    def _get_chunks(self, full_text: str, query: str, mode: str) -> list[str]:
        """Get evidence chunks based on decomposition mode."""
        if mode == "sentences":
            return self._make_sentence_chunks(full_text)
        elif mode == "relevant":
            return self._make_relevant_chunks(full_text, query)
        elif mode == "relevant_sentences":
            return self._make_relevant_sentences(full_text, query)
        else:  # "chars" (default)
            return self._make_evidence_chunks(full_text)

    def run_nli_only(
        self,
        examples: list[BenchmarkExample],
        nli: NLIEvaluator,
        entailment_threshold: float = 0.5,
        **kwargs,
    ) -> list[PredictionResult]:
        """
        Run NLI verification on FActScore atomic facts.

        Each fact is scored against Wikipedia evidence. When decompose_evidence
        is True, splits the evidence and takes the max entailment score.

        decompose_mode controls how evidence is split:
          - "chars": overlapping character chunks (default, original)
          - "sentences": sentence-boundary groups (3 sentences per chunk)
          - "relevant": BM25-ranked chunks (top 5 most relevant)
        """
        decompose = kwargs.get("decompose_evidence", False)
        decompose_mode = kwargs.get("decompose_mode", "chars")
        predictions = []

        for i, ex in enumerate(examples):
            if not ex.evidence_sentences:
                predictions.append(PredictionResult(
                    example_id=ex.id,
                    gold_label=ex.gold_label,
                    predicted_label="NOT_SUPPORTED",
                    correct=(ex.gold_label == "NOT_SUPPORTED"),
                    tier="nli-only",
                ))
                continue

            start = time.time()

            evidence = ex.evidence_sentences[0]

            if decompose:
                full_text = ex.full_source_text or evidence
                chunks = self._get_chunks(full_text, ex.claim_or_query, decompose_mode)

                best_ent = 0.0
                best_con = 0.0
                best_neu = 1.0
                for chunk in chunks:
                    result = nli.predict_single(chunk, ex.claim_or_query)
                    if result.entailment > best_ent:
                        best_ent = result.entailment
                        best_con = result.contradiction
                        best_neu = result.neutral

                ent_score = best_ent
                con_score = best_con
                neu_score = best_neu
            else:
                result = nli.predict_single(evidence, ex.claim_or_query)
                ent_score = result.entailment
                con_score = result.contradiction
                neu_score = result.neutral

            elapsed_ms = int((time.time() - start) * 1000)

            predicted_label = "SUPPORTED" if ent_score >= entailment_threshold else "NOT_SUPPORTED"

            predictions.append(PredictionResult(
                example_id=ex.id,
                gold_label=ex.gold_label,
                predicted_label=predicted_label,
                correct=(predicted_label == ex.gold_label),
                entailment_score=ent_score,
                contradiction_score=con_score,
                neutral_score=neu_score,
                latency_ms=elapsed_ms,
                tier="nli-only",
                details={
                    "topic": ex.source_doc_title,
                    "source_model": ex.metadata.get("source_model", ""),
                },
            ))

            if (i + 1) % 100 == 0:
                correct = sum(1 for p in predictions if p.correct)
                print(f"\r  Progress: {i + 1}/{len(examples)} ({correct/(i+1):.1%} accuracy)", end="", flush=True)

        if len(examples) > 100:
            print()

        return predictions

    def run_xgboost(
        self,
        examples: list[BenchmarkExample],
        nli: NLIEvaluator,
        entailment_threshold: float = 0.5,
        **kwargs,
    ) -> list[PredictionResult]:
        """
        Run XGBoost multi-signal verification on FActScore.

        Extracts 56 features per fact (NLI scores, lexical overlap, BM25
        relevance, cross-signal interactions) across sentence-level evidence
        chunks, then uses a pre-trained XGBoost model to classify each fact.

        This approach achieves 90%+ binary F1 by combining NLI confidence
        with lexical grounding signals, catching both false positives
        (high NLI but low overlap) and false negatives (low NLI but
        strong lexical/entity match).
        """
        raise NotImplementedError(
            "The XGBoost aggregation strategy requires trained models not included "
            "in this public release. See the paper for methodology details."
        )
