"""
Starter code for retrieval evaluation.
Copy patterns into benchmarks/retrieval.py and benchmark suite methods.

This file is NOT meant to be run; it's a reference for implementation.
"""

# ============================================================================
# PART 1: RetriovalPipeline class (new file: benchmarks/retrieval.py)
# ============================================================================

import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi  # pip install rank-bm25


class RetriovalPipeline:
    """
    Hybrid retrieval: BM25 + dense embeddings + cross-encoder reranking.
    Supports both sentence-level and passage-level scoring (dual granularity).
    """

    def __init__(self, bm25_model=None, embedding_model=None, cross_encoder_model=None):
        """
        Args:
            bm25_model: BM25Okapi instance (pre-built on corpus)
            embedding_model: SentenceTransformer or similar
            cross_encoder_model: Cross-encoder for reranking
        """
        self.bm25 = bm25_model
        self.embedding_model = embedding_model or "cross-encoder/nli-deberta-v3-base"
        self.cross_encoder_model = cross_encoder_model

    def retrieve_sentences(self, query: str, sentences: List[str], top_k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Retrieve top-k sentences using hybrid search.

        Args:
            query: Claim or question text
            sentences: List of sentences to search within
            top_k: Number of top results to return

        Returns:
            (indices, scores) where indices are into `sentences` list
        """
        # Step 1: BM25 scoring
        bm25_scores = self._bm25_score(query, sentences)

        # Step 2: Dense embedding similarity
        dense_scores = self._dense_similarity(query, sentences)

        # Step 3: Hybrid combination (α = 0.3 BM25, 0.7 dense, tunable)
        alpha = 0.3
        hybrid_scores = alpha * bm25_scores + (1 - alpha) * dense_scores

        # Step 4: Cross-encoder reranking (optional, expensive)
        if self.cross_encoder_model is not None:
            reranked_scores = self._cross_encoder_rerank(query, sentences, indices=np.argsort(-hybrid_scores)[:top_k*2])
            # Merge reranked scores back into full list
            final_scores = np.zeros(len(sentences))
            for idx, rerank_score in zip(np.argsort(-hybrid_scores)[:top_k*2], reranked_scores):
                final_scores[idx] = rerank_score
        else:
            final_scores = hybrid_scores

        # Step 5: Sort and return top-k
        top_indices = np.argsort(-final_scores)[:top_k]
        top_scores = final_scores[top_indices]

        return top_indices.tolist(), top_scores.tolist()

    def retrieve_passages(self, query: str, passages: List[str], top_k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Retrieve top-k passages (paragraphs, abstracts) using hybrid search.
        Same logic as retrieve_sentences, but operates on passage level.
        """
        # Identical logic; just operate on passages instead of sentences
        return self.retrieve_sentences(query, passages, top_k)

    def retrieve_dual_granular(
        self,
        query: str,
        sentences: List[str],
        passages: List[str],
        sentence_to_passage_mapping: List[int],  # sentence_to_passage_mapping[i] = passage index for sentence i
        top_k: int = 10,
    ) -> Tuple[List[int], List[float], str]:
        """
        Retrieve using dual granularity: score sentences AND passages, combine.

        Args:
            query: Claim or question
            sentences: List of individual sentences
            passages: List of passages (paragraphs, abstracts)
            sentence_to_passage_mapping: For each sentence, which passage does it belong to?
            top_k: Number of results to return

        Returns:
            (sentence_indices, scores, granularity_used)
            granularity_used: "sentence" or "passage" (whichever had higher score)
        """
        # Score at sentence level
        sentence_indices, sentence_scores = self.retrieve_sentences(query, sentences, top_k=top_k*2)

        # Score at passage level
        passage_indices, passage_scores = self.retrieve_passages(query, passages, top_k=top_k*2)

        # Expand passage results back to sentence level
        sentence_scores_from_passages = np.zeros(len(sentences))
        for passage_idx, passage_score in zip(passage_indices, passage_scores):
            # Find all sentences belonging to this passage
            for sent_idx, sent_passage_idx in enumerate(sentence_to_passage_mapping):
                if sent_passage_idx == passage_idx:
                    sentence_scores_from_passages[sent_idx] = passage_score

        # Combine: take max of sentence-level and passage-level scores
        combined_scores = np.maximum(
            np.array([self._get_sentence_score(i, sentence_indices, sentence_scores) for i in range(len(sentences))]),
            sentence_scores_from_passages
        )

        # Sort and return top-k
        top_indices = np.argsort(-combined_scores)[:top_k]
        top_scores = combined_scores[top_indices]

        return top_indices.tolist(), top_scores.tolist(), "dual_granular"

    # ========== Helper methods ==========

    def _bm25_score(self, query: str, documents: List[str]) -> np.ndarray:
        """
        Compute BM25 scores for query against all documents.
        Requires self.bm25 (BM25Okapi instance) to be pre-built.
        """
        if self.bm25 is None:
            # Build BM25 on-the-fly (slow for repeated calls; cache instead)
            tokenized_docs = [doc.split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs)

        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        return scores / (np.max(scores) + 1e-8)  # Normalize to [0, 1]

    def _dense_similarity(self, query: str, documents: List[str]) -> np.ndarray:
        """
        Compute dense embedding similarity (cosine distance).
        """
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.embedding_model)
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

        query_embedding = model.encode(query)
        doc_embeddings = model.encode(documents)

        # Cosine similarity
        scores = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        return (scores + 1) / 2  # Normalize from [-1, 1] to [0, 1]

    def _cross_encoder_rerank(self, query: str, documents: List[str], indices: List[int]) -> List[float]:
        """
        Rerank candidate documents using cross-encoder.
        """
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(self.embedding_model)
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

        pairs = [[query, documents[i]] for i in indices]
        scores = model.predict(pairs)
        return scores.tolist()

    def _get_sentence_score(self, sentence_idx: int, top_indices: List[int], top_scores: List[float]) -> float:
        """Helper to look up score for a sentence."""
        if sentence_idx in top_indices:
            return top_scores[top_indices.index(sentence_idx)]
        return 0.0


# ============================================================================
# PART 2: Metrics computation (add to base.py or separate file)
# ============================================================================

def compute_retrieval_metrics(
    retrieved_indices: List[int],
    gold_indices: List[int],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """
    Compute Recall@k, Precision@k, F1 for a single example.

    Args:
        retrieved_indices: Top-k retrieved indices from retrieval pipeline
        gold_indices: Ground-truth evidence indices
        k_values: Values of k to compute metrics for (default: [1, 3, 5, 10])

    Returns:
        Dict with keys: recall_1, recall_3, ..., precision_1, ..., f1_1, ...
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics = {}
    gold_set = set(gold_indices)

    for k in k_values:
        retrieved_at_k = set(retrieved_indices[:k])
        intersection = len(retrieved_at_k & gold_set)

        recall_k = intersection / len(gold_set) if len(gold_set) > 0 else 0.0
        precision_k = intersection / k if k > 0 else 0.0
        f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0.0

        metrics[f"recall_{k}"] = recall_k
        metrics[f"precision_{k}"] = precision_k
        metrics[f"f1_{k}"] = f1_k

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, idx in enumerate(retrieved_indices, start=1):
        if idx in gold_set:
            mrr = 1 / rank
            break

    metrics["mrr"] = mrr

    return metrics


# ============================================================================
# PART 3: Adding to SciFact benchmark (update benchmarks/scifact.py)
# ============================================================================

def run_retrieval_eval(self, examples, retrieval_pipeline, corpus):
    """
    Add this method to SciFact class in benchmarks/scifact.py

    Compute retrieval metrics for all examples.

    Args:
        examples: List of BenchmarkExample (from load())
        retrieval_pipeline: RetriovalPipeline instance
        corpus: Dict mapping doc_id to {abstract: [sentences]}

    Returns:
        List of dicts with retrieval metrics
    """
    from . retrieval import compute_retrieval_metrics

    results = []

    for example in examples:
        # Get all sentences across corpus
        all_sentences = []
        sentence_to_doc = []
        doc_id = example.source_doc_id

        if not doc_id or doc_id not in corpus:
            # No corpus doc; skip or use fallback
            continue

        abstract = corpus[doc_id].get("abstract", [])
        all_sentences = abstract

        # Run retrieval
        retrieved_indices, retrieved_scores = retrieval_pipeline.retrieve_sentences(
            query=example.claim_or_query,
            sentences=all_sentences,
            top_k=10,
        )

        # Compute metrics
        gold_indices = example.evidence_sentence_indices
        metrics = compute_retrieval_metrics(retrieved_indices, gold_indices)

        results.append({
            "example_id": example.id,
            "num_retrieved": len(retrieved_indices),
            "num_gold": len(gold_indices),
            **metrics,
        })

    return results


def run_verification_gap(self, examples, retrieval_pipeline, corpus, nli_evaluator):
    """
    Add this method to SciFact class.

    Measure gap between oracle and retrieved evidence.

    Args:
        examples: List of BenchmarkExample
        retrieval_pipeline: RetriovalPipeline instance
        corpus: Dict mapping doc_id to corpus data
        nli_evaluator: NLIEvaluator instance (from benchmarks/nli.py)

    Returns:
        Dict with oracle_f1, retrieved_f1, gap, etc.
    """
    from benchmarks.base import PredictionResult
    import numpy as np

    oracle_results = []
    retrieved_results = []

    for example in examples:
        # Oracle: use gold evidence
        oracle_result = nli_evaluator.classify_claim(
            claim=example.claim_or_query,
            evidence_sentences=example.evidence_sentences,
        )
        oracle_results.append({
            "gold_label": example.gold_label,
            "pred_label": self.map_nli_label(oracle_result["label"]),
        })

        # Retrieved: use top-5 retrieved evidence
        doc_id = example.source_doc_id
        if not doc_id or doc_id not in corpus:
            # Fallback: no evidence, predict NEI
            retrieved_results.append({
                "gold_label": example.gold_label,
                "pred_label": "NOT_ENOUGH_INFO",
            })
            continue

        abstract = corpus[doc_id].get("abstract", [])
        retrieved_indices, _ = retrieval_pipeline.retrieve_sentences(
            query=example.claim_or_query,
            sentences=abstract,
            top_k=5,
        )
        retrieved_evidence = [abstract[i] for i in retrieved_indices if i < len(abstract)]

        if not retrieved_evidence:
            retrieved_results.append({
                "gold_label": example.gold_label,
                "pred_label": "NOT_ENOUGH_INFO",
            })
        else:
            retrieved_result = nli_evaluator.classify_claim(
                claim=example.claim_or_query,
                evidence_sentences=retrieved_evidence,
            )
            retrieved_results.append({
                "gold_label": example.gold_label,
                "pred_label": self.map_nli_label(retrieved_result["label"]),
            })

    # Compute F1 for both
    oracle_correct = sum(1 for r in oracle_results if r["pred_label"] == r["gold_label"])
    retrieved_correct = sum(1 for r in retrieved_results if r["pred_label"] == r["gold_label"])

    oracle_f1 = oracle_correct / len(oracle_results) if oracle_results else 0.0
    retrieved_f1 = retrieved_correct / len(retrieved_results) if retrieved_results else 0.0
    gap = oracle_f1 - retrieved_f1

    return {
        "oracle_f1": oracle_f1,
        "retrieved_f1": retrieved_f1,
        "gap": gap,
        "oracle_correct": oracle_correct,
        "retrieved_correct": retrieved_correct,
    }


def ablate_granularity(self, examples, retrieval_pipeline, corpus):
    """
    Add this method to SciFact class.

    Compare sentence-only vs. passage-only vs. dual granularity.

    Returns:
        Dict with metrics for each granularity mode
    """
    # Reuse run_retrieval_eval, but with different retrieval modes
    # This is a lightweight ablation; main results come from run_retrieval_eval

    # Note: For SciFact, "passage" = full abstract (one passage per abstract)
    # So we'd need a separate method to retrieve by passage instead of by sentence

    # TODO: Implement if time permits; low priority for initial eval
    pass


# ============================================================================
# PART 4: Example usage in a script
# ============================================================================

if __name__ == "__main__":
    """
    Example: Run retrieval evaluation on SciFact

    python -c "
    from benchmarks.scifact import SciFact
    from benchmarks.retrieval import RetriovalPipeline

    # Load benchmark
    scifact = SciFact()
    examples = scifact.load(split='dev')
    corpus = scifact._load_corpus()

    # Create retrieval pipeline
    pipeline = RetriovalPipeline()

    # Run evaluation
    metrics = scifact.run_retrieval_eval(examples, pipeline, corpus)

    # Aggregate metrics
    import pandas as pd
    df = pd.DataFrame(metrics)
    print(df[['recall_1', 'recall_5', 'recall_10', 'precision_5', 'f1_5']].describe())

    # Gap analysis
    gap_result = scifact.run_verification_gap(examples, pipeline, corpus, nli_evaluator)
    print(f\"Oracle F1: {gap_result['oracle_f1']:.1%}\")
    print(f\"Retrieved F1: {gap_result['retrieved_f1']:.1%}\")
    print(f\"Gap: {gap_result['gap']:.1%}\")
    "
    """
    pass
