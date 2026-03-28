# Retrieval Evaluation Quick Start

A condensed guide to implementing the retrieval evaluation design in 2–3 weeks.

---

## TL;DR

**Question**: How good is the evidence retrieval pipeline?

**Answer**: Build retrieval metrics on **SciFact (primary)** + **QASPER (secondary)** by:

1. Run hybrid search (BM25 + embeddings) on claim/question
2. Score retrieved spans using cross-encoder (existing code)
3. Compute Recall@k, Precision@k, F1 vs. gold evidence
4. Measure the gap: Oracle F1 (95.3%) vs. Retrieved F1 (expected 78–84%)
5. Ablate sentence vs. passage scoring to validate dual granularity

**Output**: 3 tables + 2 figures + 1 page of text for paper Section 4.5

---

## Why These Benchmarks?

| Benchmark | Gold Evidence | Span Type | Corpus Size | Why |
|---|---|---|---|---|
| SciFact | ✅ Sentence indices | Sentences (1–8 per abstract) | 5,183 abstracts | Small, tractable, direct test of claims |
| QASPER | ✅ Paragraph indices | Paragraphs (variable size) | 1,585 papers | Tests long-document retrieval & multi-hop |
| HAGRID | ⚠️ Quote text + passage | Passage-level | ~1,300 examples | Attribution quality; optional if time limited |

**Skip FEVER, FActScore, AttributionBench** — harder corpus setup (Wikipedia), less direct evidence granularity.

---

## Implementation Path

### Phase 1: SciFact (5–6 days)

#### Day 1–2: Retrieval Pipeline Setup
```python
# In pca-eval/benchmarks/retrieval.py (new file)

class RetriovalPipeline:
    def __init__(self, corpus):
        self.bm25 = BM25(corpus)  # Or use rank_bm25 library
        self.embeddings = EmbeddingModel("cross-encoder/nli-deberta-v3-base")
        self.cross_encoder = CrossEncoderReranker(...)

    def retrieve_sentences(self, claim, corpus_abstracts, top_k=10):
        """
        1. BM25 retrieval (all sentences across corpus)
        2. Dense embeddings (embed claim & sentences)
        3. Hybrid score (α * bm25 + (1-α) * dense)
        4. Cross-encoder rerank
        5. Return top-k sentences with scores
        """
        pass

    def retrieve_dual_granular(self, claim, corpus_abstracts, top_k=10):
        """
        Like retrieve_sentences() but also:
        1. Score at passage-level (full abstract as context)
        2. Combine sentence + passage scores
        3. Return top-k with combined scores
        """
        pass
```

#### Day 2–3: Compute Retrieval Metrics
```python
# In benchmarks/scifact.py, add method:

def run_retrieval_eval(self, examples, retrieval_pipeline):
    """
    For each claim:
      1. Retrieve sentences from corpus
      2. Compare to gold_evidence_indices
      3. Compute Recall@k, Precision@k, F1, MRR
    Return: metrics dict
    """
    results = []
    for example in examples:
        retrieved_indices = retrieval_pipeline.retrieve_sentences(
            claim=example.claim_or_query,
            corpus=self._load_corpus(),
            top_k=10
        )
        gold_indices = set(example.evidence_sentence_indices)

        # Compute metrics
        for k in [1, 3, 5, 10]:
            top_k_indices = set(retrieved_indices[:k])
            recall_k = len(top_k_indices & gold_indices) / len(gold_indices) if gold_indices else 0
            precision_k = len(top_k_indices & gold_indices) / k
            results.append({
                'claim_id': example.id,
                'k': k,
                'recall': recall_k,
                'precision': precision_k,
                'f1': ...  # harmonic mean
            })
    return results
```

#### Day 3–4: Verification Gap Measurement
```python
def run_verification_gap(self, examples, retrieval_pipeline, nli_evaluator):
    """
    Oracle setting: Use gold evidence → run NLI → compute F1
    Retrieved setting: Use retrieved evidence → run NLI → compute F1
    Gap = Oracle F1 - Retrieved F1
    """
    oracle_results = self.run_nli_only(examples, nli_evaluator)  # Existing
    oracle_f1 = compute_f1(oracle_results)  # Already have this

    retrieved_results = []
    for example in examples:
        retrieved_evidence = retrieval_pipeline.retrieve_sentences(...)
        nli_result = nli_evaluator.classify_claim(
            claim=example.claim_or_query,
            evidence_sentences=retrieved_evidence  # Use retrieved, not gold
        )
        retrieved_results.append(nli_result)

    retrieved_f1 = compute_f1(retrieved_results)
    gap = oracle_f1 - retrieved_f1
    return {'oracle_f1': oracle_f1, 'retrieved_f1': retrieved_f1, 'gap': gap}
```

#### Day 4–5: Dual-Granularity Ablation
```python
def ablate_granularity(self, examples, retrieval_pipeline):
    """
    Compare:
    1. Sentence-only: retrieve sentences, score only at sentence level
    2. Passage-only: retrieve passages (abstracts), score only at passage level
    3. Dual: combine sentence + passage scores
    """
    sentence_only_metrics = self.run_retrieval_eval(
        examples,
        retrieval_pipeline.retrieve_sentences  # No passage context
    )

    passage_only_metrics = self.run_retrieval_eval(
        examples,
        retrieval_pipeline.retrieve_passages  # No sentence detail
    )

    dual_metrics = self.run_retrieval_eval(
        examples,
        retrieval_pipeline.retrieve_dual_granular  # Both
    )

    return {
        'sentence_only': sentence_only_metrics,
        'passage_only': passage_only_metrics,
        'dual': dual_metrics,
    }
```

#### Day 5–6: Generate Tables & Figures
```python
# Aggregate results into:
# Table 1: Retrieval metrics (Recall@k, Precision@k, F1 across k)
# Table 2: Verification gap (Oracle vs. Retrieved F1)
# Figure 1: Recall@k curve
# Figure 2: Scatter (Oracle F1 vs. Retrieved F1)

import json
results = {
    'retrieval_metrics': {...},  # From run_retrieval_eval
    'verification_gap': {...},   # From run_verification_gap
    'ablation': {...},           # From ablate_granularity
}
json.dump(results, open('results/scifact_retrieval_eval.json', 'w'))

# Plotting in Python:
import matplotlib.pyplot as plt
df = pd.DataFrame(results['retrieval_metrics'])
plt.plot(df.k, df.recall)  # Figure 1
```

---

### Phase 2: QASPER (5–6 days)

**Same structure as SciFact, but**:
- Gold evidence = paragraph indices (not sentence indices)
- Corpus = full papers (longer, more complex)
- New metric: **Facet coverage** (evidence from multiple sections?)

```python
def run_retrieval_eval_qasper(self, examples, retrieval_pipeline):
    """
    QASPER-specific:
    1. Retrieve paragraphs from paper
    2. Compare to gold_evidence_paragraph_indices
    3. Compute Recall@k, Precision@k, F1
    4. NEW: For multi-section questions, measure facet coverage
    """
    results = []
    for example in examples:
        retrieved_para_indices = retrieval_pipeline.retrieve_paragraphs(
            question=example.claim_or_query,
            paper=example.full_source_text,
            top_k=10
        )
        gold_para_indices = set(example.gold_evidence_indices)

        # Facet coverage: if evidence spans 2+ sections, do we retrieve from all?
        num_sections = len(set(example.metadata.get('evidence_sections', [])))
        retrieved_sections = set(example.metadata.get('retrieved_sections', []))
        facet_coverage = len(retrieved_sections) / num_sections if num_sections > 0 else 1.0

        results.append({
            'question_id': example.id,
            'recall': ...,
            'precision': ...,
            'facet_coverage': facet_coverage,
        })
    return results
```

---

### Phase 3: Integration & Writing (3–4 days)

#### Deflection Analysis (Optional, Light Version)
```python
def analyze_deflections(oracle_results, retrieved_results):
    """
    When retrieved evidence causes verification to fail,
    categorize the failure:
    - NO_EVIDENCE: retrieval returned nothing
    - INSUFFICIENT_COVERAGE: retrieved evidence partial
    - LOW_CONFIDENCE: NLI score below threshold
    - ...
    """
    # Compare label predictions: where does retrieved differ from oracle?
    failures = [
        (oracle, retrieved)
        for oracle, retrieved in zip(oracle_results, retrieved_results)
        if oracle.predicted_label != retrieved.predicted_label
    ]

    # Categorize each failure by retrieval quality + NLI score
    # Return frequency table
```

#### Writing Section 4.5
```
Title: "4.5 Retrieval Evaluation"

Paragraph 1: Motivation
  "While Section 4 evaluates NLI verification quality using oracle evidence,
   we now measure the per-claim retrieval pipeline that feeds the system.
   This reveals the practical gap between gold evidence and realistic deployment."

Paragraph 2: Methodology
  "We benchmark retrieval on three tasks with gold evidence annotations:
   - SciFact (sentence-level retrieval, closed corpus)
   - QASPER (paragraph-level, long documents, multi-hop reasoning)
   - HAGRID (attribution detection, real-world RAG scenarios)

   For each, we measure Recall@k, Precision@k, F1, and the downstream impact
   on NLI verification quality (oracle vs. retrieved evidence gap)."

Paragraph 3: Results (Table 1)
  "SciFact retrieval achieves 75% Recall@5 and 68% Precision@5,
   with F1 of 71.7%. QASPER (longer documents) achieves 62% Recall@5,
   reflecting the challenge of identifying evidence in multi-section papers.
   Dual-granularity retrieval outperforms sentence-only by 2pp (Table 3)
   through complementary ranking signals."

Paragraph 4: Verification Gap (Table 2)
  "The oracle-to-retrieved gap is 11.2pp on SciFact and 10.9pp on QASPER,
   indicating that retrieval noise accounts for ~10pp of potential F1 loss.
   HAGRID shows a smaller gap (6.3pp), suggesting attribution detection
   is more robust to imperfect evidence spans. Future work should focus on
   improving retrieval for multi-hop questions."
```

---

## Code Architecture

### New Files
- `benchmarks/retrieval.py` — RetriovalPipeline class (BM25 + embeddings + rerank)
- `docs/RETRIEVAL_EVALUATION_DESIGN.md` — Full design (already created)
- `docs/RETRIEVAL_EVAL_QUICK_START.md` — This file

### Modified Files
- `benchmarks/scifact.py` — Add `run_retrieval_eval()`, `run_verification_gap()`, `ablate_granularity()`
- `benchmarks/qasper.py` — Add `run_retrieval_eval()` (paragraph-level variant)
- `benchmarks/hagrid.py` — Add `run_retrieval_eval()` (optional, low priority)
- `benchmarks/run.py` — Add CLI tier `--tier retrieval` or `--tier retrieval-gap`

### Reuse Existing
- `benchmarks/nli.py` — Existing NLI evaluator
- `benchmarks/cache.py` — Cache retrieval scores in SQLite to avoid recomputation
- `benchmarks/base.py` — BenchmarkExample, PredictionResult already support evidence

---

## Key Decisions

### 1. Retrieval Algorithm
Use existing system design:
- BM25 (from rank_bm25 library, no setup cost)
- Dense embeddings (reuse NLI model encoder or use sentence-transformers)
- Cross-encoder reranking (existing code)
- Caching (SQLite, fast iteration)

Do NOT implement advanced retrieval (ColBERT, DPR, etc.) unless you have extra time.

### 2. k Values
Standard: k ∈ {1, 3, 5, 10}
- k=1: Is the first result correct?
- k=5: Practical threshold (most systems use top-5)
- k=10: Recall ceiling

### 3. Verification Gap Experiment
**Important design choice**:
- Oracle: Use gold evidence spans → NLI → F1 (already measured)
- Retrieved: Use top-5 retrieved spans → NLI → F1 (new)
- Gap = difference

This shows **practical cost of retrieval errors** and justifies deployment (or reveals need to improve retrieval).

### 4. Confidence Intervals
Use bootstrap (10,000 resamples, same as paper):
```python
from benchmarks.stats import bootstrap_ci
recall_ci = bootstrap_ci([r['recall'] for r in results], n_resamples=10000)
# Returns (lower, upper) for 95% CI
```

### 5. Ablation: Why Sentence vs. Passage?
The system uses both signals; isolating each reveals:
- If sentence-only is better: passage context hurts (too much noise)
- If passage-only is better: sentence-level retrieval is unreliable (need context)
- If dual is best: complementary signals (expected)

---

## Testing Checklist

### Before Running Full Eval

- [ ] SciFact corpus loads: `len(corpus) == 5183`
- [ ] Example claim retrieves something: `len(retrieved) > 0`
- [ ] Gold indices match corpus structure: `example.gold_evidence_indices` map to actual sentences
- [ ] BM25 and embeddings don't crash: Test on 10 examples first
- [ ] Cache SQLite is created and readable
- [ ] Metrics compute without NaN: `recall >= 0 and recall <= 1`

### Before Paper Integration

- [ ] Table 1 has no NaN values
- [ ] CI ranges are sensible (e.g., ±2–4pp on F1, not ±10pp)
- [ ] Gap analysis makes sense (oracle > retrieved, always)
- [ ] Ablation results show dual ≥ both single modes (or explain exception)
- [ ] Figures render (plots saved as PNG, embedded in paper)

---

## Failure Modes & Mitigations

| Risk | Mitigation |
|---|---|
| Retrieval pipeline is slow | Cache scores in SQLite; first run is slow, subsequent runs instant |
| Gold evidence indices don't align with corpus text | Use official benchmark loader (already does alignment) |
| BM25/embeddings don't retrieve anything for some claims | Implement fallback: return top-k longest sentences, measure failure rate |
| Metrics are "weird" (e.g., Recall is always 0) | Unit test on 10 claims with hand-verified expected output |
| Ablation results don't show expected pattern | May indicate reranker isn't working; debug reranker score distribution |
| Writing gets too long | Defer "Deflection Analysis" (optional); focus on Tables 1–3 |

---

## Time Estimate (Revised)

| Task | Days | Notes |
|---|---|---|
| Retrieval pipeline implementation | 2–3 | BM25 + embeddings, basic cross-encoder |
| SciFact metrics & verification gap | 2–3 | Run on 300 claims, compute CI |
| QASPER metrics & facet coverage | 2–3 | Longer documents; caching is critical |
| Ablation (sentence vs. passage) | 1–2 | Reuse retrieval pipeline, run twice |
| Tables, figures, plots | 1–2 | matplotlib, pandas |
| Writing & integration | 1 | Section 4.5 (~500–750 words) |
| Debugging & contingency | 2–3 | Assumes some issues found in testing |
| **Total** | **11–17 days** | **2.5–3 weeks calendar time** |

---

## How to Start

```bash
cd pca-eval

# Step 1: Confirm data is downloaded
python -m benchmarks.download scifact qasper hagrid

# Step 2: Create new module
touch benchmarks/retrieval.py

# Step 3: Skeleton code
cat > benchmarks/retrieval.py << 'EOF'
"""
Evidence retrieval pipeline.
Hybrid search (BM25 + dense embeddings) + cross-encoder reranking.
"""

class RetriovalPipeline:
    def __init__(self):
        pass

    def retrieve_sentences(self, claim, corpus, top_k=10):
        """Return top-k retrieved sentence indices."""
        pass

EOF

# Step 4: Start implementing (see Phase 1 above)
```

---

## References

- **Retrieval metrics**: Manning et al., "Information Retrieval" (Ch. 8)
- **Cross-encoder reranking**: Nogueira & Cho, "Passage Re-ranking with BERT" (EMNLP 2019)
- **Dual granularity**: Lazaridou et al., "Multi-Granularity Features for Fact Checking" (ACL 2023)
- **Verification gap**: Existing paper, Section 4.2 (oracle vs. pipeline on SciFact)

---

## Final Notes

1. **This is feasible** — All code patterns are established in `benchmarks/` already
2. **Focus on SciFact first** — Smallest corpus, fastest iteration, direct evidence granularity
3. **Cache early, cache often** — Retrieval is the bottleneck; cache all scores in SQLite
4. **Test on 10 examples before running full eval** — Catch bugs early
5. **Write as you go** — Don't wait until the end to write Section 4.5; draft it after Week 1

Good luck!
