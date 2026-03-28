# Retrieval Evaluation Design for Proof-Carrying Answers

## Executive Summary

The PCA paper currently evaluates only the NLI verification component using oracle (gold) evidence. This design outlines a **feasible retrieval evaluation** that measures whether the per-claim evidence retrieval pipeline (hybrid BM25 + dense embedding + cross-encoder reranking at dual granularity) actually finds the right evidence spans.

**Key finding**: Three benchmarks have gold evidence annotations with **evidence span-level granularity**: SciFact, QASPER, and HAGRID. These are ideal for retrieval evaluation. A solo researcher can implement this in ~2-3 weeks by:

1. Running retrieval on SciFact abstracts and QASPER papers (closed-corpus retrieval)
2. Comparing retrieved spans against gold evidence indices
3. Measuring retrieval quality (Recall@k, span precision) and the **gap** between oracle and retrieved-evidence verification
4. Ablating dual-granularity retrieval (sentence vs. passage)

---

## Part 1: Benchmark Suitability Analysis

### Usable Benchmarks (Ranked by Feasibility)

| Benchmark | Gold Evidence | Span-Level? | Closed Corpus? | Dataset Size | Retrieval Eval Feasibility |
|-----------|---------------|-------------|---|---|---|
| **SciFact** | ✅ YES | ✅ Sentence indices | ✅ YES (5,183 abstracts) | 300 dev claims | 🟢 **HIGH** |
| **QASPER** | ✅ YES | ✅ Paragraph indices | ✅ YES (1,585 papers) | ~1,715 Q&A pairs | 🟢 **HIGH** |
| **HAGRID** | ✅ YES | ⚠️ Passage/quote level | ✅ YES (per query) | ~1,300 examples | 🟡 **MEDIUM** |
| FEVER | ✅ YES | ✅ Sentence level | ⚠️ PARTIAL* | 19,895 dev claims | 🟡 **MEDIUM** |
| FActScore | ❌ NO (Wikipedia) | N/A | ⚠️ PARTIAL** | 14,274 facts | 🔴 **LOW** |
| AttributionBench | ✅ YES | ⚠️ Passage level | ⚠️ VARIES | 26K examples | 🔴 **LOW** |
| FACTS Grounding | ✅ YES | ✅ Sentence level | ✅ YES | 3,287 examples | 🟢 **HIGH*** |

**Key annotations:**
- *FEVER: Evidence sentences have Wikipedia sentence IDs, but the corpus requires Wikipedia downloads (~4GB) and the paper doesn't discuss sentence-level retrieval quality.
- **FActScore: Wikipedia articles are dynamic; the exact sentence structure used during annotation may not match current Wikipedia state.
- ***FACTS Grounding: Excellent retrieval eval candidate if corpus is available (Google DeepMind dataset). Not in `pca-eval` currently; would require integration.

---

## Part 2: Recommended Benchmarks (2-3 max for solo researcher)

### PRIMARY: SciFact (Highest Priority)

**Why**:
- Smallest, most tractable corpus (5,183 abstracts with 1–8 sentences each)
- Gold evidence is **sentence indices** (exactly matches the system's sentence-level retrieval unit)
- Closed corpus—no Wikipedia/web access needed
- Directly tests claim decomposition + per-claim retrieval (matches PCA pipeline)
- 300 dev claims is feasible for full retrieval pipeline measurement

**Gold annotation structure** (from code):
```python
evidence = {
    "doc_id_1": [
        {"sentences": [0, 2, 4], "label": "SUPPORT"},  # Sentence indices
        {"sentences": [1, 3], "label": "CONTRADICT"}
    ]
}
```

**Retrieval eval design**:
1. For each claim, retrieve sentences from all abstracts using hybrid search
2. Compare retrieved sentence indices against gold evidence indices
3. Compute:
   - **Recall@k** (did we retrieve ANY gold evidence sentence within top-k?)
   - **Precision@k** (of top-k retrieved, how many are in gold evidence?)
   - **F1@k** (harmonic mean)
   - **Span-level F1** (treat as binary span-matching task across all sentences)

**Expected gaps** (oracle vs. retrieval):
- Oracle (gold): ~95.3% F1 (reported in paper)
- Retrieved: Expect ~70–80% F1. Gap reveals retrieval quality.
- Analysis: Is the gap due to low recall (missing evidence) or low precision (ranking noise)?

---

### SECONDARY: QASPER (Excellent for Multi-Facet Reasoning)

**Why**:
- Tests retrieval in **long documents** (1,585 papers, full-text, many paragraphs)
- Gold evidence is **paragraph indices** (testing passage-level retrieval)
- Questions often require **multi-hop reasoning** (evidence spans multiple paragraphs)
- Closed corpus, but much larger than SciFact (realistic challenge)
- ~1,715 Q&A pairs; feasible if retrieval is pre-cached

**Gold annotation structure**:
```python
example = {
    "question": "What is X?",
    "answers": [
        {
            "answer": "Paragraph 3 states that X is Y.",
            "evidence": [
                {"name": "section_name", "paragraphs": [3, 5]},  # Paragraph indices
                ...
            ]
        }
    ]
}
```

**Retrieval eval design**:
1. For each question, retrieve paragraphs from the full paper
2. Measure:
   - **Coverage**: Did retrieval include at least ONE gold evidence paragraph?
   - **Recall@k**: Of all gold evidence paragraphs, how many appeared in top-k?
   - **Multi-facet coverage**: For questions with evidence from 2+ sections, does retrieval cover all facets?
   - **Comparison by question type**: Simple factual Q&A vs. reasoning-heavy questions

**Expected gaps**:
- Oracle: 88.2% F1 (from paper)
- Retrieved: Expect ~65–75% F1. Larger gap than SciFact because papers are longer and retrieval is harder.

---

### TERTIARY: HAGRID (Attribution Detection)

**Why**:
- Directly measures **proof-carrying answer quality** (core PCA use case)
- Each answer sentence has gold attribution annotations (which passages support it)
- Binary: ATTRIBUTABLE vs. NOT_ATTRIBUTABLE
- Moderate corpus size (~1,300 examples)

**Challenges**:
- Gold evidence is stored as text quotes, not indices (requires string matching against corpus)
- Passages vary in size/structure (less uniform than SciFact sentences)
- Harder to compute exact span-level metrics

**Retrieval eval design**:
1. For each answer sentence, retrieve passages using hybrid search
2. Measure:
   - **Attribution recall**: Does retrieved set include the gold passage(s)?
   - **Precision**: Of retrieved passages, how many are actually attributable?
   - **Deflection rate**: When retrieval fails, how often should the claim be deflected (per PCA logic)?

---

## Part 3: Proposed Retrieval Metrics

### Core Metrics (All three benchmarks)

#### 1. **Recall@k** (Evidence Coverage)
```
Recall@k = |retrieved_top_k ∩ gold_evidence| / |gold_evidence|
```
- k ∈ {1, 3, 5, 10}
- Measures: "Did we retrieve ANY correct evidence in the top-k results?"
- Interpretation: Low recall → retrieval is missing correct evidence entirely

#### 2. **Precision@k** (Rank Quality)
```
Precision@k = |retrieved_top_k ∩ gold_evidence| / k
```
- k ∈ {1, 3, 5, 10}
- Measures: "What fraction of top-k results are correct?"
- Interpretation: Low precision → too much noise; reranking isn't working

#### 3. **Span-Level F1** (Exact Match)
```
Precision = |retrieved ∩ gold| / |retrieved|
Recall = |retrieved ∩ gold| / |gold|
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
- Treats retrieval as a classification task: for every retrievable span, predict whether it's evidence
- Interpretation: Balanced measure of both coverage and precision

#### 4. **MRR (Mean Reciprocal Rank)**
```
MRR = mean(1 / rank_of_first_gold_span)
```
- Measures: How early is the first correct evidence ranked?
- Interpretation: High MRR → good ranking; low MRR → correct evidence is buried

---

### Benchmark-Specific Metrics

#### SciFact (Sentence-Level)
- **Sentence-level Recall@k**: Fraction of claims where ≥1 gold evidence sentence appears in top-k
- **Evidence coverage**: Average number of gold evidence sentences retrieved in top-k

**Example**: For claim with 3 gold evidence sentences, does retrieval find all 3 in top-10? (Coverage). Or just find ≥1? (Recall).

#### QASPER (Paragraph-Level, Multi-Facet)
- **Facet coverage**: For questions with evidence from N distinct sections, retrieve evidence from ≥N sections?
- **Evidence redundancy**: Same question answered by different paragraphs; measure if we retrieve the full set OR sufficient redundancy

**Example**: Question "What datasets were used?" has evidence in Methods and Results. Does retrieval cover both sections?

#### HAGRID (Attribution Quality)
- **Attribution accuracy (with retrieved evidence)**: Run NLI on retrieved passages instead of gold passages; measure deflection rate
- **Harmful errors**: Falsely attributing hallucinations (Type II error)
- **Useful errors**: Missing real attributions (Type I error)

---

## Part 4: Impact of Retrieval on Verification (Verification Gap Analysis)

### Experiment: Oracle vs. Retrieved Evidence

**Protocol**:
1. **Oracle setting** (baseline, already reported):
   - Use gold evidence spans → run NLI → measure F1
   - Result: Known (SciFact: 95.3%, QASPER: 88.2%, HAGRID: 87.5%)

2. **Retrieved setting** (new):
   - Run retrieval pipeline per-claim
   - Return top-5 retrieved spans (or adaptive top-k)
   - Run NLI on retrieved spans → measure F1
   - Compute gap: F1_oracle - F1_retrieved

3. **Analyze gap sources**:
   - **Missing evidence**: Retrieval recall too low → deflect more claims → lower F1
   - **Ranking noise**: Retrieved evidence mixes correct + incorrect → NLI confusion → lower F1
   - **Insufficient coverage**: Multi-span reasoning fails → incomplete coverage → lower F1

### Expected Results

| Benchmark | Oracle F1 | Retrieved F1 (est.) | Gap | Gap Analysis |
|-----------|-----------|----------------------|-----|---|
| SciFact | 95.3% | 78–85% | ~10–17pp | Retrieval recall + ranking (smaller corpus) |
| QASPER | 88.2% | 70–78% | ~10–18pp | Paragraph retrieval + multi-hop reasoning |
| HAGRID | 87.5% | 75–82% | ~5–12pp | Attribution detection is more forgiving |

**Interpretation**:
- If gap is **small (5pp)**: Retrieval is working well; system could deploy
- If gap is **large (15pp+)**: Retrieval needs improvement (reranking, decomposition, etc.)
- If gap **varies by claim type**: Design interventions per type (e.g., multi-hop queries need special handling)

### Deflection Analysis (PCA Specific)

Measure the **eight deflection categories** from the paper (section 2.3) when retrieval fails:

```python
Deflection reasons = {
    "NO_EVIDENCE": retrieval returned nothing,
    "INSUFFICIENT_COVERAGE": retrieved evidence partial,
    "LOW_CONFIDENCE": NLI score below threshold,
    "CONTRADICTION_DETECTED": conflicting evidence,
    "OUT_OF_SCOPE": query outside document domain,
    "AMBIGUOUS_QUERY": multiple interpretations,
    "EXPLICIT_ABSENCE": documents state info unavailable,
    "TANGENTIAL_ONLY": retrieved passages mention terms but don't address query,
}
```

**Metric**: For each deflection reason, measure:
- Frequency among retrieved-evidence failures (vs. oracle failures)
- Is deflection justified? (Compare to oracle ground truth: should this have been deflected?)

---

## Part 5: Dual-Granularity Retrieval Ablation

### Baseline (Current): Dual Granularity
```
Retrieval pipeline:
  → Hybrid search (BM25 + embeddings)
  → Score at sentence-level AND passage-level
  → Cross-encoder rerank (dual-signal)
  → Return top-k sentences + top-k passages
Result: Use best of (sentence scores, passage scores)
```

### Ablation 1: Sentence-Only
```
  → Hybrid search (BM25 + embeddings)
  → Score only at sentence-level
  → Cross-encoder rerank on sentences
  → Return top-k sentences
Effect: Lose coarse-grained context
```

### Ablation 2: Passage-Only
```
  → Hybrid search (BM25 + embeddings)
  → Score only at passage-level (paragraphs/abstracts)
  → Cross-encoder rerank on passages
  → Return top-k passages
Effect: Lose fine-grained precision
```

### Expected Results (SciFact Example)

| Retrieval Mode | Recall@5 | Precision@5 | F1 | Notes |
|---|---|---|---|---|
| **Dual (baseline)** | 75% | 68% | 71.4% | Best of both; avoids both extremes |
| Sentence-only | 82% | 54% | 65.0% | High recall, low precision (noise) |
| Passage-only | 62% | 78% | 69.4% | Low recall (context too broad), high precision |

**Interpretation**:
- Dual granularity should **outperform both** (complementary signals)
- If not: Design of cross-encoder reranking may need review
- Trade-off analysis: precision vs. recall by claim type

---

## Part 6: Concrete Table/Figure Format for Paper

### Table 1: Retrieval Quality Across Benchmarks

```
┌─────────────────────────────────────────────────────────────┐
│ Table: Retrieval Performance (Top-k Metrics)                │
├──────────────┬──────────────┬──────────────┬───────────────┤
│ Benchmark    │ Metric       │ k=5          │ k=10          │
├──────────────┼──────────────┼──────────────┼───────────────┤
│ SciFact      │ Recall       │ 75.3% (±2.1) │ 87.1% (±1.8)  │
│              │ Precision    │ 68.4% (±2.5) │ 58.2% (±2.1)  │
│              │ F1           │ 71.7%        │ 69.9%         │
│              │ MRR          │ 0.742        │ 0.742         │
├──────────────┼──────────────┼──────────────┼───────────────┤
│ QASPER       │ Recall       │ 62.1% (±3.2) │ 74.8% (±2.9)  │
│              │ Precision    │ 59.3% (±3.4) │ 48.1% (±2.8)  │
│              │ F1           │ 60.6%        │ 58.4%         │
│              │ Facet Cov.   │ 81.2%        │ 91.5%         │
├──────────────┼──────────────┼──────────────┼───────────────┤
│ HAGRID       │ Recall       │ 68.4% (±3.8) │ 79.1% (±3.2)  │
│              │ Precision    │ 71.2% (±3.5) │ 65.3% (±3.4)  │
│              │ F1           │ 69.7%        │ 71.4%         │
│              │ MRR          │ 0.681        │ 0.681         │
└──────────────┴──────────────┴──────────────┴───────────────┘
```

### Table 2: Verification Gap (Oracle vs. Retrieved)

```
┌──────────────────────────────────────────────────────────┐
│ Table: Impact of Retrieval on Verification               │
├──────────────┬──────────────┬──────────────┬──────────────┤
│ Benchmark    │ Oracle F1    │ Retrieved F1 │ Gap (Δ)      │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ SciFact      │ 95.3%        │ 84.1%        │ -11.2pp      │
│ QASPER       │ 88.2%        │ 77.3%        │ -10.9pp      │
│ HAGRID       │ 87.5%        │ 81.2%        │ -6.3pp       │
└──────────────┴──────────────┴──────────────┴──────────────┘

Error bars: 95% bootstrap CI (10,000 resamples)
```

### Table 3: Dual-Granularity Ablation (SciFact Example)

```
┌──────────────────────────────────────────────────────┐
│ Table: Sentence vs. Passage Retrieval                │
├──────────────┬───────────────┬─────────────────────┤
│ Retrieval    │ Recall@5      │ Precision@5 │ F1   │
├──────────────┼───────────────┼─────────────┼──────┤
│ Dual-Gran.   │ 75.3%         │ 68.4%       │ 71.7%│
│ Sentence     │ 82.1%         │ 54.3%       │ 65.1%│
│ Passage      │ 62.4%         │ 77.8%       │ 69.5%│
└──────────────┴───────────────┴─────────────┴──────┘

Δ (Dual - Sentence): -6.1pp (loss of precision with coarse context)
Δ (Dual - Passage):  +2.2pp (gain from fine-grained ranking)
```

### Figure 1: Retrieval Recall vs. k (All Benchmarks)

```
Recall@k Curve (k = 1 to 20):

  Recall
    100% |                                    ╱─ QASPER
         |                            ╱───────
         |  80% |                ╱───────────  HAGRID
         |      |            ╱────
         |  60% |        ╱─────────────────  SciFact
         |      |    ╱────
         |  40% | ╱───
         |      |╱
         |   0% ├─────────────────────────────────
         |      0    5    10   15   20
         └─────────────────────────────────
                        k (top-k results)

Observation: Diminishing returns after k=10 (recall plateaus)
```

### Figure 2: Oracle vs. Retrieved F1 Distribution

```
Scatter plot (one point per claim in SciFact dev set):

F1_Retrieved
     100% ├─────────────────────────────────────
         │     ╱ Oracle=Retrieved line
         │    ╱
      80% |   ╱
         │  ╱ (gap region)
         │ ╱
      60% |
         │ ●●●●●
         │  ●  ●●
      40% |
         │
      20% ├──────────────────────────────────────
         └──────────────────────────────────────
         0%  20%  40%  60%  80% 100%
                    F1_Oracle

Interpretation:
- Points on diagonal: retrieval doesn't hurt verification
- Points below: retrieval quality gap (red flag for deployment)
```

---

## Part 7: Implementation Checklist (Solo Researcher, ~2–3 weeks)

### Week 1: SciFact Retrieval Eval
- [ ] Load SciFact dev set (300 claims, 5,183 abstracts)
- [ ] Implement sentence-level retrieval pipeline (BM25 + dense embeddings)
- [ ] Score sentences at dual granularity (sentence + passage context)
- [ ] Compute Recall@k, Precision@k, F1, MRR for k ∈ {1, 3, 5, 10}
- [ ] Run NLI on retrieved spans; measure oracle vs. retrieved gap
- [ ] Generate Table 2 (verification gap)

### Week 2: QASPER + Ablations
- [ ] Implement paragraph-level retrieval for QASPER papers
- [ ] Compute facet coverage (multi-hop evidence)
- [ ] Run dual-granularity ablation (sentence vs. passage vs. dual)
- [ ] Generate Table 3 (ablation results)

### Week 3: Integration + Analysis
- [ ] Implement HAGRID retrieval (quote matching, passage retrieval)
- [ ] Compile all three benchmarks into Table 1
- [ ] Deflection reason categorization (analyze when retrieval causes deflections)
- [ ] Generate Figures 1 & 2
- [ ] Write brief "Retrieval Evaluation" section (1–1.5 pages for paper)

### Code Changes Minimal
- Add new methods to existing benchmark suites:
  - `run_retrieval_only()` — retrieval metrics only
  - `run_retrieval_then_nli()` — verify gap between oracle + retrieval
  - `ablate_granularity()` — sentence vs. passage comparison
- Cache retrieval results in SQLite (fast iteration on analysis)
- Reuse existing NLI evaluator for verification gap

---

## Part 8: How This Fits in the Paper

### New Section: "4.5 Retrieval Evaluation" (~1–1.5 pages)

**Opening**:
> While Section 4 evaluates NLI verification quality using oracle (gold) evidence, we now measure the quality of the per-claim evidence retrieval pipeline that feeds the verification system. This reveals the **practical impact of retrieval noise** on end-to-end PCA generation.

**Subsections**:
1. **Benchmarks with Gold Evidence Spans** — SciFact, QASPER, HAGRID
2. **Retrieval Metrics** — Recall@k, Precision@k, F1, MRR (Table 1)
3. **Verification Gap** — Oracle vs. retrieved evidence (Table 2, Figure 2)
4. **Dual-Granularity Ablation** — Sentence vs. passage (Table 3, Figure 1)
5. **Deflection Analysis** — Which failure modes trigger deflection? (brief)

**Conclusion**:
> Retrieval quality introduces a 6–12pp gap in verification F1, with the largest gap on QASPER (multi-hop reasoning). Dual-granularity retrieval outperforms sentence-only retrieval by 2pp through complementary ranking signals, but opportunities remain in long-document scenarios.

### Placement in Results Section
- Insert as **Section 4.5** (after ablation study, 4.3)
- Or insert as **subsection under 4.2** (expand "Results")

---

## Part 9: Feasibility & Timeline

### Effort Estimate (Solo)
- **Retrieval pipeline setup**: 3–5 days (implement hybrid search, reranking cache)
- **SciFact eval**: 2–3 days (load data, compute metrics)
- **QASPER eval**: 2–3 days (longer documents; facet coverage is tricky)
- **Ablation + analysis**: 2–3 days
- **Figures + tables**: 1–2 days
- **Buffer for debugging**: 2–3 days
- **Total**: 14–22 days of focused work (~2.5–3 weeks calendar time)

### Why This is Feasible
1. **Closed-corpus only** — No web crawling or Wikipedia handling
2. **Gold evidence already annotated** — No manual labeling needed
3. **Reuse existing NLI evaluator** — Don't rebuild verification logic
4. **Metrics are standard** — Recall@k, Precision@k, F1 are well-known
5. **Small datasets** — SciFact (300 claims) + QASPER (~1,715 Q&A) run quickly on CPU

### What Could Go Wrong
- **Retrieval pipeline bugs** (BM25 tokenization, embedding model) — Mitigate: unit tests on 10 examples first
- **Evidence span matching** (string offsets, tokenization mismatch) — Mitigate: use existing indices (SciFact, QASPER provide them)
- **Performance** (1,700 QASPER papers × 10+ questions each is large) — Mitigate: cache retrieval scores in SQLite upfront
- **Metric interpretation** (when is 70% recall "good"?) — Mitigate: compare against baselines (e.g., random retrieval baseline)

---

## Part 10: Baseline Comparisons (Optional)

### Naive Baselines (Measure against these)

| Baseline | Method | Expected Recall@5 | Notes |
|---|---|---|---|
| **Random** | Uniform sampling | ~20% | Lower bound |
| **BM25-only** | No dense embedding | ~55% | Lexical match limit |
| **Dense-only** | No BM25 | ~60% | Embedding quality limit |
| **Majority class** | Always pick longest paragraph | ~30% | Sanity check |

### Contextualize Against Literature

If you find **Recall@5 > 75%** on SciFact:
- Compare to MultiVerS (Wadden et al., 2022) evidence extraction F1 (~70%)
- Compare to dense retrieval baselines (DPR, ColBERT) on fact-checking tasks (~65–75%)
- Your result shows the dual-granularity reranking is competitive

---

## Conclusion

**Key deliverables**:
1. **Table 1**: Retrieval metrics (Recall@k, Precision@k) across SciFact, QASPER, HAGRID
2. **Table 2**: Verification gap (oracle vs. retrieved) — shows practical impact
3. **Table 3**: Ablation (dual vs. sentence vs. passage) — validates design choice
4. **Figures 1–2**: Recall curves and oracle-vs-retrieved scatter
5. **Brief narrative** (~1–1.5 pages) integrating into paper Section 4.5

This evaluation **closes the gap** between the paper's oracle-evidence results and real-world deployment, where retrieval quality directly determines whether the PCA system works. It's a natural fit for the existing paper structure and adds practical credibility to the approach.

