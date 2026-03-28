# Retrieval Evaluation Design — Document Index

Complete design for measuring evidence retrieval quality in Proof-Carrying Answers. Includes specifications, implementation guide, code templates, and integration strategy.

## Files in This Directory

### 1. **RETRIEVAL_EVALUATION_DESIGN.md** (24 KB, ~4,500 words)
**Comprehensive design specification.** Read this first for full context.

**Contents**:
- Part 1: Benchmark suitability analysis (which of 7 benchmarks to use)
- Part 2: Recommended benchmarks (SciFact, QASPER, HAGRID) with rationale
- Part 3: Proposed metrics (Recall@k, Precision@k, F1, MRR, facet coverage)
- Part 4: Verification gap experiment (oracle vs. retrieved evidence)
- Part 5: Dual-granularity ablation (sentence vs. passage vs. dual)
- Part 6: Table/figure format for paper (3 tables, 2 figures)
- Part 7: Implementation checklist (3 weeks, week-by-week)
- Part 8: Code changes (what to modify)
- Part 9: Feasibility & timeline
- Part 10: Baseline comparisons (contextualize against literature)

**When to read**: Start here; use for detailed reference during implementation.

---

### 2. **RETRIEVAL_EVAL_QUICK_START.md** (16 KB, ~2,500 words)
**Solo researcher implementation guide.** Condensed, action-oriented version of Design.

**Contents**:
- TL;DR (what problem are we solving, why these benchmarks)
- Code architecture (new files, modified files, reused code)
- Phase 1 detailed walkthrough (SciFact, days 1-6)
- Phase 2 detailed walkthrough (QASPER, days 7-12)
- Phase 3 detailed walkthrough (integration & writing, days 13-17)
- Key decisions (algorithm choices, k values, ablation strategy)
- Testing checklist
- Time estimate (revised, 11-17 days total)
- How to start (bash commands)
- References & notes

**When to read**: Use during implementation; execute each phase sequentially.

---

### 3. **RETRIEVAL_EVAL_SUMMARY.txt** (21 KB, ~3,500 words)
**Executive summary with decision framework.** One-page reference + detailed sections.

**Contents**:
- Executive summary (problem, solution, outcome)
- Feasibility assessment (2-3 weeks, 15-19 days)
- Benchmark selection matrix (SciFact primary, QASPER secondary, HAGRID optional)
- Metrics & measurement strategy (detailed, with formulas)
- Tables & figures for paper (inline mockups)
- Writing: Section 4.5 content (~550 words)
- Implementation checklist (3-week breakdown)
- Code changes required (minimal)
- Risks & mitigations (6 key risks with solutions)
- Decision: proceed or not?

**When to read**: Before starting; validate decision to proceed. Reference during design reviews.

---

### 4. **RETRIEVAL_IMPLEMENTATION_STARTER.py** (16 KB, ~500 lines of code)
**Copy-paste-ready code templates.** Not meant to run standalone; use as reference.

**Contents**:
- Part 1: RetriovalPipeline class skeleton (BM25 + embeddings + cross-encoder)
  - retrieve_sentences()
  - retrieve_passages()
  - retrieve_dual_granular()
  - Helper methods (_bm25_score, _dense_similarity, _cross_encoder_rerank)
- Part 2: Metrics computation (compute_retrieval_metrics)
  - Recall@k, Precision@k, F1, MRR for single examples
- Part 3: Adding to SciFact benchmark
  - run_retrieval_eval() method
  - run_verification_gap() method
  - ablate_granularity() method
- Part 4: Example usage / testing script

**When to read**: During implementation; copy patterns into benchmarks/retrieval.py and benchmark suite files.

---

## Quick Navigation

**By Role**:

| Role | Read First | Then |
|---|---|---|
| **Project Lead** | RETRIEVAL_EVAL_SUMMARY.txt | RETRIEVAL_EVALUATION_DESIGN.md (Part 1–2) |
| **Solo Researcher** | RETRIEVAL_EVAL_QUICK_START.md | RETRIEVAL_IMPLEMENTATION_STARTER.py |
| **Code Reviewer** | RETRIEVAL_IMPLEMENTATION_STARTER.py | RETRIEVAL_EVALUATION_DESIGN.md (Part 8) |
| **Paper Writer** | RETRIEVAL_EVALUATION_DESIGN.md (Part 6) | RETRIEVAL_EVAL_SUMMARY.txt (Writing section) |

**By Timeline**:

| Stage | Documents |
|---|---|
| **Decision** | RETRIEVAL_EVAL_SUMMARY.txt (exec summary + decision framework) |
| **Planning** | RETRIEVAL_EVAL_QUICK_START.md (timeline, checklist) |
| **Implementation** | RETRIEVAL_IMPLEMENTATION_STARTER.py (code patterns) |
| **Validation** | RETRIEVAL_EVALUATION_DESIGN.md (Part 9, risks & mitigations) |
| **Writing** | RETRIEVAL_EVAL_SUMMARY.txt (section 4.5 content) |
| **Integration** | RETRIEVAL_EVALUATION_DESIGN.md (Part 6, tables/figures) |

**By Topic**:

| Topic | Reference |
|---|---|
| Benchmark selection | RETRIEVAL_EVALUATION_DESIGN.md (Part 1–2) |
| Metrics definition | RETRIEVAL_EVALUATION_DESIGN.md (Part 3) |
| Verification gap | RETRIEVAL_EVALUATION_DESIGN.md (Part 4) |
| Ablation study | RETRIEVAL_EVALUATION_DESIGN.md (Part 5) |
| Paper integration | RETRIEVAL_EVALUATION_DESIGN.md (Part 6) |
| Table/figure formats | RETRIEVAL_EVAL_SUMMARY.txt (executive section) |
| Code architecture | RETRIEVAL_EVAL_QUICK_START.md (code architecture section) |
| Testing strategy | RETRIEVAL_EVAL_QUICK_START.md (testing checklist) |
| Risk mitigation | RETRIEVAL_EVAL_SUMMARY.txt (risks section) |
| Implementation detail | RETRIEVAL_IMPLEMENTATION_STARTER.py |

---

## Summary Table

| File | Size | Audience | Purpose | Read Time |
|---|---|---|---|---|
| RETRIEVAL_EVALUATION_DESIGN.md | 24 KB | Everyone | Comprehensive spec | 20 min |
| RETRIEVAL_EVAL_QUICK_START.md | 16 KB | Solo researcher | Action-oriented guide | 15 min |
| RETRIEVAL_EVAL_SUMMARY.txt | 21 KB | Leads, decision-makers | Executive reference | 10 min |
| RETRIEVAL_IMPLEMENTATION_STARTER.py | 16 KB | Developers | Code templates | 15 min (reference) |

**Total documentation**: ~77 KB, ~10,500 words

---

## Key Decisions Summarized

### ✅ Use These Benchmarks

1. **SciFact** (PRIMARY)
   - 300 dev claims, 5,183 abstracts
   - Gold evidence: sentence indices (exact)
   - Expected Recall@5: ~75%, Precision@5: ~68%
   - Oracle-to-retrieved gap: ~11pp

2. **QASPER** (SECONDARY)
   - ~1,715 Q&A pairs, 1,585 papers
   - Gold evidence: paragraph indices
   - Expected Recall@5: ~62%, Facet coverage: ~81%
   - Oracle-to-retrieved gap: ~11pp

3. **HAGRID** (OPTIONAL)
   - ~1,300 examples
   - Gold evidence: passage-level attribution
   - Expected oracle-to-retrieved gap: ~6pp

### ❌ Skip These Benchmarks

- **FEVER**: Wikipedia corpus setup (~4GB), not closed-corpus
- **FActScore**: Wikipedia changes over time, gold annotations may not match
- **AttributionBench**: Mixed corpus types, less uniform spans
- **FACTS Grounding**: Data not in pca-eval repo, integration overhead

### 📊 Metrics to Compute

**For all benchmarks**:
- Recall@k (k ∈ {1,3,5,10})
- Precision@k
- F1 (harmonic mean)
- MRR (mean reciprocal rank)

**For QASPER**:
- Facet coverage (multi-section questions)

**Core experiment**:
- Verification gap: F1_oracle - F1_retrieved

**Ablation**:
- Dual granularity vs. sentence-only vs. passage-only

### 📈 Tables & Figures for Paper

**Table 1**: Retrieval metrics (k=1,3,5,10 for all benchmarks)
**Table 2**: Verification gap (oracle vs. retrieved F1)
**Table 3**: Ablation (dual vs. sentence vs. passage)
**Figure 1**: Recall@k curves (all benchmarks)
**Figure 2**: Oracle vs. retrieved scatter (per-claim)

### ⏱️ Timeline

- **Week 1**: SciFact retrieval eval (5-6 days)
- **Week 2**: QASPER retrieval eval + ablation (5-6 days)
- **Week 3**: Integration, figures, writing (3-4 days)
- **Contingency**: Debugging (2-3 days)
- **Total**: 15-19 days → 2.5-3 weeks

### 💻 Code Changes (Minimal)

**New file**:
- `benchmarks/retrieval.py` (~200 lines)

**Modified files**:
- `benchmarks/scifact.py` (+50-100 lines)
- `benchmarks/qasper.py` (+50-100 lines)
- `benchmarks/run.py` (add CLI option)

**Reused**:
- benchmarks/nli.py (existing NLI evaluator)
- benchmarks/cache.py (SQLite caching)
- benchmarks/base.py (data structures)

---

## Next Steps

1. **Review** RETRIEVAL_EVAL_SUMMARY.txt (10 min)
2. **Decide**: Proceed? If yes, continue to step 3.
3. **Read** RETRIEVAL_EVAL_QUICK_START.md (15 min)
4. **Create** benchmarks/retrieval.py skeleton
5. **Start Week 1** following RETRIEVAL_EVAL_QUICK_START.md Phase 1
6. **Reference** RETRIEVAL_IMPLEMENTATION_STARTER.py for code patterns
7. **Validate** using testing checklist
8. **Write** Section 4.5 using content from RETRIEVAL_EVAL_SUMMARY.txt

---

## Document Version & Dates

- **Created**: 2026-03-27
- **For**: Detent.ai "Proof-Carrying Answers" paper
- **Status**: Ready for implementation
- **Next review**: After Week 1 results (2026-04-03)

---

## Questions?

Refer to:
- **"Why these benchmarks?"** → RETRIEVAL_EVALUATION_DESIGN.md Part 1–2
- **"How long will this take?"** → RETRIEVAL_EVAL_QUICK_START.md (timeline)
- **"What's the code structure?"** → RETRIEVAL_EVAL_QUICK_START.md (code architecture)
- **"Show me example code"** → RETRIEVAL_IMPLEMENTATION_STARTER.py
- **"What could go wrong?"** → RETRIEVAL_EVAL_SUMMARY.txt (risks section)
- **"What do the tables look like?"** → RETRIEVAL_EVALUATION_DESIGN.md Part 6 or RETRIEVAL_EVAL_SUMMARY.txt

---

**All files located**: `/sessions/gallant-eloquent-galileo/mnt/detent/pca-eval/docs/`
