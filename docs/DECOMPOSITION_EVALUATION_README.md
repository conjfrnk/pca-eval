# Claim Decomposition Evaluation — Complete Framework

## Overview

This directory contains a comprehensive evaluation framework for claim decomposition in the Proof-Carrying Answers (PCA) system. The paper identifies decomposition quality as critical to verification fidelity but currently does **not evaluate it**. These three documents close that gap.

**The problem**: If decomposition produces non-atomic claims, drops claims, or mislabels types, the downstream verification accuracy suffers. This framework measures all three layers and their downstream impact.

---

## Three Documents, Three Purposes

### 1. CLAIM_DECOMPOSITION_EVALUATION_DESIGN.md (668 lines)
**The "What & Why" — Complete research design**

- Defines three evaluation layers: **Atomicity**, **Completeness**, **Type Accuracy**
- Explains metrics, gold annotation strategy, and sample size calculations
- Shows how to reuse existing datasets (FActScore: 14K facts, SAFE: 2K responses)
- Proposes hybrid approach: automatic detection + LLM-as-judge + human validation
- Includes downstream impact tracing: decomposition quality → retrieval recall → verification F1
- Provides concrete subsection structure + table formats ready for paper integration
- Expected output: All tables (3a–3d) and figures for paper

**Start here** if you need to understand the complete evaluation design.

---

### 2. DECOMPOSITION_EVAL_IMPLEMENTATION_CHECKLIST.md (299 lines)
**The "How" — Week-by-week implementation plan**

- **Phase 1 (Week 1)**: Metrics infrastructure
  - Pattern-based atomicity detector (AND/OR/NOT chains)
  - LLM-as-judge semantic atomicity scorer
  - Completeness metrics (recall, drop-rate)
  - Type accuracy metrics (confusion matrix)
- **Phase 2 (Week 2)**: Gold annotation & data integration
  - FActScore integration (14K facts)
  - SAFE integration (2K responses)
  - Manual annotation (200 claims, 2 annotators, κ ≥ 0.70)
- **Phase 3 (Week 3)**: Measurement & results
  - Run all evaluations on 5 benchmarks
  - Compute bootstrap confidence intervals
  - Statistical significance testing (McNemar's test, Spearman correlation)
  - Generate results JSON
- **Phase 4 (Week 4)**: Paper integration
  - Draft Section 3.4 (Claim Decomposition Evaluation)
  - Generate publication-ready tables + figures
  - Integrate into main.tex

**Use this** to track progress and execute implementation.

---

### 3. DECOMPOSITION_METRICS_SPEC.md (573 lines)
**The "Specifications" — Technical API & architecture**

- Class definitions: `AtomicityScorer`, `CompletenessEvaluator`, `TypeAccuracyEvaluator`, `DownstreamTracer`
- Each class with full interface, metrics, implementation notes
- CLI integration: `python -m benchmarks.run scifact --eval-decomposition`
- Output file structure and JSON schema
- Dependencies, testing strategy, backward compatibility
- Example outputs and expected tables

**Use this** when coding the metrics module.

---

## Quick Start

### For the Reader (Understanding the Design)
1. Read **CLAIM_DECOMPOSITION_EVALUATION_DESIGN.md** Part 1–2 (Metrics + Gold Data Strategy)
2. Skim Part 6 (Subsection Structure) to see how it fits in the paper
3. Refer to Part 7 (Tables) for expected output format

### For the Implementer (Building the Code)
1. Read **DECOMPOSITION_EVAL_IMPLEMENTATION_CHECKLIST.md** Phases 1–2 (Weeks 1–2)
2. Reference **DECOMPOSITION_METRICS_SPEC.md** for API design
3. Execute Phase 3–4 while consulting **CLAIM_DECOMPOSITION_EVALUATION_DESIGN.md** Part 5 (Downstream Impact)

### For the Writer (Paper Integration)
1. Use **CLAIM_DECOMPOSITION_EVALUATION_DESIGN.md** Part 6 (Subsection Structure)
2. Insert generated tables (Part 7) into main.tex
3. Generate figures from **DECOMPOSITION_EVAL_IMPLEMENTATION_CHECKLIST.md** Phase 4.2

---

## Key Design Decisions

### 1. Reuse Existing Data (No New Annotation Overhead)
- **FActScore (14K facts)**: Use as gold standard for atomicity baseline + completeness eval
- **SAFE (2K responses)**: Use for completeness + type accuracy validation
- **Manual annotation (200 claims)**: Only for type accuracy validation (higher-risk, requires careful labeling)

**Benefit**: Reduces annotation burden from ~500 claims → 200 claims; leverages published datasets.

### 2. Hybrid Evaluation Approach
- **Automatic (Pattern-based)**: Coordinate operators (AND/OR/NOT), fast, 100% coverage
- **LLM-as-Judge (Semantic)**: Nuanced atomicity scoring, ~500 stratified claims, ~$5–20 cost
- **Human Annotation**: Type accuracy (highest signal), 200 claims, 2 annotators, κ ≥ 0.70

**Benefit**: Balances cost, coverage, and signal quality. Not all metrics need human annotation.

### 3. Downstream Impact as Validation
- **Core insight**: Poor decomposition → lower retrieval recall → lower verification F1
- **Ablation design**: Oracle (gold evidence) vs. Full (real decomposition) vs. Gold-decomposition-only
- **Gap attribution**: Isolate impact of decomposition from impact of retrieval

**Benefit**: Provides empirical evidence that decomposition quality matters for the real task.

### 4. Statistical Rigor
- **Bootstrap confidence intervals** (95% CI) for all metrics
- **Significance testing** (McNemar's test, Spearman correlation, p-values)
- **Stratified analysis** by claim type (E/I/S), domain (SciFact/QASPER/HAGRID)

**Benefit**: Results are defensible; no hand-waving about uncertainty.

---

## Metrics at a Glance

| Layer | Metric | Method | Coverage | Target |
|-------|--------|--------|----------|--------|
| **Atomicity** | Coordination Index | Automatic regex | 100% of claims | ≥ 95% |
| **Atomicity** | Semantic Atomicity | LLM-as-judge | ~500 claims | ≥ 85% |
| **Completeness** | Claim Recall | Overlap vs. FActScore/SAFE | 3K–5K facts | ≥ 90% |
| **Completeness** | Drop Rate | Missing claims | 3K–5K facts | ≤ 5% |
| **Type Accuracy** | Type Accuracy | Confusion matrix | 200 manual | ≥ 80% |
| **Downstream** | Correlation | Spearman ρ | 500+ claims | ρ ≥ 0.45 |
| **Downstream** | Gap Analysis | Ablation (Oracle vs Full) | Full pipeline | 3–8pp decomposition gap |

---

## Expected Timeline & Effort

| Phase | Duration | Effort | Key Output |
|-------|----------|--------|-----------|
| 1. Metrics implementation | 1 week | 40h | `benchmarks/decomposition.py` + unit tests |
| 2. Gold data + annotation | 1 week | 50h | 200 manual annotations + FActScore/SAFE integration |
| 3. Measurement & analysis | 1 week | 40h | Results JSON + correlation analysis |
| 4. Paper writing | 1 week | 40h | Section 3.4 + tables + figures |
| **Total** | **4 weeks** | **170h (FTE)** | **Publishable paper section** |

---

## Paper Integration

### Proposed Location
After Section 3.3 (Architecture), before Section 4 (Experiments).

### New Section: 3.4 Claim Decomposition Evaluation
```
3.4.1 Atomicity (500 words)
3.4.2 Completeness (400 words)
3.4.3 Type Accuracy (400 words)
3.4.4 Downstream Impact (500 words)
───────────────────────────
Total: ~1,800 words (~3 pages with tables/figures)
```

### Tables to Add
- **Table 3a**: Atomicity metrics by benchmark
- **Table 3b**: Completeness (recall + drop-by-type)
- **Table 3c**: Type accuracy confusion matrix + per-type metrics
- **Table 3d**: Downstream impact correlations + ablation gap

### Figures to Add
- **Figure 4a**: Scatter plot (atomicity vs. retrieval recall)
- **Figure 4b**: Bar chart (decomposition vs. retrieval impact)
- **Figure 4c** (optional): Heatmap (type confusion matrix)

---

## Contingency Plans

**If FActScore/SAFE integration is slower than expected**:
- Focus on manual annotation + SciFact/QASPER only (highest-ROI subset)
- Sample 1K facts from FActScore instead of full dataset

**If manual annotation inter-annotator agreement is poor (κ < 0.70)**:
- Revise annotation guidelines
- Add third annotator for high-disagreement claims
- Reduce scope to 100 high-confidence claims

**If downstream correlation is weak**:
- Check for ceiling effect (if decomposition variance is low)
- Report stratified results (which benchmarks show strong correlation?)
- Honestly report null finding; may indicate decomposition is robust

---

## Success Criteria

✅ **All metrics meet targets:**
- Atomicity ≥ 92%
- Completeness ≥ 90%
- Type Accuracy ≥ 80%
- Downstream correlation ρ ≥ 0.45 (p < 0.05)

✅ **Results are reproducible:**
- All code in `benchmarks/decomposition.py`
- CLI flags and output documented
- Results JSON generated and checked into results/

✅ **Paper section is publication-ready:**
- Written in conference style
- Tables are self-contained
- Figures have captions + color
- Cross-references work (\\cref{tab:...}, \\cref{fig:...})

---

## File Manifest

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| CLAIM_DECOMPOSITION_EVALUATION_DESIGN.md | Complete research design | 668 | ✅ Ready |
| DECOMPOSITION_EVAL_IMPLEMENTATION_CHECKLIST.md | Implementation plan (4 weeks) | 299 | ✅ Ready |
| DECOMPOSITION_METRICS_SPEC.md | Technical API specification | 573 | ✅ Ready |
| DECOMPOSITION_EVALUATION_README.md | This file (overview) | TBD | ✅ Ready |

---

## Next Steps

1. **Read & calibrate** (1–2 hours):
   - Skim all three documents
   - Ensure the approach fits project constraints
   - Confirm dataset availability (FActScore, SAFE)

2. **Assign owner** (~4 weeks):
   - Assign implementer (solo researcher or junior engineer)
   - Allocate 170 FTE hours
   - Weekly check-ins

3. **Milestone tracking** (weekly):
   - Week 1: Metrics module + unit tests
   - Week 2: Gold data integration + manual annotation
   - Week 3: All results + statistical tests
   - Week 4: Paper draft + figures

4. **Integration** (Week 5):
   - Merge metrics module into main codebase
   - Update pca-eval README with new CLI flags
   - Tag release: `decomposition-eval-v1`

---

## Questions?

- **On metrics design**: See CLAIM_DECOMPOSITION_EVALUATION_DESIGN.md Part 1–3
- **On implementation**: See DECOMPOSITION_EVAL_IMPLEMENTATION_CHECKLIST.md
- **On technical details**: See DECOMPOSITION_METRICS_SPEC.md Section 1–8
- **On paper integration**: See CLAIM_DECOMPOSITION_EVALUATION_DESIGN.md Part 6–7

All three documents are cross-referenced for easy navigation.
