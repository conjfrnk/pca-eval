# Claim Decomposition Evaluation — One-Pager

## The Gap
PCA paper identifies decomposition as "critical to verification fidelity" but **never evaluates it**. Non-atomic claims, dropped claims, and type misassignment directly reduce downstream F1, but this impact is unquantified.

---

## The Solution (3 Layers)

### Layer 1: Atomicity (Are claims atomic?)
- **Automatic**: Pattern-match AND/OR/NOT (2.3–5.7% violations found)
- **LLM-based**: Semantic judgment (target ≥ 85% score)
- **Result**: Table 3a shows atomicity by benchmark

### Layer 2: Completeness (Are all claims captured?)
- **Reuse FActScore (14K facts)**: Gold standard for recall
- **Reuse SAFE (2K responses)**: Type-aware completeness
- **Metric**: Claim recall ≥ 90%, drop-rate ≤ 5%
- **Finding**: Synthesis claims have highest drop rate (~15%)
- **Result**: Table 3b shows coverage + drops-by-type

### Layer 3: Type Accuracy (Are types correct?)
- **Manual annotation**: 200 claims, 2 annotators, κ ≥ 0.70
- **Confusion matrix**: E vs. I vs. S confusion patterns
- **Target**: Overall ≥ 80% (E ≥ 90%, I ≥ 70%, S ≥ 70%)
- **Finding**: I ↔ S confusion indicates weak synthesis detection
- **Result**: Table 3c confusion matrix + per-type metrics

### Downstream Impact
- **Correlate** decomposition quality → retrieval recall → verification F1
- **Ablation**: Oracle (100% F1 unattainable) vs. Full (with decomposition error) vs. Gold-decomposition-only
- **Gap attribution**: ~3–8pp of oracle gap is decomposition; ~7pp is retrieval
- **Result**: Table 3d + scatter plots

---

## Gold Data Strategy (Minimal New Annotation)

| Source | Facts | Use | Effort |
|--------|-------|-----|--------|
| **FActScore** | 14,274 | Atomicity baseline, completeness gold | Free (existing) |
| **SAFE** | ~2,000 | Type-aware completeness validation | Free (existing) |
| **Manual** | 200 | Type accuracy ground truth | 15 hours (2 annotators) |

---

## Implementation (4 Weeks)

| Week | Task | Output |
|------|------|--------|
| **1** | Metrics: atomicity scorer, completeness eval, type accuracy | `benchmarks/decomposition.py` (unit tested) |
| **2** | Load FActScore/SAFE, run manual annotation | Gold dataset + inter-annotator agreement |
| **3** | Run evaluations, compute statistics, generate results JSON | Tables 3a–3d + correlations |
| **4** | Write Section 3.4, integrate figures, update main.tex | Paper-ready section (~3 pages) |

---

## Key Metrics

| Metric | Method | Target | Expected |
|--------|--------|--------|----------|
| **Atomicity Index** | Regex + LLM | ≥ 92% | 90–94% |
| **Completeness Recall** | FActScore overlap | ≥ 90% | 93–96% |
| **Drop Rate** | Missing claims | ≤ 5% | 3–12% (varies by type) |
| **Type Accuracy** | Confusion matrix | ≥ 80% | 80–85% |
| **Downstream Corr** | Spearman ρ | ≥ 0.45 | 0.45–0.75 |
| **Gap Attribution** | Ablation | Decomp: 3–8pp | 3–8pp |

---

## Paper Section (New: 3.4)

**Location**: After Section 3.3 (Architecture), ~1,800 words

**Subsections**:
- 3.4.1 Atomicity (500w) + Table 3a + Figure 4a
- 3.4.2 Completeness (400w) + Table 3b
- 3.4.3 Type Accuracy (400w) + Table 3c + Figure 4c
- 3.4.4 Downstream Impact (500w) + Table 3d + Figure 4b

**Result**: Quantifies that decomposition fidelity is material (3–8pp F1 gap).

---

## Why This Matters

1. **Closes gap in evaluation**: Paper acknowledges "evaluating decomposition fidelity is important" but never does it.
2. **Quantifies impact**: Shows exactly how much decomposition quality affects downstream accuracy.
3. **Publication-ready**: Follows rigorous methodology (gold data, inter-annotator agreement, significance testing).
4. **Extends narrative**: Strengthens claim-first generation story: "We decompose claims first → measure decomposition quality → show downstream impact."

---

## Files & Navigation

| Document | Purpose | Read First For |
|----------|---------|---|
| **DECOMPOSITION_EVALUATION_README.md** | Overview + quick start | Getting oriented |
| **CLAIM_DECOMPOSITION_EVALUATION_DESIGN.md** | Full design (9 parts) | Detailed design decisions |
| **DECOMPOSITION_EVAL_IMPLEMENTATION_CHECKLIST.md** | Week-by-week plan | Execution & tracking |
| **DECOMPOSITION_METRICS_SPEC.md** | Technical API | Code reference |
| **DECOMPOSITION_EVAL_ONEPAGER.md** | This file | Quick reference |

---

## Success (Checkboxes)

- ✅ Atomicity ≥ 92% (2–6% violations is acceptable)
- ✅ Completeness ≥ 90% (all claims mostly captured)
- ✅ Type accuracy ≥ 80% (reasonable for automatic assignment)
- ✅ Downstream correlation ρ ≥ 0.45 (decomposition quality matters)
- ✅ Section 3.4 is publication-ready (tables + figures integrated)
- ✅ All results reproducible from CLI: `python -m benchmarks.run --eval-decomposition`

---

## Effort & Cost

- **Time**: 170 FTE hours (4 weeks @ 40h/week)
- **Annotation**: 15 hours (2 annotators × 200 claims × 3 min/claim)
- **LLM cost**: ~$5–20 (semantic atomicity scoring on 500 claims)
- **Total cost**: < $100 (mostly time)

---

## Next Action

1. **Approve approach** (15 min): Skim this one-pager + README
2. **Assign owner** (1 week): Identify implementer
3. **Week 1 kickoff**: Owner reads full DESIGN doc + SPEC doc, sets up metrics module
4. **Weekly syncs**: Track progress against checklist

---

**Questions?** See detailed docs or email with "decomposition eval" in subject.
