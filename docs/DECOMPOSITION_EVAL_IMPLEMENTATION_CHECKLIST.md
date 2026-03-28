# Claim Decomposition Evaluation — Implementation Checklist

## Phase 1: Metrics Infrastructure (Week 1)

### 1.1 Pattern-Based Atomicity Detector
- [ ] Create `benchmarks/decomposition.py` with core functions
- [ ] `detect_coordination_violations(claim: str) -> bool`
  - [ ] Match ` and `, ` or ` at top level (not in quotes)
  - [ ] Match negation chains: `not not`, `NOT NOT`
  - [ ] Match conditional + assertion patterns
  - [ ] Test on 50 sample claims from FActScore
- [ ] `coordination_index(claims: list[str]) -> float`
  - [ ] Returns (total − violations) / total
  - [ ] Handle edge cases (quoted text, proper names with "and")

### 1.2 LLM-as-Judge Semantic Atomicity
- [ ] Design prompt template (see Part 1.2 in design doc)
- [ ] Implement `semantic_atomicity_judge(claims: list[str], model="gpt-4-mini") -> dict`
  - [ ] Batch API calls for efficiency
  - [ ] Cache responses (in `benchmarks/cache.py`)
  - [ ] Parse A/B/C output and convert to score
  - [ ] Test inter-prompt consistency on 20 claims
- [ ] Set up fallback to open model (e.g., Llama-2-70B via together.ai) if OpenAI unavailable

### 1.3 Completeness Metrics
- [ ] `claim_recall(decomposed_claims: list[str], gold_claims: list[str]) -> float`
  - [ ] Use token overlap or embedding similarity for matching
  - [ ] Return |matched| / |gold_claims|
  - [ ] Include soft matching (paraphrases)
- [ ] `drop_rate(decomposed_claims: list[str], original_answer: str) -> float`
  - [ ] For each claim in original answer that's not in decomposition
  - [ ] Flag as "dropped"
  - [ ] Return count / total
- [ ] Analyze drops by type (E/I/S)

### 1.4 Type Accuracy Metrics
- [ ] `type_accuracy(predicted_types: list[str], gold_types: list[str]) -> dict`
  - [ ] Return overall accuracy + per-type precision/recall
  - [ ] Generate confusion matrix (numpy array)
  - [ ] Return Cohen's kappa if computing inter-annotator agreement
- [ ] Implement `compute_confusion_matrix(predicted, gold) -> pd.DataFrame`

### 1.5 Testing & Integration
- [ ] Create `tests/test_decomposition.py`
  - [ ] Test coordination detection (10 unit tests)
  - [ ] Test recall metric on FActScore sample (3 integration tests)
  - [ ] Test type accuracy on manual sample (2 integration tests)
- [ ] Ensure all metrics callable from CLI: `python -m benchmarks.run scifact --eval-decomposition`

---

## Phase 2: Gold Data & Annotation (Week 2)

### 2.1 FActScore Integration
- [ ] Download FActScore dataset (if not in pca-eval already)
  - [ ] 14K facts from biography generation task
  - [ ] Extract atomic facts (already decomposed in dataset)
  - [ ] Test loading: ~500 randomly sampled facts
- [ ] Implement `benchmarks/factscore_decomposition_eval.py`
  - [ ] Load FActScore biographies
  - [ ] Run PCA decomposition on biography text
  - [ ] Compute claim_recall against FActScore's atomic facts
  - [ ] Output: Coverage report (N facts, recall by model, drop-by-type breakdown)
- [ ] Generate Table 3b (FActScore row)

### 2.2 SAFE Integration
- [ ] Locate SAFE response decompositions (from Wei et al. 2024)
  - [ ] Confirm license (should be CC-BY)
  - [ ] Load ~2K responses
  - [ ] Extract gold decomposed claims
- [ ] Run PCA decomposition on same responses
- [ ] Compute claim_recall + type_accuracy against SAFE gold
- [ ] Output: Coverage report + type confusion matrix (SAFE row of Table 3b)

### 2.3 Manual Annotation (200 claims)
- [ ] **Annotation recruitment**:
  - [ ] Recruit 2 annotators (can be grad students or external)
  - [ ] Compute hourly rate (~$25–30/hr)
  - [ ] Total budget: 200 claims × 3 min/claim × 2 annotators = 20 hours ≈ $500–600
- [ ] **Annotation setup**:
  - [ ] Create annotation platform (CSV template or simple web UI)
    ```csv
    claim_id,claim_text,original_answer,atomicity_label,type_label,completeness
    1,"Company X founded in 1995","Company X was founded in 1995...",A,E,YES
    2,"Smith served as CEO for 8 years","...",B,I,YES
    ```
  - [ ] Include calibration: Have both annotators label 20 claims together first
  - [ ] Then split remaining 180 claims (90/90) + 10 overlap for agreement
- [ ] **Sample composition**:
  - [ ] SciFact: 50 claims (domain-specific, factual)
  - [ ] QASPER: 75 claims (multi-hop reasoning)
  - [ ] HAGRID: 75 claims (attribution, hallucination)
- [ ] **Annotation guidelines**:
  - [ ] Atomicity: A = can't split, B = has minor adjuncts, C = 2+ propositions
  - [ ] Type: E = direct assertion, I = inference, S = multi-source synthesis
  - [ ] Completeness: YES/NO = present in PCA decomposition output
- [ ] **Agreement computation**:
  - [ ] Compute Cohen's kappa on overlapping 10 claims
  - [ ] Target κ ≥ 0.70 (acceptable agreement)
  - [ ] If κ < 0.70, clarify guidelines and retrain

### 2.4 Data Consolidation
- [ ] Merge FActScore + SAFE + manual annotations into unified gold dataset
  - [ ] File: `benchmarks/gold_decomposition_annotations.json`
  - [ ] Schema: `{"claim_id": str, "text": str, "atomicity": str, "type": str, "completeness": bool, "source": str}`
- [ ] Test loading: `python -c "from benchmarks import load_gold_annotations; print(load_gold_annotations())"`

---

## Phase 3: Measurement & Results (Week 3)

### 3.1 Run Atomicity Evaluation
- [ ] Compute coordination index on all benchmarks
  ```bash
  python -m benchmarks.evaluate_atomicity --benchmark scifact,qasper,hagrid,factscore,safe
  ```
- [ ] Output: `results/atomicity_scores.json`
  ```json
  {
    "scifact": {"n_claims": 1850, "violations_pct": 2.3, "semantic_score": 0.94},
    ...
  }
  ```
- [ ] Generate Table 3a from results

### 3.2 Run Completeness Evaluation
- [ ] FActScore:
  ```bash
  python -m benchmarks.evaluate_completeness --source factscore --sample 5000
  ```
- [ ] SAFE:
  ```bash
  python -m benchmarks.evaluate_completeness --source safe --sample 1200
  ```
- [ ] Output: `results/completeness_scores.json` with per-claim recall + type breakdown
- [ ] Generate Table 3b from results

### 3.3 Run Type Accuracy Evaluation
- [ ] Load manual annotations
- [ ] For each claim, extract predicted type from PCA decomposition
- [ ] Compute confusion matrix + overall accuracy
  ```bash
  python -m benchmarks.evaluate_type_accuracy --manual-annotations benchmarks/gold_decomposition_annotations.json
  ```
- [ ] Output: `results/type_accuracy.json`
  ```json
  {
    "overall_accuracy": 0.814,
    "confusion_matrix": [[187, 12, 1], [18, 69, 13], [2, 21, 47]],
    "per_type": {"E": {"precision": 0.903, "recall": 0.935}, ...}
  }
  ```
- [ ] Generate Table 3c from results

### 3.4 Downstream Impact Analysis
- [ ] Instrument retrieval + verification pipeline to log:
  - [ ] For each claim: atomicity score, completeness, type_accuracy, retrieval_recall@5, nli_f1
- [ ] Run full pipeline on:
  - [ ] SciFact dev set (300 claims)
  - [ ] QASPER dev set (1.7K Q&A, ~2.5K claims)
- [ ] Compute Pearson & Spearman correlations:
  ```bash
  python -m benchmarks.analyze_downstream_impact --input results/full_pipeline_metrics.json
  ```
- [ ] Output: `results/correlation_analysis.json` + scatter plots
- [ ] Implement ablation:
  - [ ] Pipeline A (Full): Decompose → Retrieve → Verify
  - [ ] Pipeline B (Oracle): Gold evidence → Verify
  - [ ] Pipeline C (Gold Decomp): Gold atoms → Retrieve → Verify
  - [ ] Compute gaps and attribution
- [ ] Output: Table 3d results

### 3.5 Statistical Validation
- [ ] For each metric, compute 95% bootstrap confidence intervals
  ```bash
  python -m benchmarks.bootstrap_ci --metric atomicity --n 10000
  ```
- [ ] McNemar's test (if comparing two decomposition methods):
  ```bash
  python -m benchmarks.mcnemar_test --method_a results/... --method_b results/...
  ```

---

## Phase 4: Paper Integration & Writing (Week 4)

### 4.1 Section 3.4 Draft
- [ ] Create `paper/section_3_4_decomposition_eval.tex`
- [ ] Write 3.4.1 (Atomicity): ~500 words
  - [ ] Define atomicity; show examples
  - [ ] Describe metrics; reference Table 3a
  - [ ] Interpret results
- [ ] Write 3.4.2 (Completeness): ~400 words
  - [ ] Define coverage; explain FActScore/SAFE reuse
  - [ ] Reference Table 3b; discuss drop-by-type findings
- [ ] Write 3.4.3 (Type Accuracy): ~400 words
  - [ ] Describe annotation procedure + inter-annotator agreement
  - [ ] Reference Table 3c; analyze confusions
- [ ] Write 3.4.4 (Downstream Impact): ~500 words
  - [ ] Explain correlation analysis + ablation
  - [ ] Reference Table 3d + scatter plot
  - [ ] Discuss gap attribution

### 4.2 Tables & Figures
- [ ] Generate publication-ready tables:
  - [ ] Table 3a (Atomicity): Use `booktabs` style
  - [ ] Table 3b (Completeness): Aligned columns, 95% CI notation
  - [ ] Table 3c (Type Accuracy): Confusion matrix with marginals
  - [ ] Table 3d (Downstream Impact): Clean layout
- [ ] Generate figures:
  - [ ] Figure 4a: Scatter plot (atomicity vs. retrieval_recall@5) with regression line
  - [ ] Figure 4b: Bar chart (decomposition impact vs. retrieval impact gap)
  - [ ] Figure 4c (optional): Heatmap (confusion matrix from Table 3c)
- [ ] All figures should use Detent color palette (Racing Green accent)

### 4.3 Integration into main.tex
- [ ] Add `\input{section_3_4_decomposition_eval.tex}` after Section 3.3 (Architecture)
- [ ] Update bibliography (if new citations added)
- [ ] Verify cross-references: `\cref{tab:atomicity}`, `\cref{fig:downstream}`
- [ ] Update table of contents (if using `\tableofcontents`)

### 4.4 Appendix (Optional)
- [ ] Create Appendix A: Full Annotation Guidelines
  - [ ] Include example annotations (5–10 examples per task)
  - [ ] Explain edge cases (quoted text, implicit claims, etc.)
- [ ] Create Appendix B: Ablation Details
  - [ ] Detailed pseudo-code for oracle vs. full pipeline comparison
  - [ ] Discussion of why decomposition/retrieval gap is expected

### 4.5 Final Quality Checks
- [ ] Run pdflatex to ensure no compilation errors
- [ ] Verify all tables render correctly
- [ ] Spot-check figure quality (resolution, colors)
- [ ] Proofread Section 3.4 for typos + clarity
- [ ] Ensure notation is consistent with rest of paper (e.g., $c_i$ for claims)

---

## Phase 5: Release & Docs (If Time)

### 5.1 Code Release
- [ ] Ensure all code in `benchmarks/decomposition.py` is well-commented
- [ ] Add docstrings to all public functions
- [ ] Test all functions end-to-end on sample inputs
- [ ] Tag in git: `git tag -a decomposition-eval-v1 -m "Claim decomposition evaluation"`

### 5.2 Reproduction Instructions
- [ ] Update `pca-eval/README.md`:
  ```bash
  # Run decomposition evaluation
  python -m benchmarks.run scifact --eval-decomposition
  python -m benchmarks.evaluate_atomicity --benchmark all
  python -m benchmarks.analyze_downstream_impact
  ```
- [ ] Document expected output locations

---

## Contingency Plans

**If FActScore/SAFE integration takes longer than expected**:
- Focus on manual annotation + SciFact + QASPER (highest-ROI subset)
- Skip full FActScore coverage; sample 1K facts instead

**If inter-annotator agreement is poor (κ < 0.70)**:
- Revise annotation guidelines; focus on harder cases
- Add third annotator for high-disagreement claims
- Reduce scope to 100 high-confidence claims

**If correlation analysis shows weak relationships**:
- Check if decomposition variance is too low (ceiling effect)
- Analyze which benchmarks show strong vs. weak correlations
- Report null result honestly; it may indicate decomposition is robust to imperfection

---

## Time Tracking

Track progress:

| Task | Planned | Actual | Status |
|------|---------|--------|--------|
| Week 1 Metrics | 40h | | |
| Week 2 Annotation | 50h | | |
| Week 3 Analysis | 40h | | |
| Week 4 Writing | 40h | | |
| **Total** | **170h (~4 weeks FTE)** | | |

---

## Success Indicators (✓ when complete)

- [ ] All metrics passing unit tests
- [ ] FActScore coverage ≥ 90%
- [ ] Manual annotation κ ≥ 0.70
- [ ] Type accuracy ≥ 80%
- [ ] Downstream correlation ρ ≥ 0.45 (p < 0.05)
- [ ] Paper section drafted + tables generated
- [ ] All results reproducible from code
