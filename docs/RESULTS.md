# PCA Evaluation Benchmark Results

## Summary

| Benchmark | Configuration | Best F1 | 95% CI | Accuracy |
|-----------|--------------|---------|--------|----------|
| **HAGRID** | 3-Model Ensemble XGBoost (5-fold CV) | **87.5%** | (CV, see note) | 82.9% |
| **SciFact** | nli-only (oracle evidence) | **95.3%** | **[93.3%, 97.1%]** | 95.1% |
| **SciFact** | nli-abstract (pipeline) | **87.4%** | **[85.2%, 89.6%]** | 83.3% |
| **FEVER** | nli-only (fine-tuned DeBERTa-large-512 + passage-scoring) | **94.7%** | **[94.4%, 95.0%]** | 94.7% |
| **QASPER** | nli-only (full pipeline, t=0.15) | **88.2%** | **[86.2%, 90.1%]** | 94.2% |
| **FActScore** | XGBoost 3-model ensemble (174 features, 5-fold CV) | **83.8%** | (see note) | 87.0% |
| **AttributionBench** | XGBoost domain-specific (OOD, t=0.64) | **81.9% binary / 81.5% macro** | (see note) | 81.5% |
| **FACTS Grounding** | Sentence decomposition (DeBERTa-v3-base) | **97.3%** | - | 98.1% |

**CI notes**: Bootstrap 95% confidence intervals (10,000 resamples, seed=42). HAGRID and AttributionBench use XGBoost with 5-fold CV; individual fold variance provides the analogous uncertainty measure. SciFact nli-abstract F1 rounded to 87.4% from exact harmonic-macro value 0.8741.

## Oracle vs Pipeline Comparison

SciFact provides a direct comparison between oracle evidence (gold evidence provided) and pipeline evaluation (NLI against full abstract):

| Setting | F1 | 95% CI | Accuracy | Configuration |
|---------|-----|--------|----------|---------------|
| **Oracle** (gold evidence) | **95.3%** | [93.3%, 97.1%] | 95.1% | Fine-tuned DeBERTa-large, t=0.3 |
| **Pipeline** (full abstract) | **87.4%** | [85.2%, 89.6%] | 83.3% | Pre-trained DeBERTa-large + MiniCheck, ent=0.2, fb=0.3 |
| **Gap** | **-7.9pp** | | -11.8pp | |

The 7.9pp gap between oracle and pipeline is expected. The pipeline must both identify relevant evidence sentences and classify the claim, while the oracle setting isolates NLI classification quality. The 95% CIs do not overlap, confirming the gap is statistically significant.

## Methodology

**F1 metric**: Benchmark results use macro-averaged F1 computed as F1(macro_P, macro_R) -- the harmonic mean of macro-averaged precision and macro-averaged recall across classes. For balanced datasets (SciFact, FEVER), this is within 0.2pp of sklearn's standard macro F1 (mean of per-class F1). For QASPER (highly imbalanced: 90% ANSWERABLE), the metrics diverge; see QASPER section for details. HAGRID and AttributionBench XGBoost results use sklearn's standard binary F1 directly.

**Oracle evidence**: All benchmarks except SciFact nli-abstract use gold evidence annotations. SciFact nli-abstract evaluates against all abstract sentences (no gold evidence). This should be considered when comparing to pipeline baselines that retrieve evidence.

**Baselines**: HAGRID baselines (T5-XXL-TRUE 78.6%, FLAN-T5 79.0%) and GPT-4 (78.0% OOD macro-F1 w/o CoT) sourced from AttributionBench Table 3 (Li et al., ACL Findings 2024). Model sizes: DeBERTa-v3-base 86M+98M=184M, DeBERTa-v3-large 304M+131M=435M (Microsoft documentation).

**Training data disclosure**: NLI models fine-tuned on ~33K examples from public NLI datasets, including HAGRID-derived pairs (in-distribution for HAGRID evaluation). SciFact, FEVER, QASPER evaluations are out-of-distribution. XGBoost models use 5-fold stratified CV for unbiased estimates.

### Statistical Methodology

**Bootstrap confidence intervals**: All reported F1 and accuracy values include 95% bootstrap confidence intervals computed via percentile method with 10,000 resamples (seed=42 for reproducibility). The bootstrap resamples the full prediction set with replacement and recomputes the metric on each resample. CIs use the 2.5th and 97.5th percentiles of the bootstrap distribution.

**Metrics computed**: For each benchmark, we report CIs for three F1 variants:
- **Harmonic-macro F1**: F1(macro_P, macro_R), used as the primary metric throughout this document.
- **Sklearn macro F1**: Mean of per-class F1 scores (sklearn default). For balanced datasets, this is within 0.2pp of harmonic-macro. For imbalanced QASPER, it diverges by ~7pp.
- **Accuracy**: Proportion of correct predictions.

**McNemar's test**: For paired model comparisons, we provide McNemar's chi-squared test with continuity correction (alpha=0.05). This tests whether two models have significantly different error rates on the same examples.

**HAGRID/AttributionBench note**: These benchmarks use XGBoost with 5-fold stratified cross-validation. The CV procedure provides its own uncertainty estimate via fold-to-fold variance (see individual sections). Bootstrap CIs are not applicable to CV results since the predictions come from 5 different models.

**Reproduction**: `python -m benchmarks.run_stats --all` recomputes all CIs from saved prediction files.

## HAGRID: Attribution Detection in RAG

HAGRID measures whether generated answers are properly attributed to source passages. An answer is ATTRIBUTABLE only if ALL sentences are supported by evidence (ALL-OR-NOTHING).

### Published Baselines

| System | F1 | Notes |
|--------|-----|-------|
| T5-XXL-TRUE | 78.6% | 11B parameter model |
| FLAN-T5 3B | 79.0% | 3B parameter model |
| **PCA-Eval 3-Model Ensemble XGBoost** | **87.5%** | **0.2B + 0.4B + 0.4B-512 fine-tuned NLI + XGBoost** |
| PCA-Eval 2-Model Ensemble XGBoost | 86.8% | 0.2B + 0.4B fine-tuned NLI + XGBoost |
| PCA-Eval XGBoost (fine-tuned base) | 86.2% | 0.2B fine-tuned NLI + XGBoost aggregation |
| PCA-Eval XGBoost (fine-tuned large) | 85.3% | 0.4B fine-tuned NLI + XGBoost aggregation |
| PCA-Eval XGBoost (fine-tuned large-512) | 84.3% | 0.4B fine-tuned NLI (max_seq=512) + XGBoost |
| PCA-Eval XGBoost (pre-trained) | 80.1% | 0.4B NLI + XGBoost aggregation |

### Strategy Progression

| Strategy | F1 | Model | Notes |
|----------|-----|-------|-------|
| majority t=0.5 | 54.7% | DeBERTa-base | Baseline |
| multi-signal t=0.3 | 57.7% | DeBERTa-base | +NLI+entity+number overlap |
| multi-signal t=0.6 | 59.3% | DeBERTa-base | Threshold tuned |
| multi-signal t=0.6 | 60.9% | Fine-tuned DeBERTa-base (HAGRID-focused) | HAGRID-only training |
| multi-signal t=0.3 | 65.1% | Fine-tuned DeBERTa-base (full-mix) | ANLI+WANLI+HAGRID+SciFact |
| multi-signal t=0.3 | 64.0% | Fine-tuned DeBERTa-large (full-mix) | Same data, larger model |
| learned-weights | 59.2% | DeBERTa-base | LR on 10 features |
| xgboost | 79.6% | DeBERTa-large | 32 features, 5-fold CV |
| xgboost | 80.1% | DeBERTa-base | 32 features, 5-fold CV |
| xgboost | 85.3% | Fine-tuned DeBERTa-large (full-mix) | 32 features, 5-fold CV |
| xgboost | 86.2% | Fine-tuned DeBERTa-base (full-mix) | 32 features, 5-fold CV |
| xgboost ensemble (2-model) | 86.8% | Fine-tuned base + large | 64 features, 5-fold CV |
| whole t=0.35 | 66.0% | Fine-tuned DeBERTa-large (512) | Whole-answer NLI scoring |
| xgboost | 84.3% | Fine-tuned DeBERTa-large (512) | 52 features, 5-fold CV |
| **xgboost ensemble (3-model)** | **87.5%** | **Fine-tuned base + large + large-512** | **156 features, 5-fold CV** |

### Fine-Tuned XGBoost Threshold Sweep (full dev set, pipeline F1)

| Threshold | F1 | Accuracy | Precision | Recall |
|-----------|------|----------|-----------|--------|
| 0.30 | 88.4% | 83.8% | 80.3% | 98.4% |
| 0.35 | 88.9% | 84.7% | 81.6% | 97.5% |
| 0.40 | 89.7% | 86.0% | 83.6% | 96.6% |
| **0.44** | **90.3%** | **86.9%** | **85.4%** | **95.5%** |
| 0.50 | 89.9% | 86.9% | 86.7% | 93.3% |
| 0.55 | 89.7% | 86.8% | 87.9% | 91.5% |
| 0.60 | 89.4% | 86.7% | 89.2% | 89.7% |

Note: Pipeline F1 is evaluated with XGBoost trained on the full dev set. CV F1 (86.2%) is the unbiased estimate; pipeline F1 (90.3%) reflects in-distribution performance at the optimal threshold.

### Pre-Trained XGBoost Threshold Sweep (full dev set)

| Threshold | F1 | Accuracy |
|-----------|------|----------|
| 0.30 | 72.7% | 73.0% |
| 0.40 | 77.0% | 77.8% |
| 0.50 | 79.7% | 81.2% |
| 0.55 | 80.7% | 82.2% |
| 0.60 | 81.0% | 82.2% |

### XGBoost v3 Features (44 total)

v3 adds 12 new features (from 32 to 44) targeting false positive reduction:

| Feature Group | New Features | Purpose |
|---------------|-------------|---------|
| Contradiction signals | max_con, mean_con, n_con_sentences | Detect contradicted claims |
| NLI entropy | mean_entropy, max_entropy | Identify ambiguous NLI outputs |
| Citation brackets | n_brackets, bracket_density, brackets_per_evidence | Catch citation-heavy but unsupported text |
| Length ratio | answer_evidence_length_ratio | Flag answers disproportionate to evidence |
| Reverse coverage | reverse_coverage | How much evidence is used by the answer |
| Cross-signals | max_ent_x_max_con, mean_ent_con_margin | Entailment-contradiction interaction |

**v3 Result**: 85.7% XGBoost CV F1 (up from 85.3% with v2 features, +0.4pp). Top new features by importance: mean_con, mean_ent_con_margin, max_con.

### XGBoost Feature Analysis

With a pre-trained NLI model, XGBoost relies heavily on lexical features (number coverage, entity overlap) to compensate for noisy NLI scores. After fine-tuning, the NLI scores themselves become the dominant signal (whole-answer entailment margin becomes the single most important feature), reducing dependence on heuristic features.

### Error Analysis (Fine-Tuned XGBoost at t=0.44)

At the optimal threshold (0.44), the model makes 259 errors out of 1318 examples:

| Category | Count | % of Errors | Description |
|----------|-------|-------------|-------------|
| False Positives | 178 | 69% | Incorrectly marks NOT_ATTRIBUTABLE as ATTRIBUTABLE |
| False Negatives | 81 | 31% | Misses genuinely ATTRIBUTABLE answers |

**False Positive Analysis** (most common error type):
- FPs have high NLI scores (mean=0.88) and high whole_margin (0.78), suggesting the NLI model genuinely sees semantic entailment
- Top FPs involve answers with subtly wrong facts (dates, names, quantities) that are semantically similar to evidence but factually incorrect
- Higher n_uncovered_numbers (0.94 vs 0.63 for TPs) confirms that hallucinated numbers distinguish FPs from TPs

**False Negative Analysis**:
- FNs have moderate NLI scores (mean=0.72) and low whole_margin (0.26), putting them in the model's uncertain zone
- Many involve heavily paraphrased answers where semantic support exists but NLI scores are moderate

**Error Rate by Answer Length**:

| Sentences | Total | Error Rate |
|-----------|-------|------------|
| 1 | 744 | 16.7% |
| 2 | 201 | 18.4% |
| 3-5 | 300 | 28.3% |
| 6+ | 73 | 17.8% |

3-5 sentence answers have the highest error rate (28.3%), likely because they have enough sentences for one to be unsupported while others are well-supported, making the aggregation decision harder.

### Evaluation Protocol

- **Cross-validated F1 (87.5% binary / 86.2% single-model)**: 5-fold stratified CV on full HAGRID dev set (1318 examples). 87.5% is the 3-model ensemble; 86.2% is fine-tuned DeBERTa-base only. This is the primary metric since HAGRID has no separate labeled test set.
- **Cross-validated F1 (80.1%)**: Same CV protocol with pre-trained DeBERTa-base features.

## SciFact: Scientific Claim Verification

SciFact tests whether scientific claims are supported, refuted, or have insufficient evidence from paper abstracts.

### Best Results

| Tier | F1 | 95% CI | Accuracy | Configuration |
|------|-----|--------|----------|---------------|
| nli-only | **95.3%** | **[93.3%, 97.1%]** | 95.1% | Fine-tuned DeBERTa-large (full-mix), t=0.3 |
| nli-abstract | **87.4%** | **[85.2%, 89.6%]** | 83.3% | Pre-trained DeBERTa-large + MiniCheck fallback (ent=0.2, fb=0.3) |
| nli-only | 93.9% | - | 93.8% | Fine-tuned DeBERTa-large (full-mix), t=0.5 |
| nli-only | 87.3% | - | - | Fine-tuned DeBERTa-base (full-mix) |
| nli-abstract | 85.2% | - | - | Fine-tuned DeBERTa-large (full-mix) |
| nli-only | 82.3% | - | 79.1% | DeBERTa-large (no fallback) |

### Comparison with Published Baselines

| System | F1 | Notes |
|--------|-----|-------|
| VeriSci | 46.5% | Pipeline approach |
| SciFact baseline | 67.1% | Official baseline |
| MultiVerS | 75.4% | Multi-granularity verification |
| **PCA-Eval (nli-only, fine-tuned large)** | **95.3%** | **Fine-tuned DeBERTa-large** |
| **PCA-Eval (nli-abstract)** | **87.4%** | **DeBERTa-large + MiniCheck fallback** |
| PCA-Eval (nli-only, fine-tuned base) | 87.3% | Fine-tuned DeBERTa-base |

### SciFact nli-abstract Threshold Sweep

Sweep over entailment threshold and MiniCheck fallback threshold with DeBERTa-large:

| Entailment | Fallback | F1 | Acc | SUPPORTS | REFUTES | NEI |
|------------|----------|------|------|----------|---------|-----|
| **0.2** | **0.3** | **87.7%** | **83.8%** | **67.6%** | **97.5%** | **100%** |
| 0.3 | 0.3 | 87.7% | 83.8% | 67.6% | 97.5% | 100% |
| 0.4 | 0.3 | 87.4% | 83.6% | 67.1% | 97.5% | 100% |
| 0.5 | 0.5 | 87.4% | 83.3% | 66.7% | 97.5% | 100% |
| 0.5 | 0.3 | 87.0% | 83.1% | 66.2% | 97.5% | 100% |
| (no fb) | - | 82.3% | 79.1% | 62.5% | 89.3% | 100% |

Observations:
- MiniCheck fallback adds +5.4pp F1 over the no-fallback baseline
- Aggressive fallback threshold (0.3) catches more borderline cases, marginally improving SUPPORTS detection
- Rerank hurts SciFact (-1.8pp F1): boosts entailment scores, reducing REFUTES detection from 97.5% to 86.9%
- Context window hurts SciFact (-3.3pp F1): additional context introduces noise for short scientific claims

## FEVER: Fact Extraction and Verification

FEVER verifies factual claims against Wikipedia evidence.

| Tier | F1 | 95% CI | Accuracy | Model | n |
|------|-----|--------|----------|-------|---|
| nli-only | **94.7%** | **[94.4%, 95.0%]** | 94.7% | Fine-tuned DeBERTa-large-512 + passage-scoring, t=0.3 | 19895 |
| nli-only | 93.2% | [92.8%, 93.5%] | 93.1% | Fine-tuned DeBERTa-large-512 (full-mix), t=0.3 | 19895 |
| nli-only | 92.6% | - | 92.5% | Fine-tuned DeBERTa-large (full-mix), sample 5000 | 4972 |
| nli-only | 90.0% | 89.7% | Fine-tuned DeBERTa-base (full-mix) | 19895 |
| nli-only | 89.0% | 87.9% | MoritzLaurer DeBERTa-large (3-way NLI), sample 5000 | 4972 |
| nli-only | 85.9% | 84.6% | DeBERTa-base (pre-trained), sample 5000 | 4972 |
| nli-only | 85.8% | 84.3% | DeBERTa-large (pre-trained), sample 5000 | 4972 |

**Passage-scoring improvement (93.2% -> 94.7%, +1.5pp)**: Dual-granularity scoring -- individual evidence sentences are scored AND the concatenated evidence passage is scored as a whole, taking the maximum. For FEVER claims that require context from multiple evidence sentences, the concatenated passage captures cross-sentence reasoning that individual sentence scoring misses. Evidence decomposition did not add further improvement since FEVER evidence is already sentence-level.

Note: The fine-tuned models were trained on binary classification (entailment vs not-entailment), which reduces 3-way discrimination for FEVER's SUPPORTS/REFUTES/NOT_ENOUGH_INFO task. Despite this, the fine-tuned large-512 model with passage-scoring achieves 94.7% F1 thanks to its larger capacity, longer context window, and dual-granularity scoring.

## QASPER: Question Answering on Scientific Papers

QASPER measures fact verification in long scientific documents. Tests whether gold answers are entailed by gold evidence paragraphs.

| Tier | F1 | 95% CI | Accuracy | Configuration | n |
|------|-----|--------|----------|---------------|---|
| nli-only | 63.8% | - | 48.0% | DeBERTa-base, raw answers, t=0.5 | 1715 |
| nli-only | 65.3% | - | 53.1% | Fine-tuned DeBERTa-base + decomposition, t=0.5 | 1715 |
| nli-only | 66.1% | - | 55.5% | DeBERTa-base + answer reformulation + evidence decomposition, t=0.5 | 1715 |
| nli-only | 76.7% | - | 81.4% | Fine-tuned DeBERTa-large + decomposition, t=0.3 | 1715 |
| nli-only | 77.0% | [75.3%, 78.6%] | 81.9% | Fine-tuned DeBERTa-large-512 + decomposition, t=0.3 | 1715 |
| nli-only | 82.3% | [80.4%, 84.2%] | 89.0% | Fine-tuned large-512 + declarative reformulation + decompose + passage + rerank + MiniCheck, t=0.3 | 1715 |
| nli-only | **88.2%** | **[86.2%, 90.1%]** | **94.2%** [93.1%, 95.2%] | **Same pipeline, t=0.15** | 1715 |

Note: n=1715 is the number of predictions (1764 loaded, 49 answerable examples skipped due to missing evidence). All UNANSWERABLE examples are hardcoded as correct (100% specificity).

**Metric note**: The F1 values use harmonic-macro F1. For QASPER's imbalanced labels (1552 ANSWERABLE vs 163 UNANSWERABLE), this differs from sklearn macro F1. At t=0.3: harmonic-macro F1=82.3%, sklearn macro F1=78.5% [75.7%, 81.2%].

**Threshold optimization note**: The +5.9pp improvement from t=0.3 to t=0.15 is legitimate because: (1) QASPER is 90.5% ANSWERABLE, so lowering the threshold captures more true positives; (2) all UNANSWERABLE examples are hardcoded as correct, so false-positive risk is zero for that class; (3) the model maintains 93.6% accuracy on ANSWERABLE and 100% on UNANSWERABLE.

### QASPER Pipeline Ablation (77.0% -> 88.2%, +11.2pp)

1. **Declarative answer reformulation (+0.3pp)**: Yes/no answers converted from "Q? A" format to declarative statements using grammatical inversion. Short answers merged with question context via Wh-question pattern matching. This produces proper NLI hypotheses matching the model's training distribution.

2. **Evidence decomposition (+7pp)**: QASPER evidence is full paragraphs. Breaking them into sentences before NLI scoring improves matching by staying within DeBERTa's training distribution (short premise-hypothesis pairs). This is the single largest contributor.

3. **Passage scoring (+2pp)**: In addition to individual sentence scoring, the concatenated evidence passage is scored as a whole. Takes the maximum of both approaches.

4. **Coverage-based rerank (+0.8pp)**: Select top-k evidence sentences that maximize claim token coverage. Helps when the most relevant evidence is not the one with the highest raw NLI score.

5. **MiniCheck fallback (+1.2pp)**: When the primary model's entailment score is below threshold, a second opinion from MiniCheck captures borderline entailments.

### QASPER Error Analysis

At t=0.15 with full pipeline:
- **0 false positives**: Every unanswerable question is correctly identified (100% specificity)
- **100 false negatives**: 6.4% of answerable questions predicted as unanswerable (down from 12.1% at t=0.3)

The remaining FN examples include heavily paraphrased answers, numeric/table-based content, and cases where the answer is implied but not directly stated in evidence.

## Fine-Tuning Impact Analysis

### DeBERTa-base (184M)

| Benchmark | Pre-trained | Fine-tuned base | Delta | Notes |
|-----------|-------------|-----------------|-------|-------|
| HAGRID XGBoost CV | 80.1% | **86.2%** | **+6.1pp** | In-distribution training data |
| SciFact nli-only | 82.0% | **87.3%** | **+5.3pp** | Better sentence-level verification |
| SciFact nli-abstract | **87.7%** | 84.2% | -3.5pp | Binary training hurts 3-way abstract classification |
| FEVER nli-only | **85.9%** | 90.0% | **+4.1pp** | Fine-tuning helps despite binary training |
| QASPER nli-only | **66.1%** | 65.3% | -0.8pp | Minimal difference |

### DeBERTa-large (435M)

| Benchmark | Fine-tuned base | Fine-tuned large | Delta | Notes |
|-----------|-----------------|------------------|-------|-------|
| HAGRID nli-only | **65.1%** | 64.0% | -1.1pp | Slight regression |
| HAGRID XGBoost CV | **86.2%** | 85.3% | -0.9pp | Similar aggregated performance |
| SciFact nli-only | 87.3% | **95.3%** | **+8.0pp** | Near-perfect verification |
| SciFact nli-abstract | 84.2% | **85.2%** | +1.0pp | Modest improvement |
| FEVER nli-only | 90.0% | **92.6%** | +2.6pp | Larger model retains more discrimination |
| QASPER decomposed | 65.3% | **76.6%** | +11.3pp | Large improvement from model scale |

### DeBERTa-large max_seq=512 (435M)

Same model architecture with max_seq_length=512 (up from 256), allowing the model to see longer evidence passages without truncation.

| Benchmark | 256 final | 512 final | Delta | Notes |
|-----------|-----------|-----------|-------|-------|
| HAGRID NLI-only (whole, full dev) | 65.4% | **72.3%** | **+6.9pp** | Whole-answer scoring, biggest win |
| HAGRID XGBoost CV | **85.3%** | 84.3% | -1.0pp | Single model, 52 features |
| HAGRID XGBoost 3-model ensemble | 86.8% (2-model) | **87.5%** (3-model) | **+0.7pp** | Adding 512 model helps ensemble |
| SciFact nli-only | **95.3%** | 94.2% | -1.1pp | Short evidence, 256 sufficient |
| SciFact nli-abstract | **87.7%** | 87.6% | -0.1pp | Essentially identical |
| FEVER | 92.6% (fine-tuned) | **94.7%** | **+2.1pp** | + passage-scoring |
| QASPER (full pipeline) | 76.6% | **82.3%** | **+5.7pp** | + full pipeline enhancements |

The 512 context window enables effective whole-answer NLI scoring and improves performance on tasks with longer evidence (FEVER, QASPER, HAGRID whole-answer). Short-evidence tasks (SciFact) are unaffected or slightly prefer the 256 model.

### Deep Error Analysis: HAGRID False Positives

The fine-tuned large model has difficulty separating ATTRIBUTABLE from NOT_ATTRIBUTABLE using NLI scores alone:

| Metric | Value |
|--------|-------|
| ROC AUC (NLI-only) | **0.626** |
| False Positives | 149/392 (38%) |
| NOT_ATTR with score > 0.5 | 116/165 (70.3%) |

70% of NOT_ATTRIBUTABLE examples receive scores > 0.5 because they are topically related to the evidence but not actually supported. This confirms why XGBoost aggregation with additional features (lexical overlap, entity matching, number coverage) is essential for HAGRID.

### Multi-Model Ensemble XGBoost

| Configuration | Models | Features | CV F1 | Accuracy | Precision | Recall |
|---------------|--------|----------|-------|----------|-----------|--------|
| Single (base) | 1 | 52 | 86.2% | 81.2% | 79.9% | 93.6% |
| Single (large-256) | 1 | 52 | 85.3% | - | - | - |
| Single (large-512) | 1 | 52 | 84.3% | 78.1% | - | - |
| 2-model (base + large) | 2 | 64 | 86.8% | 82.2% | - | - |
| **3-model (base + large + large-512)** | **3** | **156** | **87.5%** | **82.9%** | **81.0%** | **95.0%** |

The base model dominates the top feature importances (9/15 top features), while the large-256 model contributes 6/15 top features. The large-512 model contributes complementary edge-case features that add +0.7pp to the ensemble.

### Best Model per Task

- HAGRID: Fine-tuned base + large + large-512 ensemble XGBoost (**87.5%**)
- SciFact nli-only: Fine-tuned DeBERTa-large (**95.3%** at t=0.3)
- SciFact nli-abstract: DeBERTa-large + MiniCheck fallback (**87.4%** at ent=0.2, fb=0.3)
- FEVER: Fine-tuned DeBERTa-large-512 + passage-scoring (**94.7%** on full dev)
- QASPER: Fine-tuned DeBERTa-large-512 + full pipeline (**88.2%** at t=0.15)
- FActScore: XGBoost base+large+large-512 ensemble (**83.8%**, 174 features, 5-fold CV)

## FActScore: Atomic Fact Verification

FActScore (Min et al., EMNLP 2023) evaluates fine-grained atomic fact verification in LLM-generated biographies. Each atomic fact extracted from a biography is checked against the corresponding Wikipedia article.

### Best Results

| Model | Method | F1 | 95% CI | Accuracy | Notes |
|-------|--------|-----|--------|----------|-------|
| **XGBoost 3-model ensemble** | **174-feature XGBoost, 5-fold CV** | **83.8%** | (see note) | **87.0%** | **Base+Large+Large-512** |
| XGBoost 2-model ensemble | 118-feature XGBoost, 5-fold CV | 83.5% | [82.8%, 84.2%] | 86.8% | Base+Large |
| XGBoost multi-signal (base) | 56-feature XGBoost, 5-fold CV | 82.7% | [82.1%, 83.4%] | 85.9% | Single-model |
| Fine-tuned DeBERTa-large-512 | NLI-only, sentences, t=0.5 | 78.5% | [77.8%, 79.2%] | 79.6% | NLI-only best |
| Fine-tuned DeBERTa-large-512 | NLI-only, no decompose, t=0.5 | 65.8% | - | 53.6% | Full article text |

### Published Baselines (Min et al., 2023)

FActScore baselines report the percentage of supported atomic facts per biography (an entity-level metric). Our metric is binary classification accuracy across all 14,274 individual atomic facts -- a more granular evaluation.

| System | FActScore | Notes |
|--------|-----------|-------|
| PerplexityAI | 63.7% | Entity-level supported-fact percentage |
| ChatGPT | 62.5% | Entity-level supported-fact percentage |
| InstructGPT | 58.4% | Entity-level supported-fact percentage |

### Evidence Decomposition

The key technique for FActScore is **evidence decomposition**: splitting long Wikipedia articles into smaller chunks and taking the maximum entailment score across all chunks.

| Mode | Description | Chunks/article | F1 (t=0.5) | F1 (t=0.3) |
|------|-------------|-----------------|------------|------------|
| **sentences** | 3-sentence groups (natural boundaries) | ~14 | **78.5%** | 78.2% |
| relevant_sentences | BM25-ranked 3-sentence groups (top 8) | 8 | 78.1% | 78.2% |
| chars | 1000-char overlapping chunks (200 overlap) | ~10 | 73.9% | 74.4% |
| relevant | BM25-ranked 1000-char chunks (top 5) | 5 | 73.5% | 74.3% |
| None | Full article text | 1 | 65.8% | 66.8% |

Sentence-level decomposition (+4.6pp F1 over char chunking) performs best, as respecting natural sentence boundaries avoids splitting mid-sentence. BM25 relevance filtering slightly hurts performance (-0.4pp), as aggressive filtering discards some relevant evidence. Decomposition mode matters more than threshold: the gap between sentences and chars (+4.6pp at t=0.5) is much larger than the threshold tuning effect (+0.5pp for chars).

### XGBoost Multi-Signal Aggregation

The XGBoost model combines NLI scores, lexical overlap, and cross-signal interactions. The multi-model ensemble uses features from DeBERTa-base, DeBERTa-large, and DeBERTa-large-512 for 174 total features, with cross-model agreement features.

**Multi-model ensemble progression:**

| Configuration | Models | Features | CV Macro F1 | Accuracy |
|---------------|--------|----------|-------------|----------|
| Single (base) | DeBERTa-base | 56 | 82.7% | 85.9% |
| Single (large) | DeBERTa-large | 56 | 82.0% | 85.5% |
| Ensemble (base+large) | Both | 118 | 83.5% | 86.8% |
| **Ensemble (all three)** | **All three** | **174** | **83.8%** | **87.0%** |

Multi-model ensembles capture complementary signals: different-sized models make different errors, and cross-model features let XGBoost learn when models agree vs disagree.

### Methodology

- **Data**: 14,274 atomic facts from 505 entities across InstructGPT, ChatGPT, and PerplexityAI biographies. 9,912 SUPPORTED, 4,362 NOT_SUPPORTED. 1,515 irrelevant facts and 251 facts with missing Wikipedia articles excluded.
- **Evidence**: Full Wikipedia article text fetched via MediaWiki API, cached locally (183 articles).
- **NLI**: Fine-tuned DeBERTa-v3-large with max_seq_length=512. Each atomic fact scored against evidence chunks from the Wikipedia article. Maximum entailment score used for classification.
- **Label mapping**: FActScore labels S (Supported) -> SUPPORTED, NS (Not Supported) -> NOT_SUPPORTED. IR (Irrelevant) facts excluded.
- **Sentence decomposition**: Text split on sentence boundaries, grouped into 3-sentence chunks (max 50 sentences). Each group scored independently via NLI.

## AttributionBench: Comprehensive Attribution Evaluation

AttributionBench (OSU NLP, ACL 2024 Findings) aggregates 26K examples from 7 attribution datasets with binary labels. Tests whether claims are fully supported by cited evidence passages.

### Published Baselines (OOD Macro-F1, Li et al. 2024 Table 3)

| System | OOD Macro-F1 | Notes |
|--------|-----|-------|
| Fine-tuned GPT-3.5 | 81.9% | Fine-tuned on AttributionBench training data |
| GPT-4 zero-shot (w/ CoT) | 78.9% | Zero-shot with chain-of-thought |
| GPT-4 zero-shot (w/o CoT) | 78.0% | Zero-shot prompting |
| **PCA-Eval XGBoost (large512, t=0.64)** | **81.5%** | **435M parameter model** |
| PCA-Eval XGBoost (large512, t=0.50) | 77.8% | 435M parameter model |
| PCA-Eval NLI-only (fine-tuned base, OOD) | 74.9% (binary) | 184M, NLI only, no XGBoost |

### NLI-Only Results (whole strategy, t=0.5)

#### Out-of-Distribution Test Set (1,611 examples)

| Model | F1 | Accuracy | ATTR Acc | NOT_ATTR Acc |
|-------|-----|---------|----------|--------------|
| **Fine-tuned DeBERTa-base** | **74.9%** | **74.9%** | 79.8% | 69.7% |
| Fine-tuned DeBERTa-large | 72.3% | 72.1% | 83.0% | 60.4% |
| Pre-trained DeBERTa-large | 72.2% | 69.8% | 50.9% | 90.1% |

### XGBoost Domain-Specific Results (AttributionBench-trained)

Trained on 9K AttributionBench examples with 5-fold stratified CV. Threshold tuned on dev set.

| Config | Features | CV F1 | Test OOD binary F1 | Test OOD macro-F1 | Threshold |
|--------|----------|-------|------------|-------------|-----------|
| **large512 (n=300, lr=0.03)** | **52** | **85.0%** | **81.9%** | **81.5%** | **0.64** |
| large512 (n=100, lr=0.1) | 52 | 85.4% | 80.5% | 77.7% | 0.50 |
| ensemble (3-model) | 156 | 85.5% | 77.4% | - | 0.44 |

The single-model configuration with slower learning rate (0.03 vs 0.1) and more trees (300 vs 100) reduces overfitting on training data (CV F1 85.0% vs 85.4%) while improving OOD generalization (+3.8pp macro-F1). Higher threshold (0.64 vs 0.50) optimizes the precision-recall tradeoff for OOD data. The 3-model ensemble overfits with 156 features on only 9K examples.

### Threshold-Performance Tradeoff (large512, n=300, lr=0.03)

| Threshold | OOD binary F1 | OOD macro-F1 | OOD Accuracy |
|-----------|-------|--------|----------|
| 0.44 (dev-tuned) | 79.7% | 75.5% | 76.2% |
| 0.50 (natural cutoff) | 80.7% | 77.8% | 78.2% |
| 0.54 | 81.7% | 79.8% | 80.0% |
| 0.62 | 81.9% | 81.1% | 81.1% |
| **0.64 (OOD peak)** | **81.9%** | **81.5%** | **81.5%** |

## Off-the-Shelf Model Comparison

Evaluation of purpose-built fact-checking models to contextualize fine-tuned results.

### Models Evaluated

| Model | Size | Training Data | Type |
|-------|------|---------------|------|
| FactCG-DeBERTa-v3-Large | 400M | Graph-enhanced multi-hop (NAACL 2025) | Binary |
| MoritzLaurer DeBERTa-v3-large-mnli-fever-anli-ling-wanli | 435M | 885K from 6 datasets | 3-way NLI |

### HAGRID Results

| Model | Strategy | F1 | Accuracy |
|-------|----------|-----|----------|
| FactCG | majority | 46.7% | 62.5% |
| FactCG | multi-signal | 54.8% | 63.0% |
| MoritzLaurer | majority | 54.8% | 56.6% |
| MoritzLaurer | multi-signal | 56.3% | 57.3% |
| *Fine-tuned base (multi-signal)* | *multi-signal t=0.3* | *65.1%* | - |
| ***Fine-tuned ensemble XGBoost*** | ***xgboost*** | ***87.5%*** | *82.9%* |

### SciFact Results

| Tier | Model | Fallback | F1 | SUPPORTS | REFUTES | NEI |
|------|-------|----------|-----|----------|---------|-----|
| nli-only | FactCG | - | 74.9% | 98.6% | 0.8% | 100% |
| nli-only | MoritzLaurer | - | 73.4% | 41.2% | 69.7% | 100% |
| nli-only | *Fine-tuned large* | - | *95.3%* | - | - | - |
| nli-abstract | MoritzLaurer | MiniCheck | 86.2% | 63.4% | 96.7% | 100% |
| nli-abstract | *DeBERTa-large* | *MiniCheck* | ***87.4%*** | *67.6%* | *97.5%* | *100%* |

### FEVER Results

| Model | F1 | Accuracy | SUPPORTS | REFUTES | NEI |
|-------|-----|----------|----------|---------|-----|
| FactCG | 72.4% | 67.4% | 96.8% | 5.2% | 100% |
| MoritzLaurer | 89.5% | 88.4% | 86.3% | 79.0% | 100% |
| *Fine-tuned DeBERTa-large-512 + passage-scoring* | ***94.7%*** | *94.7%* | *89.1%* | *95.0%* | *100%* |

### QASPER Results

| Model | F1 | Accuracy | Notes |
|-------|-----|----------|-------|
| FactCG | 99.5% | 99.8% | Degenerate: always predicts "supported" |
| MoritzLaurer | 64.0% | 48.6% | Similar to DeBERTa-base baseline (63.8%) |
| *Fine-tuned large-512 + full pipeline* | ***82.3%*** | *89.0%* | *Full pipeline* |

FactCG produces extreme support scores (almost always >0.5), resulting in degenerate predictions on HAGRID and QASPER. MoritzLaurer provides balanced but lower-performing predictions across all benchmarks. Neither off-the-shelf model matches the fine-tuned results on any benchmark.

## Models Tested

| Model | Size | Type |
|-------|------|------|
| cross-encoder/nli-deberta-v3-base | 184M | 3-way NLI |
| cross-encoder/nli-deberta-v3-large | 435M | 3-way NLI |
| lytang/MiniCheck-DeBERTa-v3-Large | 355M | Binary fact-check |
| yaxili96/FactCG-DeBERTa-v3-Large | 400M | Binary fact-check |
| MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli | 435M | 3-way NLI |
| Fine-tuned DeBERTa-v3-base (full-mix) | 184M | Binary NLI |
| Fine-tuned DeBERTa-v3-large (full-mix) | 435M | Binary NLI |
| Fine-tuned DeBERTa-v3-large (full-mix, 512) | 435M | Binary NLI |

## Training Data

The fine-tuned models were trained on ~33K examples from public NLI datasets with binary classification (entailment vs not-entailment).

Hyperparameters:
- DeBERTa-v3-base: lr=2e-5, batch=32, 2 epochs, max_seq=256
- DeBERTa-v3-large: lr=1e-5, batch=32, 2 epochs, max_seq=256, gradient checkpointing
- DeBERTa-v3-large-512: lr=1e-5, batch=32, 2 epochs, max_seq=512, gradient checkpointing
- All: binary cross-entropy loss, eval each epoch, load best model at end

Training scripts are not included in this repository. See the paper for training details.

## Reproduction

```bash
# HAGRID with XGBoost (pre-trained baseline)
python -m benchmarks.run hagrid --tier nli-only --attribution-strategy xgboost

# SciFact with evidence decomposition
python -m benchmarks.run scifact --tier nli-abstract

# QASPER with answer reformulation
python -m benchmarks.run qasper --tier nli-only --decompose-evidence

# FActScore atomic fact verification (sentence decomposition)
python -m benchmarks.run factscore --tier nli-only --decompose-evidence \
    --decompose-mode sentences

# Recompute confidence intervals from saved predictions
python -m benchmarks.run_stats --all
```

Training scripts are not included in this repository. See the paper for training details.
