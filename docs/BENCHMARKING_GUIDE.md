# Benchmarking Guide

Academic benchmarks for the NLI verification pipeline.

## Benchmarks

| # | Benchmark | What it tests | Dataset size | Why it matters |
|---|-----------|--------------|-------------|----------------|
| 1 | **SciFact** | Claim verification against scientific abstracts | 1,409 claims / 5,183 abstracts | Directly tests the NLI pipeline on the exact task PCA-Eval performs |
| 2 | **FEVER** | Fact verification against Wikipedia | 185K claims (we sample) | Tests NLI at scale |
| 3 | **QASPER** | QA on scientific papers with evidence | ~5K questions / 1,585 papers | Tests question -> answer -> evidence loop |
| 4 | **HAGRID** | Hallucination detection in RAG | ~4K examples | Directly measures attribution quality |
| 5 | **FActScore** | Atomic fact verification in LLM biographies | ~4.6K atomic facts across 3 models | Tests NLI generalization to fine-grained fact checking |
| 6 | **AttributionBench** | Attribution evaluation for claims against cited evidence | ~3K examples (ID + OOD splits) | Comprehensive attribution benchmark with in-distribution and out-of-distribution evaluation |
| -- | **AIS** | Attribution metric (not a dataset) | Applied to other benchmarks | Formal metric for attribution-based verification |

## Quick Start

All commands run from the `pca-eval/` package root.

> **Note:** The results reported in the paper use fine-tuned DeBERTa models and XGBoost ensemble classifiers that are not included in this repository. Out-of-the-box runs use the pre-trained `cross-encoder/nli-deberta-v3-base` model and will produce lower numbers. See the "Comparing to Published Baselines" section for context on what to expect with pre-trained vs. fine-tuned models.

```bash
cd pca-eval

# 1. Ensure Python 3.12+ is on the PATH

# 2. Download all datasets (~200MB total, no FEVER wiki)
python -m benchmarks.download

# 3. Validate everything loaded correctly
python -m benchmarks.run all --tier dry-run

# 4. Run SciFact NLI. $0. ~2 min.
python -m benchmarks.run scifact --tier nli-only

# 5. Find the optimal threshold. $0. ~5 min (first run), instant after.
python -m benchmarks.run scifact --tier sweep
```

## Execution Tiers

The framework has five tiers designed so that all local inference runs complete before any API-based evaluation.

| Tier | Command | Cost | Time | What it does |
|------|---------|------|------|-------------|
| `dry-run` | `--tier dry-run` | $0 | seconds | Load data, validate format, print stats |
| `nli-only` | `--tier nli-only` | $0 | minutes | Run DeBERTa cross-encoder on gold evidence |
| `nli-abstract` | `--tier nli-abstract` | $0 | minutes | NLI on full abstracts (SciFact: tests retrieval implicitly) |
| `sweep` | `--tier sweep` | $0 | minutes | Sweep thresholds 0.3-0.8, find optimal F1 |
| `ais` | `--tier ais` | $0 | minutes | AIS attribution score on any benchmark |

All tiers run the same `cross-encoder/nli-deberta-v3-base` model that PCA-Eval uses in the verification pipeline. No API calls. No OpenAI charges. Pure local CPU inference.

## Recommended Workflow

### Phase 1: Validate (10 minutes, $0)

```bash
# Download everything
python -m benchmarks.download

# Dry-run all benchmarks to confirm data loaded
python -m benchmarks.run all --tier dry-run
```

The output should show example counts and label distributions for each benchmark. If any fail to download, the error message indicates what to fix.

### Phase 2: SciFact (30 minutes, $0)

SciFact is the most directly relevant benchmark, as it tests claim verification against scientific abstracts.

```bash
# Full NLI run on all SciFact dev claims (300 claims)
python -m benchmarks.run scifact --tier nli-only

# Threshold sweep to find optimal operating point
python -m benchmarks.run scifact --tier sweep

# NLI on full abstracts (not just gold evidence sentences)
# This is harder and more realistic
python -m benchmarks.run scifact --tier nli-abstract
```

The **sweep** is critical: it runs NLI at thresholds from 0.3 to 0.8 and shows accuracy/F1 at each point. The first run does actual inference; subsequent runs are instant because all NLI scores are cached in `benchmarks/cache/responses.db`.

Reference targets: **F1 > 65%** on SciFact with oracle evidence is competitive with published baselines. **F1 > 70%** is strong. With pre-trained models, F1 > 70% is a good result. The paper reports 95.3% using fine-tuned DeBERTa-large models not included in this repository.

### Phase 3: All Benchmarks NLI-Only (1-2 hours, $0)

```bash
# FEVER: sample 5K claims (full dev set is ~19K)
python -m benchmarks.run fever --tier nli-only --sample 5000

# QASPER: check answer-evidence entailment
python -m benchmarks.run qasper --tier nli-only

# HAGRID: attribution detection
python -m benchmarks.run hagrid --tier nli-only

# FActScore: atomic fact verification against Wikipedia
python -m benchmarks.run factscore --tier nli-only

# AttributionBench: attribution evaluation (ID and OOD splits)
python -m benchmarks.run attribution_bench --tier nli-only

# AIS metric on QASPER and HAGRID (they have answer text)
python -m benchmarks.run qasper --tier ais
python -m benchmarks.run hagrid --tier ais
```

### Phase 4: Full Pipeline Benchmarks (optional, for end-to-end evaluation)

For the full-pipeline benchmarks (ingesting documents, running the full verification pipeline with an LLM), the backend must be running. This costs API money. Only proceed after NLI-only numbers are satisfactory.

This tier is not yet wired up in the runner. When ready, the architecture is:
1. Ingest benchmark documents via the API
2. Run queries through the full verification pipeline
3. Compare pipeline outputs against gold labels

The NLI-only results directly test the core verification model. Full-pipeline evaluation adds end-to-end coverage.

## Understanding Results

### SciFact Results

```
  SciFact | tier=nli-only | n=300
  Accuracy:    72.3%        <- Overall label accuracy
  Precision:   68.5%        <- When the model says SUPPORTS, how often is it correct?
  Recall:      74.2%        <- Of actual SUPPORTS, how many were found?
  F1:          71.2%        <- Harmonic mean (the headline number)
  Per-label accuracy:
    NOT_ENOUGH_INFO:  65.0%
    REFUTES:          58.3%
    SUPPORTS:         82.1%
  Evidence F1: 67.8%        <- Did the model identify the right rationale sentences?
```

### FEVER Results

FEVER is larger scale. Without Wikipedia text downloaded, only NEI claims can be auto-scored. Download wiki pages (`python -m benchmarks.download --include-wiki`, ~4GB) for full scoring.

### QASPER Results

QASPER NLI-only tests whether gold answers are entailed by gold evidence. High scores indicate that the NLI model handles scientific text well.

### HAGRID Results

HAGRID tests attribution detection -- exactly what PCA-Eval does. The attribution_rate per answer indicates what fraction of answer sentences are NLI-verifiable against source passages.

### AIS Metric

AIS is a meta-metric: "what fraction of generated statements are attributable to sources?" It is applied to QASPER and HAGRID examples that have both answers and evidence. This metric directly formalizes the notion of source-grounded verification.

## Caching and Cost Control

All NLI predictions are cached in SQLite at `benchmarks/cache/responses.db`.

- **First run:** Loads model (~10s), runs inference (~100ms/claim)
- **Subsequent runs:** Instant cache hits, no model needed

This means:
- Threshold sweeps are effectively free after the first pass
- Re-runs with different aggregation logic incur zero cost
- The cache persists across sessions

To clear the cache:
```bash
rm benchmarks/cache/responses.db
```

## File Structure

```
benchmarks/
    __init__.py         Package init
    run.py              CLI entry point (python -m benchmarks.run)
    run_stats.py        Statistical analysis CLI (bootstrap CIs, McNemar's test)
    download.py         Dataset downloader (python -m benchmarks.download)
    base.py             Base suite class, metrics, reporting
    cache.py            SQLite response cache
    nli.py              Standalone DeBERTa NLI wrapper
    stats.py            Statistical utilities (bootstrap CIs, paired comparisons)
    scifact.py          SciFact benchmark suite
    fever.py            FEVER benchmark suite
    qasper.py           QASPER benchmark suite
    hagrid.py           HAGRID benchmark suite
    factscore.py        FActScore benchmark suite
    attribution_bench.py  AttributionBench benchmark suite
    ais.py              AIS attribution metric
    data/               Downloaded datasets (.gitignored)
    cache/              NLI score cache (.gitignored)
    results/            JSON result files (committed)
```

## Comparing to Published Baselines

### SciFact

| System | Label F1 | Evidence F1 |
|--------|---------|-------------|
| VeriSci (Wadden 2020) | 49.2 | -- |
| ParagraphJoint | 62.2 | -- |
| MultiVerS (Wadden 2022) | 67.2 | -- |
| DeBERTa cross-encoder, pre-trained (oracle evidence) | ~70-75 | -- |
| DeBERTa cross-encoder, fine-tuned (oracle evidence) | 95.3 | -- |
| **PCA-Eval (nli-only)** | **run it** | **run it** |
| **PCA-Eval (nli-abstract)** | **run it** | **run it** |

The pre-trained row reflects what you will see running the out-of-the-box model. The fine-tuned row (95.3%) uses DeBERTa-large fine-tuned on ANLI+WANLI+HAGRID+SciFact data, which is not included in this repository.

### FEVER

| System | Label Accuracy |
|--------|---------------|
| Majority class | 52.1% |
| NSMN | 68.2% |
| KGAT | 70.4% |
| DeBERTa NLI (oracle evidence) | ~85-90% |
| **PCA-Eval (nli-only)** | **run it** |

### FActScore

| System | FActScore (% supported) |
|--------|------------------------|
| InstructGPT (Min et al. 2023) | 58.4% |
| ChatGPT (Min et al. 2023) | 62.5% |
| PerplexityAI (Min et al. 2023) | 63.7% |
| XGBoost ensemble (fine-tuned, 174-feature) | 83.8% |
| **PCA-Eval (nli-only)** | **run it** |

The published baselines above are human-evaluated FActScores on model-generated biographies. Our system evaluates the same atomic facts using NLI verification against Wikipedia evidence.

### AttributionBench

| System | OOD Macro-F1 |
|--------|-------------|
| GPT-4 zero-shot w/o CoT (Li et al. 2024) | 78.0% |
| GPT-4 zero-shot w/ CoT (Li et al. 2024) | 78.9% |
| Fine-tuned GPT-3.5 (Li et al. 2024) | 81.9% |
| XGBoost ensemble (fine-tuned, domain features) | 81.5% |
| **PCA-Eval (nli-only)** | **run it** |

The OOD (out-of-distribution) test set is the primary comparison target, as it tests generalization to unseen domains including HAGRID examples.

## Troubleshooting

### "Dataset not found"
```bash
python -m benchmarks.download scifact   # Download specific dataset
python -m benchmarks.download           # Download all
```

### "NLI model loading slow"
First load takes ~10s to download and cache the model (~440MB). Subsequent loads are instant. The model is cached by `sentence-transformers` in `~/.cache/torch/sentence_transformers/`.

### "FEVER has 0 claims with evidence text"
FEVER evidence requires Wikipedia text. Either:
- Download wiki pages: `python -m benchmarks.download --include-wiki` (4GB)
- Or just use `--sample` to run on what is available

### "Import error"
Make sure `pca-eval` is on the Python path:
```bash
cd pca-eval
python -m benchmarks.run scifact --tier dry-run
```

### Changing the NLI model
The default is `cross-encoder/nli-deberta-v3-base` (the same model used in the verification pipeline). To test with a smaller/faster model, edit the `model_name` parameter in `nli.py`. Options:
- `cross-encoder/nli-deberta-v3-base` (recommended, ~440MB)
- `cross-encoder/nli-deberta-v3-small` (faster, ~180MB)
- `cross-encoder/nli-roberta-base` (alternative architecture)
