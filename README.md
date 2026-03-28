# PCA-Eval

Benchmark evaluation scripts and reproducibility artifacts for **Proof-Carrying Answers (PCAs)** — machine-verifiable AI outputs with typed claims and formal evidence chains.

This repository contains the evaluation pipeline used to produce the benchmark results reported in:

> **[Proof-Carrying Answers: Machine-Verifiable AI Outputs with Typed Claims and Structured Evidence Chains](https://detent.ai/papers/proof-carrying-answers-mar2026.pdf)**
> Connor Frank. Detent.ai, 2026.

See also: [AI That Shows Its Work](https://detent.ai/blog/proof-carrying-answers) — a blog post introducing the architecture.

## What is a PCA?

A Proof-Carrying Answer (PCA) is an AI output where every claim is backed by machine-verifiable evidence. Instead of trusting the model, you verify the proof. This evaluation suite measures how well NLI-based verification detects whether AI-generated claims are actually supported by source evidence.

## Requirements

- **Python 3.12+**
- ~3 GB disk space (PyTorch + NLI model + benchmark datasets)
- CPU only — no GPU required

## Installation

```bash
git clone https://github.com/conjfrnk/pca-eval.git
cd pca-eval
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

For development (linting + tests):

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Step 1: Download benchmark datasets (~100 MB)
python -m benchmarks.download

# Step 2: Validate data loads correctly (instant, no model needed)
python -m benchmarks.run scifact --tier dry-run

# Step 3: Run NLI verification (~5 min, downloads model on first run)
python -m benchmarks.run scifact --tier nli-only

# Run all benchmarks
python -m benchmarks.run all --tier nli-only
```

> **First run note:** The NLI model (~400 MB) is downloaded automatically from HuggingFace Hub on first use. Subsequent runs use the cached model.

## Benchmarks

| Benchmark | Task | Metric | Fine-tuned | Pre-trained |
|-----------|------|--------|-----------|-------------|
| SciFact | Claim verification vs. scientific abstracts | Oracle F1 | 95.3% | ~70% |
| FEVER | Fact verification vs. Wikipedia | F1 | 94.7% | ~65% |
| QASPER | QA with evidence on scientific papers | F1 | 88.2% | ~55% |
| HAGRID | Hallucination detection in RAG systems | Binary F1 | 87.5% | ~55% |
| FActScore | Atomic fact verification in biographies | Binary F1 | 83.8% | ~60% |
| AttributionBench | Cross-domain attribution verification | Macro F1 | 81.5% | ~55% |

> **Reproducibility note:** The "Fine-tuned" column shows results from DeBERTa models fine-tuned on NLI data (not included; see [paper](https://detent.ai/papers/proof-carrying-answers-mar2026.pdf) for training details). The "Pre-trained" column shows approximate results with the default `cross-encoder/nli-deberta-v3-base` model. Pre-trained baselines are documented in `results/`.

## Execution Tiers

The pipeline uses a tiered model for cost-efficient iteration:

| Tier | What it does | Cost | Time |
|------|-------------|------|------|
| `dry-run` | Load data, validate format, print stats | $0 | Seconds |
| `nli-only` | Run local DeBERTa NLI model on gold evidence | $0 | Minutes |
| `nli-abstract` | Run NLI against full abstracts (SciFact) | $0 | Minutes |
| `sweep` | Threshold sweep to find optimal F1 | $0 | Minutes |
| `calibrate` | Temperature + threshold calibration | $0 | Minutes |
| `ais` | AIS attribution metric | $0 | Minutes |

## Running Individual Benchmarks

```bash
# SciFact with a specific threshold
python -m benchmarks.run scifact --tier nli-only --threshold 0.3

# FEVER on a sample of 5000 examples
python -m benchmarks.run fever --tier nli-only --sample 5000

# HAGRID with a different attribution strategy
python -m benchmarks.run hagrid --tier nli-only --attribution-strategy whole

# Threshold sweep to find optimal operating point
python -m benchmarks.run scifact --tier sweep

# Use a different model
python -m benchmarks.run scifact --tier nli-only --model cross-encoder/nli-deberta-v3-large
```

## Models

| Model | Parameters | Notes |
|-------|-----------|-------|
| `cross-encoder/nli-deberta-v3-base` | 184M | Default, fast (~100ms/pair) |
| `cross-encoder/nli-deberta-v3-large` | 435M | Higher accuracy (~300ms/pair) |
| `lytang/MiniCheck-DeBERTa-v3-Large` | 355M | Binary fact-checking specialist |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | 435M | High-accuracy NLI |

Fine-tuned model checkpoints are not included. See the [paper](https://detent.ai/papers/proof-carrying-answers-mar2026.pdf) for training details.

## Project Structure

```
pca-eval/
├── benchmarks/          # Evaluation pipeline
│   ├── run.py           # CLI entry point
│   ├── base.py          # Base classes and data types
│   ├── nli.py           # NLI model wrapper
│   ├── ais.py           # AIS metric implementation
│   ├── cache.py         # Response cache (SQLite)
│   ├── stats.py         # Statistical utilities
│   ├── run_stats.py     # Bootstrap CI and McNemar's test
│   ├── download.py      # Dataset downloader
│   ├── scifact.py       # SciFact benchmark
│   ├── fever.py         # FEVER benchmark
│   ├── qasper.py        # QASPER benchmark
│   ├── hagrid.py        # HAGRID benchmark
│   ├── factscore.py     # FActScore benchmark
│   └── attribution_bench.py  # AttributionBench benchmark
├── results/             # Pre-trained baseline result logs
├── tests/               # Test suite
├── docs/                # Benchmarking guide and detailed results
├── NOTICE.md            # Dataset licenses and attributions
└── pyproject.toml       # Project configuration
```

## Tests

```bash
pytest tests/ -x -q
```

## Statistical Analysis

Compute bootstrap confidence intervals on saved results:

```bash
# Analyze a specific result file
python -m benchmarks.run_stats results/scifact_nli-only_*.json

# Analyze all result files
python -m benchmarks.run_stats --all

# Compare two models (McNemar's test)
python -m benchmarks.run_stats --compare results/file_a.json results/file_b.json
```

## Citation

```bibtex
@article{frank2026pca,
  title={Proof-Carrying Answers: Machine-Verifiable AI Outputs with Typed Claims and Structured Evidence Chains},
  author={Frank, Connor},
  year={2026},
  url={https://detent.ai/papers/proof-carrying-answers-mar2026.pdf}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).

Dataset licenses and third-party attributions are listed in [NOTICE.md](NOTICE.md).
