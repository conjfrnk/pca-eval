# PCA-Eval

Benchmark evaluation scripts and reproducibility artifacts for **Proof-Carrying Answers (PCAs)**.

This repository contains the evaluation pipeline used to produce the benchmark results reported in:

> **Proof-Carrying Answers: Machine-Verifiable AI Outputs with Typed Claims and Formal Evidence Chains**
> Connor Frank. arXiv preprint, 2026.

## Benchmarks

| Benchmark | Task | Metric | Result |
|-----------|------|--------|--------|
| SciFact | Claim verification against scientific abstracts | Oracle F1 | 95.3% |
| FEVER | Fact verification against Wikipedia | F1 | 94.7% |
| QASPER | Question answering with evidence on scientific papers | F1 | 88.2% |
| HAGRID | Hallucination detection in RAG systems | Binary F1 | 87.5% |
| FActScore | Atomic fact verification in biographical text | Binary F1 | 83.8% |
| AttributionBench | Cross-domain attribution verification | Macro F1 | 81.5% |
| FACTS Grounding | Document-grounded response verification | Accuracy | 98.1% |

> **Reproducibility note:** The results above were obtained using fine-tuned DeBERTa models not included in this repository (see paper for training details). Running benchmarks with the default pre-trained model (`cross-encoder/nli-deberta-v3-base`) will produce lower scores. Pre-trained baselines are documented in the `results/` directory.

> **Note:** FACTS Grounding results were obtained using a separate evaluation script not included in this release.

## Installation

```bash
git clone https://github.com/conjfrnk/pca-eval.git
cd pca-eval
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

The evaluation pipeline uses a tiered execution model for cost-efficient iteration:

```bash
# Tier 0: Load data, validate format, print stats ($0, seconds)
python -m benchmarks.run scifact --tier dry-run

# Tier 1: Run local DeBERTa NLI model ($0, minutes)
python -m benchmarks.run scifact --tier nli-only

# Run with a specific threshold
python -m benchmarks.run scifact --tier nli-only --threshold 0.3

# Run all benchmarks
python -m benchmarks.run all --tier nli-only
```

## Running Individual Benchmarks

The reported results for SciFact, FEVER, and QASPER require fine-tuned DeBERTa checkpoints (not included). Omit `--model-path` to run with the default pre-trained model.

### SciFact (Oracle)
```bash
python -m benchmarks.run scifact --tier nli-only --threshold 0.3 \
    --model-path path/to/model
```

### FEVER
```bash
python -m benchmarks.run fever --tier nli-only --threshold 0.3 \
    --model-path path/to/model --passage-scoring
```

### QASPER
```bash
python -m benchmarks.run qasper --tier nli-only --threshold 0.15 \
    --model-path path/to/model --passage-scoring
```

### HAGRID (XGBoost ensemble)
```bash
python -m benchmarks.run hagrid --tier nli-only --threshold 0.3 \
    --attribution-strategy xgboost
```

### FActScore
```bash
python -m benchmarks.run factscore --tier nli-only --threshold 0.3
```

### AttributionBench
```bash
python -m benchmarks.run attribution_bench --tier nli-only --threshold 0.3 \
    --attribution-strategy xgboost
```

## Threshold Sweep

Find the optimal operating point for any benchmark:

```bash
python -m benchmarks.run scifact --tier sweep
```

## Models

The evaluation supports multiple NLI models:

| Model | Parameters | Notes |
|-------|-----------|-------|
| `cross-encoder/nli-deberta-v3-base` | 184M | Default, fast (~100ms/pair) |
| `cross-encoder/nli-deberta-v3-large` | 435M | Higher accuracy (~300ms/pair) |
| `lytang/MiniCheck-DeBERTa-v3-Large` | 355M | Binary fact-checking specialist |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | 435M | High-accuracy NLI |

Fine-tuned model checkpoints used in the paper are not included in this repository.
See the paper for training details (dataset composition, hyperparameters, and training procedure).

## Project Structure

```
pca-eval/
├── benchmarks/          # Evaluation pipeline
│   ├── run.py           # CLI entry point
│   ├── base.py          # Base classes and data types
│   ├── nli.py           # NLI model wrapper
│   ├── ais.py           # AIS metric implementation
│   ├── scifact.py       # SciFact benchmark
│   ├── fever.py         # FEVER benchmark
│   ├── qasper.py        # QASPER benchmark
│   ├── hagrid.py        # HAGRID benchmark
│   ├── factscore.py     # FActScore benchmark
│   ├── attribution_bench.py  # AttributionBench benchmark
│   ├── download.py      # Dataset downloader
│   ├── cache.py         # Response cache (SQLite)
│   ├── stats.py         # Statistical utilities
│   └── run_stats.py     # Bootstrap CI and McNemar's test
├── results/             # Benchmark result logs
├── tests/               # Evaluation test suite
├── docs/                # Benchmarking guide and detailed results
└── pyproject.toml       # Project configuration
```

## Tests

```bash
pytest tests/ -x -q
```

## Citation

```bibtex
@article{frank2026pca,
  title={Proof-Carrying Answers: Machine-Verifiable AI Outputs with Typed Claims and Formal Evidence Chains},
  author={Frank, Connor},
  journal={arXiv preprint},
  year={2026}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
