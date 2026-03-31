# PCA-Eval

Open-source benchmark evaluation suite for Proof-Carrying Answers. Runs NLI-based verification across six academic benchmarks (SciFact, FEVER, QASPER, HAGRID, FActScore, AttributionBench) to measure how well the verification pipeline detects whether AI claims are supported by source evidence. Public repo at github.com/conjfrnk/pca-eval. Produces the results reported in the PCA paper.

## Status

Active -- public repo, used for ongoing benchmark evaluation

## Tech Stack

- Python 3.12+
- PyTorch + HuggingFace Transformers (DeBERTa NLI models)
- scikit-learn (metrics, XGBoost aggregation)
- pytest + ruff (dev tooling)

## Key Files

- `benchmarks/run.py` -- CLI entry point for all benchmarks
- `benchmarks/nli.py` -- NLI model wrapper
- `benchmarks/base.py` -- Base classes and data types
- `results/` -- Saved result JSONs with per-example predictions
- `README.md` -- Full usage guide, benchmark table, and project structure
