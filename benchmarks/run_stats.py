#!/usr/bin/env python3
"""
Statistical analysis CLI for benchmark results.

Computes bootstrap 95% confidence intervals and McNemar's paired tests
on saved benchmark result files.

Usage:
    cd pca-eval

    # Single result file
    python -m benchmarks.run_stats benchmarks/results/scifact_nli-only_20260217_011036.json

    # Compare two models (McNemar's test)
    python -m benchmarks.run_stats --compare file_a.json file_b.json

    # All canonical results
    python -m benchmarks.run_stats --all

    # Custom bootstrap count
    python -m benchmarks.run_stats --n-bootstrap 50000 file.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from .base import RESULTS_DIR
from .stats import (
    compute_all_cis,
    load_predictions_from_result,
    mcnemar_test,
)

logger = logging.getLogger(__name__)

# Canonical best result files (double-verified 2026-02-17)
CANONICAL_RESULTS = {
    "scifact_nli-only": "scifact_nli-only_20260217_011036.json",
    "scifact_nli-abstract": "scifact_nli-abstract_20260216_235427.json",
    "fever_nli-only": "fever_nli-only_20260217_011244.json",
    "qasper_nli-only": "qasper_nli-only_20260217_011254.json",
    "factscore_nli-only": "factscore_nli-only_20260217_225838.json",
}


def analyze_result_file(
    path: Path,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Analyze a single result file and return CI data."""
    y_true, y_pred, metadata = load_predictions_from_result(path)

    print(f"\n{'=' * 60}")
    print(f"  {metadata['benchmark_name']} | {metadata['tier']} | n={metadata['num_examples']}")
    print(f"  Source: {path.name}")
    print(f"  Reported F1: {metadata['f1']:.4f}")
    print(f"{'=' * 60}")

    cis = compute_all_cis(y_true, y_pred, n_bootstrap=n_bootstrap, seed=seed)

    for _name, result in cis.items():
        print(f"  {result}")

    return {
        "metadata": metadata,
        "confidence_intervals": {name: r.to_dict() for name, r in cis.items()},
    }


def compare_results(
    path_a: Path,
    path_b: Path,
) -> dict:
    """Compare two result files using McNemar's test."""
    y_true_a, y_pred_a, meta_a = load_predictions_from_result(path_a)
    y_true_b, y_pred_b, meta_b = load_predictions_from_result(path_b)

    if y_true_a != y_true_b:
        print("WARNING: Ground truth labels differ between files. Comparison may not be valid.")

    print(f"\n{'=' * 60}")
    print("  McNemar's Test: Paired Model Comparison")
    print(f"  A: {meta_a['benchmark_name']} {meta_a['tier']} (F1={meta_a['f1']:.4f})")
    print(f"  B: {meta_b['benchmark_name']} {meta_b['tier']} (F1={meta_b['f1']:.4f})")
    print(f"{'=' * 60}")

    result = mcnemar_test(y_true_a, y_pred_a, y_pred_b)
    print(f"  {result}")

    return {
        "model_a": meta_a,
        "model_b": meta_b,
        "mcnemar": result.to_dict(),
    }


def run_all_canonical(n_bootstrap: int = 10000, seed: int = 42) -> dict:
    """Run CI analysis on all canonical benchmark results."""
    all_results = {}

    for key, filename in CANONICAL_RESULTS.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"\n  SKIP: {key} -- file not found: {filename}")
            continue
        result = analyze_result_file(path, n_bootstrap=n_bootstrap, seed=seed)
        all_results[key] = result

    # Save combined results
    output_path = RESULTS_DIR / "confidence_intervals.json"
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Combined results saved to: {output_path}")

    return all_results


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Compute bootstrap confidence intervals for benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Result JSON file(s) to analyze",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all canonical benchmark results",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare two result files using McNemar's test (requires exactly 2 files)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap resamples (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: prints to stdout)",
    )

    args = parser.parse_args()

    if args.all:
        results = run_all_canonical(n_bootstrap=args.n_bootstrap, seed=args.seed)
    elif args.compare:
        if len(args.files) != 2:
            print("ERROR: --compare requires exactly 2 result files")
            return 1
        results = compare_results(Path(args.files[0]), Path(args.files[1]))
    elif args.files:
        results = {}
        for f in args.files:
            path = Path(f)
            if not path.exists():
                print(f"ERROR: File not found: {f}")
                return 1
            result = analyze_result_file(path, n_bootstrap=args.n_bootstrap, seed=args.seed)
            results[path.stem] = result
    else:
        parser.print_help()
        return 1

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\n  Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
