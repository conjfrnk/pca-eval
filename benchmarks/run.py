#!/usr/bin/env python3
"""
Benchmark runner for PCA evaluation pipeline.

Execution tiers (for cost-efficient iteration):
    dry-run      - Load data, validate format, print stats.          $0, seconds.
    nli-only     - Run local DeBERTa NLI model only.                 $0, minutes.
    nli-abstract - Run NLI against full abstracts (not just gold).   $0, minutes.
    sweep        - Threshold sweep across operating points.          $0, minutes.
    calibrate    - Find optimal temperature and thresholds.          $0, minutes.
    ais          - AIS attribution metric.                           $0, minutes.

Usage:
    cd pca-eval
    python -m benchmarks.run scifact --tier dry-run
    python -m benchmarks.run scifact --tier nli-only
    python -m benchmarks.run scifact --tier nli-only --threshold 0.6
    python -m benchmarks.run scifact --tier nli-only --sample 100
    python -m benchmarks.run all --tier nli-only
    python -m benchmarks.run all --tier dry-run
"""

import argparse
import logging
import sys
import time

logger = logging.getLogger(__name__)

BENCHMARK_SUITES = {}


def _get_suites():
    """Lazy import to avoid loading all modules at startup."""
    if BENCHMARK_SUITES:
        return BENCHMARK_SUITES

    from .attribution_bench import AttributionBench
    from .factscore import FActScore
    from .fever import FEVER
    from .hagrid import HAGRID
    from .qasper import QASPER
    from .scifact import SciFact

    BENCHMARK_SUITES["scifact"] = SciFact()
    BENCHMARK_SUITES["fever"] = FEVER()
    BENCHMARK_SUITES["qasper"] = QASPER()
    BENCHMARK_SUITES["hagrid"] = HAGRID()
    BENCHMARK_SUITES["attribution_bench"] = AttributionBench()
    BENCHMARK_SUITES["factscore"] = FActScore()
    return BENCHMARK_SUITES


def _make_nli_instance(
    cache,
    model_name: str | None = None,
    model_path: str | None = None,
    decompose_evidence: bool = False,
):
    """Create an NLI instance with the given cache and optional model."""
    from .nli import DEFAULT_MODEL
    from .nli import NLIEvaluator as _Cls
    name = model_name or DEFAULT_MODEL
    return _Cls(
        model_name=name,
        cache=cache,
        model_path=model_path,
        decompose_evidence=decompose_evidence,
    )


def _make_ais_instance(nli, threshold):
    """Create an AIS scorer instance."""
    from .ais import AISScorer as _Cls
    return _Cls(nli=nli, entailment_threshold=threshold)


def run_dry_run(suite_name: str) -> None:
    """Tier 0: Load data, validate, print statistics. $0, seconds."""
    suites = _get_suites()
    suite = suites[suite_name]

    print(f"\n{'=' * 60}")
    print(f"  DRY RUN: {suite.name}")
    print(f"  {suite.description}")
    print(f"{'=' * 60}")

    if not suite.is_downloaded():
        print(f"  Dataset not downloaded. Run: python -m benchmarks.download {suite_name}")
        print("  Downloading now...")
        suite.download()

    if not suite.is_downloaded():
        print(f"  FAILED: Could not download {suite_name}")
        return

    try:
        examples = suite.load()
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return

    print(f"  Total examples: {len(examples)}")

    # Label distribution
    label_dist: dict[str, int] = {}
    for ex in examples:
        label_dist[ex.gold_label] = label_dist.get(ex.gold_label, 0) + 1
    print(f"  Labels: {dict(sorted(label_dist.items()))}")

    # Evidence stats
    with_evidence = sum(1 for ex in examples if ex.evidence_sentences)
    avg_evidence = (
        sum(len(ex.evidence_sentences) for ex in examples if ex.evidence_sentences)
        / max(with_evidence, 1)
    )
    print(f"  Examples with evidence text: {with_evidence}/{len(examples)}")
    print(f"  Avg evidence sentences: {avg_evidence:.1f}")

    # Show a few examples
    print("\n  Sample examples:")
    for ex in examples[:3]:
        claim_preview = ex.claim_or_query[:80] + ("..." if len(ex.claim_or_query) > 80 else "")
        print(f"    [{ex.gold_label}] {claim_preview}")
        if ex.evidence_sentences:
            ev_preview = ex.evidence_sentences[0][:60] + "..."
            print(f"      Evidence: {ev_preview}")

    print("\n  READY for nli-only tier.")
    print(f"{'=' * 60}")


def run_nli_only(
    suite_name: str,
    sample_n: int = 0,
    entailment_threshold: float = 0.5,
    contradiction_threshold: float = 0.5,
    abstract_mode: bool = False,
    seed: int = 42,
    model_name: str | None = None,
    model_path: str | None = None,
    use_rerank: bool = False,
    use_context_window: bool = False,
    use_confidence_margin: bool = False,
    use_minicheck_fallback: bool = False,
    fallback_model: str | None = None,
    attribution_strategy: str = "majority",
    split: str | None = None,
) -> None:
    """Tier 1: Run local NLI model. $0, minutes."""
    from .cache import ResponseCache

    suites = _get_suites()
    suite = suites[suite_name]

    tier_name = "nli-abstract" if abstract_mode else "nli-only"
    model_display = model_path or model_name or "default"

    print(f"\n{'=' * 60}")
    print(f"  {suite.name} | tier={tier_name} | threshold={entailment_threshold} | model={model_display}")
    features = []
    if use_rerank:
        features.append("rerank")
    if use_context_window:
        features.append("context-window")
    if use_confidence_margin:
        features.append("confidence-margin")
    if use_minicheck_fallback:
        fb_display = fallback_model or "MiniCheck"
        features.append(f"fallback({fb_display})")
    if suite_name in ("hagrid", "attribution_bench") and attribution_strategy != "majority":
        features.append(f"strategy={attribution_strategy}")
    if split:
        features.append(f"split={split}")
    if features:
        print(f"  Features: {', '.join(features)}")
    print(f"{'=' * 60}")

    if not suite.is_downloaded():
        print(f"  Downloading {suite_name}...")
        suite.download()

    load_kwargs = {"split": split} if split else {}
    examples = suite.load(**load_kwargs)

    if sample_n and sample_n < len(examples):
        examples = suite.sample(examples, sample_n, seed=seed)
        print(f"  Sampled {len(examples)} examples (seed={seed})")

    cache = ResponseCache(enabled=True)
    nli = _make_nli_instance(cache, model_name=model_name, model_path=model_path)

    print(f"  Running NLI on {len(examples)} examples...")
    start_time = time.time()

    if abstract_mode and hasattr(suite, "run_nli_abstract_retrieval"):
        predictions = suite.run_nli_abstract_retrieval(
            examples, nli,
            entailment_threshold=entailment_threshold,
            contradiction_threshold=contradiction_threshold,
            use_rerank=use_rerank,
            use_context_window=use_context_window,
            use_confidence_margin=use_confidence_margin,
            use_minicheck_fallback=use_minicheck_fallback,
            fallback_model=fallback_model,
        )
    else:
        nli_kwargs: dict = {"entailment_threshold": entailment_threshold}
        if suite_name in ("scifact", "fever"):
            nli_kwargs["contradiction_threshold"] = contradiction_threshold
        nli_kwargs["use_rerank"] = use_rerank
        nli_kwargs["use_confidence_margin"] = use_confidence_margin
        nli_kwargs["use_minicheck_fallback"] = use_minicheck_fallback
        nli_kwargs["fallback_model"] = fallback_model
        if suite_name in ("hagrid", "attribution_bench"):
            nli_kwargs["attribution_strategy"] = attribution_strategy
        predictions = suite.run_nli_only(examples, nli, **nli_kwargs)

    elapsed = time.time() - start_time

    report = suite.compute_metrics(predictions)
    report.tier = tier_name
    report.total_time_s = elapsed
    report.cache_stats = cache.stats()

    print(report.summary())

    # Save results
    result_path = report.save()
    print(f"\n  Results saved to: {result_path}")

    cache.close()


def run_ais(suite_name: str, sample_n: int = 0, entailment_threshold: float = 0.5) -> None:
    """Run AIS metric on a benchmark's examples."""
    from .cache import ResponseCache

    suites = _get_suites()
    suite = suites[suite_name]

    print(f"\n  Running AIS metric on {suite_name}...")

    if not suite.is_downloaded():
        suite.download()

    examples = suite.load()
    if sample_n and sample_n < len(examples):
        examples = suite.sample(examples, sample_n)

    cache = ResponseCache(enabled=True)
    nli = _make_nli_instance(cache)
    scorer = _make_ais_instance(nli, entailment_threshold)

    report = scorer.score_from_benchmark(suite_name, examples)
    print(report.summary())

    cache.close()


def run_calibrate(
    suite_name: str,
    sample_n: int = 0,
    seed: int = 42,
    model_name: str | None = None,
) -> None:
    """Run calibration to find optimal temperature and thresholds."""
    from .cache import ResponseCache
    from .nli import calibrate_temperature, find_optimal_thresholds

    suites = _get_suites()
    suite = suites[suite_name]

    print(f"\n{'=' * 60}")
    print(f"  CALIBRATION: {suite.name}")
    print(f"{'=' * 60}")

    if not suite.is_downloaded():
        suite.download()

    examples = suite.load()
    if sample_n and sample_n < len(examples):
        examples = suite.sample(examples, sample_n, seed=seed)

    # Only calibrate on examples with evidence
    calibration_examples = [ex for ex in examples if ex.evidence_sentences]
    print(f"  Calibrating on {len(calibration_examples)} examples with evidence...")

    cache = ResponseCache(enabled=True)
    nli = _make_nli_instance(cache, model_name=model_name)

    # Build label map from suite
    label_map = {label: label for label in suite.labels}

    # Temperature calibration
    print("\n  --- Temperature Calibration ---")
    best_t = calibrate_temperature(nli, calibration_examples, label_map)
    print(f"  Best temperature: T={best_t}")

    # Per-label threshold search
    print("\n  --- Per-Label Threshold Search ---")
    thresholds = find_optimal_thresholds(nli, calibration_examples, label_map)
    print(f"  Optimal entailment threshold: {thresholds['entailment_threshold']}")
    print(f"  Optimal contradiction threshold: {thresholds['contradiction_threshold']}")

    # Run with optimal thresholds to show improvement
    print("\n  --- Results with Optimal Thresholds ---")
    predictions = suite.run_nli_only(
        calibration_examples, nli,
        entailment_threshold=thresholds["entailment_threshold"],
        **({"contradiction_threshold": thresholds["contradiction_threshold"]}
           if suite_name in ("scifact", "fever") else {}),
    )
    report = suite.compute_metrics(predictions)
    print(report.summary())

    cache.close()
    print(f"{'=' * 60}")


def run_all(tier: str, **kwargs) -> None:
    """Run all benchmarks at the specified tier."""
    suites = _get_suites()
    for name in suites:
        try:
            if tier == "dry-run":
                run_dry_run(name)
            elif tier in ("nli-only", "nli-abstract"):
                run_nli_only(name, abstract_mode=(tier == "nli-abstract"), **kwargs)
        except Exception as e:
            print(f"\n  ERROR running {name}: {e}")
            logger.exception(f"Failed to run {name}")


def run_threshold_sweep(
    suite_name: str,
    sample_n: int = 0,
    seed: int = 42,
    model_name: str | None = None,
) -> None:
    """
    Sweep entailment thresholds to find the optimal operating point.

    Runs NLI at multiple thresholds and reports accuracy/F1 for each.
    Uses cache so only the first run pays compute cost.
    """
    from .cache import ResponseCache

    suites = _get_suites()
    suite = suites[suite_name]

    if not suite.is_downloaded():
        suite.download()

    examples = suite.load()
    if sample_n and sample_n < len(examples):
        examples = suite.sample(examples, sample_n, seed=seed)

    cache = ResponseCache(enabled=True)
    nli = _make_nli_instance(cache, model_name=model_name)

    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

    print(f"\n{'=' * 60}")
    print(f"  THRESHOLD SWEEP: {suite.name} | n={len(examples)}")
    print(f"{'=' * 60}")
    print(f"  {'Threshold':>10s}  {'Accuracy':>10s}  {'F1':>10s}  {'Precision':>10s}  {'Recall':>10s}")
    print(f"  {'-' * 54}")

    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in thresholds:
        predictions = suite.run_nli_only(
            examples, nli,
            entailment_threshold=threshold,
            **({"contradiction_threshold": threshold} if suite_name in ("scifact", "fever") else {}),
        )
        report = suite.compute_metrics(predictions)

        print(f"  {threshold:>10.2f}  {report.accuracy:>10.1%}  {report.f1:>10.1%}  {report.precision:>10.1%}  {report.recall:>10.1%}")

        if report.f1 > best_f1:
            best_f1 = report.f1
            best_threshold = threshold

    print(f"  {'-' * 54}")
    print(f"  Best F1: {best_f1:.1%} at threshold={best_threshold}")
    print(f"\n  Cache stats: {cache.stats()}")
    print("  (Second+ runs are instant due to caching)")
    print(f"{'=' * 60}")

    cache.close()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Run PCA evaluation benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tiers (run in order for cost-efficient iteration):
  dry-run       Load and validate data. $0. Seconds.
  nli-only      Local NLI model.        $0. Minutes.
  nli-abstract  NLI on full abstracts.  $0. Minutes. (SciFact only)
  sweep         Threshold sweep.        $0. Minutes. (Uses cache)
  calibrate     Find optimal T and thresholds. $0. Minutes.
  ais           AIS attribution metric. $0. Minutes.

Models:
  cross-encoder/nli-deberta-v3-base   (184M, default, ~100ms)
  cross-encoder/nli-deberta-v3-large  (435M, ~300ms)
  lytang/MiniCheck-DeBERTa-v3-Large   (355M, purpose-built for fact-checking)
  yaxili96/FactCG-DeBERTa-v3-Large    (355M, binary fact-checking)
  MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli  (435M, high-accuracy NLI)

Examples:
  python -m benchmarks.run scifact --tier dry-run
  python -m benchmarks.run scifact --tier nli-only
  python -m benchmarks.run scifact --tier nli-only --model cross-encoder/nli-deberta-v3-large
  python -m benchmarks.run scifact --tier nli-only --sample 100 --threshold 0.6
  python -m benchmarks.run scifact --tier sweep
  python -m benchmarks.run scifact --tier calibrate
  python -m benchmarks.run fever --tier nli-only --sample 5000
  python -m benchmarks.run all --tier dry-run
  python -m benchmarks.run all --tier nli-only --sample 500
""",
    )
    parser.add_argument(
        "benchmark",
        choices=["scifact", "fever", "qasper", "hagrid", "attribution_bench", "factscore", "all"],
        help="Benchmark to run (or 'all')",
    )
    parser.add_argument(
        "--tier",
        choices=["dry-run", "nli-only", "nli-abstract", "sweep", "calibrate", "ais"],
        default="dry-run",
        help="Execution tier (default: dry-run)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="NLI model name (default: cross-encoder/nli-deberta-v3-base)",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to a local model directory (overrides --model)",
    )
    parser.add_argument(
        "--fallback-model", type=str, default=None,
        help="Model name for fallback when primary model is uncertain (default: MiniCheck)",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Sample N examples (0 = use all). Stratified by label.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Entailment threshold for NLI classification (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="Enable retrieve-and-rerank (concatenate top-3 sentences and re-score)",
    )
    parser.add_argument(
        "--context-window", action="store_true",
        help="Include +/-1 adjacent sentences as context (nli-abstract only)",
    )
    parser.add_argument(
        "--confidence-margin", action="store_true",
        help="Use (entailment - contradiction) margin for classification",
    )
    parser.add_argument(
        "--minicheck-fallback", action="store_true",
        help="Use MiniCheck as fallback when primary model is uncertain",
    )
    parser.add_argument(
        "--split", type=str, default=None,
        help="Data split to use (e.g. 'dev_eval' for HAGRID held-out set). Default: benchmark default.",
    )
    parser.add_argument(
        "--attribution-strategy", type=str, default="majority",
        choices=["majority", "whole", "max-score", "any-sentence", "mean-score", "weighted"],
        help="HAGRID attribution aggregation strategy (default: majority)",
    )

    args = parser.parse_args()

    if not args.model_path:
        print("Note: Using default pre-trained model. Results will differ from "
              "paper-reported numbers which used fine-tuned models. "
              "See README.md for details.")

    nli_flags = {
        "model_name": args.model,
        "model_path": args.model_path,
        "use_rerank": args.rerank,
        "use_context_window": args.context_window,
        "use_confidence_margin": args.confidence_margin,
        "use_minicheck_fallback": args.minicheck_fallback,
        "fallback_model": args.fallback_model,
        "attribution_strategy": args.attribution_strategy,
        "split": args.split,
    }

    if args.benchmark == "all":
        if args.tier == "sweep":
            for name in _get_suites():
                run_threshold_sweep(name, sample_n=args.sample, seed=args.seed, model_name=args.model)
        elif args.tier == "calibrate":
            for name in _get_suites():
                run_calibrate(name, sample_n=args.sample, seed=args.seed, model_name=args.model)
        elif args.tier == "ais":
            for name in _get_suites():
                run_ais(name, sample_n=args.sample, entailment_threshold=args.threshold)
        else:
            run_all(
                args.tier,
                sample_n=args.sample,
                entailment_threshold=args.threshold,
                seed=args.seed,
                **nli_flags,
            )
    elif args.tier == "dry-run":
        run_dry_run(args.benchmark)
    elif args.tier in ("nli-only", "nli-abstract"):
        run_nli_only(
            args.benchmark,
            sample_n=args.sample,
            entailment_threshold=args.threshold,
            abstract_mode=(args.tier == "nli-abstract"),
            seed=args.seed,
            **nli_flags,
        )
    elif args.tier == "sweep":
        run_threshold_sweep(args.benchmark, sample_n=args.sample, seed=args.seed, model_name=args.model)
    elif args.tier == "calibrate":
        run_calibrate(args.benchmark, sample_n=args.sample, seed=args.seed, model_name=args.model)
    elif args.tier == "ais":
        run_ais(args.benchmark, sample_n=args.sample, entailment_threshold=args.threshold)

    return 0


if __name__ == "__main__":
    sys.exit(main())
