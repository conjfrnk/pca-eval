"""
Statistical analysis utilities for benchmark results.

Provides bootstrap confidence intervals, McNemar's paired comparison test,
and standard sklearn metric wrappers for rigorous evaluation reporting.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import f1_score as sklearn_f1

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Result of bootstrap confidence interval estimation."""
    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_bootstrap: int
    n_examples: int

    def __str__(self) -> str:
        return (
            f"{self.metric_name}: {self.point_estimate:.1%} "
            f"[{self.ci_lower:.1%}, {self.ci_upper:.1%}] "
            f"(95% CI, n={self.n_examples})"
        )

    def to_dict(self) -> dict:
        return {
            "metric": self.metric_name,
            "point_estimate": round(self.point_estimate, 6),
            "ci_lower": round(self.ci_lower, 6),
            "ci_upper": round(self.ci_upper, 6),
            "ci_level": self.ci_level,
            "n_bootstrap": self.n_bootstrap,
            "n_examples": self.n_examples,
        }


@dataclass
class McNemarResult:
    """Result of McNemar's paired comparison test."""
    chi2: float
    p_value: float
    n_both_correct: int
    n_a_only: int
    n_b_only: int
    n_both_wrong: int
    significant: bool  # at alpha=0.05

    def __str__(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"McNemar's chi2={self.chi2:.3f}, p={self.p_value:.4f} ({sig})\n"
            f"  Both correct: {self.n_both_correct}, A only: {self.n_a_only}, "
            f"B only: {self.n_b_only}, Both wrong: {self.n_both_wrong}"
        )

    def to_dict(self) -> dict:
        return {
            "chi2": round(self.chi2, 6),
            "p_value": round(self.p_value, 6),
            "n_both_correct": self.n_both_correct,
            "n_a_only": self.n_a_only,
            "n_b_only": self.n_b_only,
            "n_both_wrong": self.n_both_wrong,
            "significant_at_05": self.significant,
        }


def bootstrap_ci(
    y_true: list[str],
    y_pred: list[str],
    metric_fn: "Callable[[list[str], list[str]], float]",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """
    Compute bootstrap confidence interval for a classification metric.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        metric_fn: Function(y_true, y_pred) -> float. Applied to each bootstrap sample.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with point estimate and CI bounds.
    """
    n = len(y_true)
    if n == 0:
        raise ValueError("Empty predictions list")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} vs {len(y_pred)}")

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Point estimate on full data
    point_estimate = metric_fn(y_true, y_pred)

    # Bootstrap
    rng = np.random.default_rng(seed)
    bootstrap_scores = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_true = y_true_arr[indices].tolist()
        boot_pred = y_pred_arr[indices].tolist()
        bootstrap_scores[i] = metric_fn(boot_true, boot_pred)

    # Percentile CI
    alpha = 1 - ci
    ci_lower = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        metric_name=getattr(metric_fn, "__name__", "metric"),
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci,
        n_bootstrap=n_bootstrap,
        n_examples=n,
    )


def mcnemar_test(
    y_true: list[str],
    preds_a: list[str],
    preds_b: list[str],
    alpha: float = 0.05,
) -> McNemarResult:
    """
    McNemar's test for paired comparison of two models on the same examples.

    Tests whether two models have significantly different error rates.

    Args:
        y_true: Ground truth labels (same for both models).
        preds_a: Predictions from model A.
        preds_b: Predictions from model B.
        alpha: Significance level.

    Returns:
        McNemarResult with chi2, p-value, and contingency counts.
    """
    n = len(y_true)
    if n != len(preds_a) or n != len(preds_b):
        raise ValueError("All lists must have the same length")

    # Build 2x2 contingency table of correctness
    n_both_correct = 0
    n_a_only = 0  # A correct, B wrong
    n_b_only = 0  # B correct, A wrong
    n_both_wrong = 0

    for true, a, b in zip(y_true, preds_a, preds_b, strict=True):
        a_correct = (a == true)
        b_correct = (b == true)
        if a_correct and b_correct:
            n_both_correct += 1
        elif a_correct and not b_correct:
            n_a_only += 1
        elif not a_correct and b_correct:
            n_b_only += 1
        else:
            n_both_wrong += 1

    # McNemar's test with continuity correction
    discordant = n_a_only + n_b_only
    if discordant == 0:
        return McNemarResult(
            chi2=0.0, p_value=1.0,
            n_both_correct=n_both_correct, n_a_only=n_a_only,
            n_b_only=n_b_only, n_both_wrong=n_both_wrong,
            significant=False,
        )

    chi2 = (abs(n_a_only - n_b_only) - 1) ** 2 / discordant
    p_value = float(scipy_stats.chi2.sf(chi2, df=1))

    return McNemarResult(
        chi2=chi2,
        p_value=p_value,
        n_both_correct=n_both_correct,
        n_a_only=n_a_only,
        n_b_only=n_b_only,
        n_both_wrong=n_both_wrong,
        significant=(p_value < alpha),
    )


# -- Metric functions for use with bootstrap_ci --


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Simple accuracy metric."""
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    return correct / len(y_true) if y_true else 0.0


def sklearn_macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Standard sklearn macro-averaged F1 (mean of per-class F1)."""
    return float(sklearn_f1(y_true, y_pred, average="macro", zero_division=0))


def sklearn_binary_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Sklearn binary F1 for the positive class."""
    labels = sorted(set(y_true) | set(y_pred))
    if len(labels) != 2:
        return sklearn_macro_f1(y_true, y_pred)
    return float(sklearn_f1(y_true, y_pred, average="binary", pos_label=labels[0], zero_division=0))


def harmonic_macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """
    Harmonic-macro F1: F1(macro_P, macro_R).

    This is the metric used in benchmarks/base.py:compute_metrics().
    It computes macro precision and recall, then their harmonic mean.
    Uses numpy arrays internally for performance in bootstrap loops.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    precisions, recalls = [], []
    for label in labels:
        true_mask = y_true_arr == label
        pred_mask = y_pred_arr == label
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)

    macro_prec = sum(precisions) / len(precisions) if precisions else 0.0
    macro_rec = sum(recalls) / len(recalls) if recalls else 0.0
    if (macro_prec + macro_rec) == 0:
        return 0.0
    return 2 * macro_prec * macro_rec / (macro_prec + macro_rec)


def compute_all_cis(
    y_true: list[str],
    y_pred: list[str],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, BootstrapResult]:
    """
    Compute bootstrap CIs for all standard metrics.

    Returns dict mapping metric name to BootstrapResult.
    """
    metrics = {
        "accuracy": accuracy,
        "harmonic_macro_f1": harmonic_macro_f1,
        "sklearn_macro_f1": sklearn_macro_f1,
    }

    results = {}
    for name, fn in metrics.items():
        result = bootstrap_ci(y_true, y_pred, fn, n_bootstrap=n_bootstrap, seed=seed)
        result.metric_name = name
        results[name] = result

    return results


def load_predictions_from_result(path: str | Path) -> tuple[list[str], list[str], dict]:
    """
    Load y_true and y_pred from a saved benchmark result JSON.

    Returns:
        (y_true, y_pred, metadata) where metadata includes benchmark_name, tier, etc.
    """
    path = Path(path)
    data = json.loads(path.read_text())

    predictions = data.get("predictions", [])
    if not predictions:
        raise ValueError(f"No predictions found in {path}")

    y_true = [p["gold_label"] for p in predictions]
    y_pred = [p["predicted_label"] for p in predictions]

    metadata = {
        "benchmark_name": data.get("benchmark_name", ""),
        "tier": data.get("tier", ""),
        "num_examples": data.get("num_examples", 0),
        "f1": data.get("f1", 0),
        "accuracy": data.get("accuracy", 0),
        "source_file": path.name,
    }

    return y_true, y_pred, metadata
