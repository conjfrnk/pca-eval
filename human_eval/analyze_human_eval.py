#!/usr/bin/env python3
"""
Analyze human evaluation results for Proof-Carrying Answers.

Computes Fleiss' kappa, confusion matrices, bootstrap CIs, and generates
summary statistics and visualizations.

Usage:
    python analyze_human_eval.py \
        --expert-csv expert_responses.csv \
        --crowd-csv crowd_responses.csv \
        --system-verdicts system_verdicts.json \
        --output-dir results/
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, bootstrap
from statsmodels.stats.inter_rater import fleiss_kappa


def encode_verdict(verdict: str) -> int:
    """Encode verdict string to integer for Fleiss' kappa."""
    mapping = {"YES": 0, "PARTIAL": 1, "NO": 2, "UNCLEAR": 3}
    return mapping[verdict]


def compute_fleiss_kappa(
    verdicts_matrix: np.ndarray,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute Fleiss' kappa and 95% bootstrap CI.

    Args:
        verdicts_matrix: Shape (n_objects, n_raters), values 0-3 for verdict types

    Returns:
        (kappa, (ci_lower, ci_upper))
    """
    kappa = fleiss_kappa(verdicts_matrix)

    # Bootstrap CI
    def stat(data, axis):
        return fleiss_kappa(data)

    rng = np.random.default_rng(seed=42)
    res = bootstrap(
        (verdicts_matrix,),
        stat,
        n_resamples=10000,
        confidence_level=0.95,
        random_state=rng,
        method="percentile",
    )

    return kappa, (res.confidence_interval.low, res.confidence_interval.high)


def load_responses(
    expert_csv: str,
    crowd_csv: str,
    system_verdicts_json: str,
) -> Tuple[pd.DataFrame, Dict]:
    """Load annotator responses and system verdicts."""

    # Load responses
    expert_df = pd.read_csv(expert_csv)
    crowd_df = pd.read_csv(crowd_csv)

    # Stack: 4 columns (expert_1, expert_2, crowd_1, crowd_2)
    responses = expert_df.merge(
        crowd_df,
        on="object_id",
        suffixes=("_expert", "_crowd")
    )

    # Load system verdicts
    with open(system_verdicts_json) as f:
        system_verdicts = json.load(f)

    responses["system_verdict"] = responses["object_id"].map(
        lambda oid: system_verdicts[oid]
    )

    return responses, system_verdicts


def compute_majority_verdict(row) -> str:
    """Compute majority verdict from 4 annotators."""
    verdicts = [row["expert_1"], row["expert_2"], row["crowd_1"], row["crowd_2"]]
    counts = pd.Series(verdicts).value_counts()
    return counts.index[0]  # Most common


def build_confusion_matrix(
    responses: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build 2x2 confusion matrix: System verdict vs. Human (majority) verdict.

    Returns:
        (confusion_df, metrics_dict)
    """
    responses["human_verdict"] = responses.apply(compute_majority_verdict, axis=1)

    # Map to binary: YES -> SUPPORTED, else -> UNSUPPORTED
    responses["system_binary"] = (responses["system_verdict"] == "SUPPORTED").astype(int)
    responses["human_binary"] = (responses["human_verdict"] == "YES").astype(int)

    confusion = pd.crosstab(
        responses["system_binary"],
        responses["human_binary"],
        rownames=["System"],
        colnames=["Human"],
        margins=True,
    )
    confusion.index = ["UNSUPPORTED", "SUPPORTED", "Total"]
    confusion.columns = ["UNSUPPORTED", "SUPPORTED", "Total"]

    # Extract TP, FP, FN, TN
    tp = confusion.loc["SUPPORTED", "SUPPORTED"]
    fp = confusion.loc["SUPPORTED", "UNSUPPORTED"]
    fn = confusion.loc["UNSUPPORTED", "SUPPORTED"]
    tn = confusion.loc["UNSUPPORTED", "UNSUPPORTED"]

    metrics = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "f1": 2 * (tp / (tp + fp)) * (tp / (tp + fn)) / (
            (tp / (tp + fp)) + (tp / (tp + fn))
        ),
        "accuracy": (tp + tn) / (tp + fp + fn + tn),
    }

    return confusion, metrics


def main():
    parser = argparse.ArgumentParser(description="Analyze human evaluation results.")
    parser.add_argument("--expert-csv", required=True)
    parser.add_argument("--crowd-csv", required=True)
    parser.add_argument("--system-verdicts", required=True)
    parser.add_argument("--output-dir", default="results/")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    # Load data
    print("Loading responses...")
    responses, system_verdicts = load_responses(
        args.expert_csv,
        args.crowd_csv,
        args.system_verdicts,
    )

    # Encode verdicts for Fleiss' kappa
    print("Computing agreement metrics...")
    verdict_cols = ["expert_1", "expert_2", "crowd_1", "crowd_2"]
    verdicts_encoded = responses[verdict_cols].applymap(encode_verdict).values

    # Overall kappa
    kappa_overall, (ci_lower, ci_upper) = compute_fleiss_kappa(verdicts_encoded)
    print(f"Fleiss' κ (overall) = {kappa_overall:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]")

    # Per-scenario kappa
    for scenario in responses["scenario"].unique():
        mask = responses["scenario"] == scenario
        verdicts_scenario = verdicts_encoded[mask]
        kappa_scenario, (ci_l, ci_u) = compute_fleiss_kappa(verdicts_scenario)
        print(f"  κ ({scenario}) = {kappa_scenario:.3f} [{ci_l:.3f}, {ci_u:.3f}]")

    # Majority agreement
    majority_agree = responses.groupby("object_id").apply(
        lambda x: (x[verdict_cols].values[0] == x[verdict_cols].values).sum() >= 3
    ).mean()
    print(f"Majority agreement (≥3/4): {majority_agree:.1%}")

    # System vs. human comparison
    print("\nSystem vs. Human Verdict Comparison:")
    confusion, metrics = build_confusion_matrix(responses)
    print(confusion)
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall: {metrics['recall']:.1%}")
    print(f"F1: {metrics['f1']:.3f}")
    print(f"Accuracy: {metrics['accuracy']:.1%}")

    # Expert vs. crowd correlation
    print("\nExpert vs. Crowd Correlation:")
    expert_verdicts = responses[["expert_1", "expert_2"]].apply(
        compute_majority_verdict, axis=1
    )
    crowd_verdicts = responses[["crowd_1", "crowd_2"]].apply(
        compute_majority_verdict, axis=1
    )

    # Convert to numeric for correlation
    expert_numeric = expert_verdicts.map(lambda x: 0 if x == "YES" else 1)
    crowd_numeric = crowd_verdicts.map(lambda x: 0 if x == "YES" else 1)

    rho, pval = spearmanr(expert_numeric, crowd_numeric)
    print(f"Spearman ρ: {rho:.3f} (p={pval:.4f})")

    # Deflection agreement
    deflections = responses[responses["system_verdict"] == "DEFLECTED"]
    if len(deflections) > 0:
        deflection_agreement = (
            (deflections["expert_1"].isin(["NO", "UNCLEAR"])).sum() +
            (deflections["expert_2"].isin(["NO", "UNCLEAR"])).sum() +
            (deflections["crowd_1"].isin(["NO", "UNCLEAR"])).sum() +
            (deflections["crowd_2"].isin(["NO", "UNCLEAR"])).sum()
        ) / (len(deflections) * 4)
        print(f"Deflection agreement: {deflection_agreement:.1%} (n={len(deflections)})")

    # Save summary to file
    summary = {
        "fleiss_kappa": {
            "overall": kappa_overall,
            "ci": [ci_lower, ci_upper],
        },
        "majority_agreement": majority_agree,
        "system_vs_human": metrics,
        "expert_crowd_correlation": {"rho": rho, "pval": pval},
    }

    with open(f"{args.output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Visualizations
    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Confusion matrix
    sns.heatmap(
        confusion.iloc[:-1, :-1],
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0, 0],
        cbar=False,
    )
    axes[0, 0].set_title("System vs. Human Verdict")
    axes[0, 0].set_xlabel("Human (Majority)")
    axes[0, 0].set_ylabel("System")

    # Metrics bar chart
    metrics_names = ["Precision", "Recall", "F1", "Accuracy"]
    metrics_values = [
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["accuracy"],
    ]
    axes[0, 1].bar(metrics_names, metrics_values, color="#1B5E3B")
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("System Performance vs. Human Judgments")

    # Annotation time distribution
    axes[1, 0].hist(responses["annotation_time"], bins=20, color="#C4A94F", edgecolor="black")
    axes[1, 0].set_xlabel("Time per Annotation (seconds)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Annotation Time Distribution")

    # Kappa by scenario
    scenarios = ["SciFact", "HAGRID", "ClaimVerify"]
    kappas = []
    for scenario in scenarios:
        mask = responses["scenario"] == scenario.lower()
        verdicts_scenario = verdicts_encoded[mask]
        kappa_scenario, _ = compute_fleiss_kappa(verdicts_scenario)
        kappas.append(kappa_scenario)

    axes[1, 1].bar(scenarios, kappas, color=["#1B5E3B", "#C4A94F", "#D4453A"])
    axes[1, 1].axhline(y=0.70, color="gray", linestyle="--", label="Target (0.70)")
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].set_ylabel("Fleiss' κ")
    axes[1, 1].set_title("Agreement by Scenario")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/results.png", dpi=300, bbox_inches="tight")
    print(f"Saved to {args.output_dir}/results.png")


if __name__ == "__main__":
    main()
