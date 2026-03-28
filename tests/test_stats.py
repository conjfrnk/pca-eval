"""Tests for statistical analysis utilities."""

import json

import pytest

from benchmarks.stats import (
    BootstrapResult,
    accuracy,
    bootstrap_ci,
    compute_all_cis,
    harmonic_macro_f1,
    load_predictions_from_result,
    mcnemar_test,
    sklearn_macro_f1,
)


class TestAccuracy:
    def test_perfect(self):
        assert accuracy(["A", "B", "C"], ["A", "B", "C"]) == 1.0

    def test_all_wrong(self):
        assert accuracy(["A", "B", "C"], ["B", "C", "A"]) == 0.0

    def test_mixed(self):
        assert accuracy(["A", "B", "A", "B"], ["A", "A", "A", "A"]) == 0.5


class TestHarmonicMacroF1:
    def test_perfect(self):
        y_true = ["A", "B", "A", "B"]
        y_pred = ["A", "B", "A", "B"]
        assert harmonic_macro_f1(y_true, y_pred) == 1.0

    def test_all_wrong(self):
        y_true = ["A", "A", "B", "B"]
        y_pred = ["B", "B", "A", "A"]
        assert harmonic_macro_f1(y_true, y_pred) == 0.0

    def test_mixed(self):
        y_true = ["A", "B", "A", "B"]
        y_pred = ["A", "A", "A", "B"]
        f1 = harmonic_macro_f1(y_true, y_pred)
        assert 0.0 < f1 < 1.0

    def test_three_class(self):
        y_true = ["A", "B", "C", "A", "B", "C"]
        y_pred = ["A", "B", "C", "A", "B", "C"]
        assert harmonic_macro_f1(y_true, y_pred) == 1.0


class TestSklearnMacroF1:
    def test_perfect(self):
        y_true = ["A", "B", "A", "B"]
        y_pred = ["A", "B", "A", "B"]
        assert sklearn_macro_f1(y_true, y_pred) == 1.0

    def test_all_wrong(self):
        y_true = ["A", "A", "B", "B"]
        y_pred = ["B", "B", "A", "A"]
        assert sklearn_macro_f1(y_true, y_pred) == 0.0

    def test_close_to_harmonic_for_balanced(self):
        """For balanced datasets, sklearn macro and harmonic macro should be close."""
        y_true = ["A", "B"] * 50
        y_pred = ["A", "B"] * 45 + ["B", "A"] * 5
        h = harmonic_macro_f1(y_true, y_pred)
        s = sklearn_macro_f1(y_true, y_pred)
        assert abs(h - s) < 0.05  # Within 5pp


class TestBootstrapCI:
    def test_basic(self):
        y_true = ["A", "B"] * 50
        y_pred = ["A", "B"] * 40 + ["B", "A"] * 10
        result = bootstrap_ci(y_true, y_pred, accuracy, n_bootstrap=1000, seed=42)

        assert isinstance(result, BootstrapResult)
        assert result.point_estimate == 0.8
        assert 0.6 < result.ci_lower < 0.8
        assert 0.8 < result.ci_upper < 1.0
        assert result.n_examples == 100
        assert result.n_bootstrap == 1000

    def test_perfect_predictions(self):
        y_true = ["A", "B"] * 50
        y_pred = ["A", "B"] * 50
        result = bootstrap_ci(y_true, y_pred, accuracy, n_bootstrap=100, seed=42)

        assert result.point_estimate == 1.0
        assert result.ci_lower == 1.0
        assert result.ci_upper == 1.0

    def test_reproducibility(self):
        y_true = ["A", "B"] * 50
        y_pred = ["A", "B"] * 40 + ["B", "A"] * 10
        r1 = bootstrap_ci(y_true, y_pred, accuracy, n_bootstrap=100, seed=42)
        r2 = bootstrap_ci(y_true, y_pred, accuracy, n_bootstrap=100, seed=42)

        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_different_seeds(self):
        y_true = ["A", "B"] * 50
        y_pred = ["A", "B"] * 40 + ["B", "A"] * 10
        r1 = bootstrap_ci(y_true, y_pred, accuracy, n_bootstrap=100, seed=42)
        r2 = bootstrap_ci(y_true, y_pred, accuracy, n_bootstrap=100, seed=99)

        # Different seeds should generally give different CIs
        # (could be the same by chance, but very unlikely)
        assert r1.ci_lower != r2.ci_lower or r1.ci_upper != r2.ci_upper

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([], [], accuracy)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci(["A"], ["A", "B"], accuracy)

    def test_ci_bounds_valid(self):
        y_true = ["A", "B", "C"] * 30
        y_pred = ["A", "B", "C"] * 25 + ["B", "C", "A"] * 5
        result = bootstrap_ci(y_true, y_pred, sklearn_macro_f1, n_bootstrap=500, seed=42)

        assert result.ci_lower <= result.point_estimate
        assert result.ci_upper >= result.point_estimate
        assert 0.0 <= result.ci_lower
        assert result.ci_upper <= 1.0

    def test_to_dict(self):
        y_true = ["A", "B"] * 50
        y_pred = ["A", "B"] * 50
        result = bootstrap_ci(y_true, y_pred, accuracy, n_bootstrap=100, seed=42)
        d = result.to_dict()

        assert "metric" in d
        assert "point_estimate" in d
        assert "ci_lower" in d
        assert "ci_upper" in d
        assert "n_bootstrap" in d
        assert "n_examples" in d

    def test_str_format(self):
        y_true = ["A", "B"] * 50
        y_pred = ["A", "B"] * 50
        result = bootstrap_ci(y_true, y_pred, accuracy, n_bootstrap=100, seed=42)
        s = str(result)

        assert "100.0%" in s
        assert "95% CI" in s


class TestMcNemar:
    def test_identical_models(self):
        y_true = ["A", "B"] * 50
        preds = ["A", "B"] * 40 + ["B", "A"] * 10
        result = mcnemar_test(y_true, preds, preds)

        assert result.chi2 == 0.0
        assert result.p_value == 1.0
        assert not result.significant
        assert result.n_a_only == 0
        assert result.n_b_only == 0

    def test_different_models(self):
        y_true = ["A", "B"] * 50
        preds_a = ["A", "B"] * 50  # Perfect
        preds_b = ["A", "B"] * 30 + ["B", "A"] * 20  # Worse
        result = mcnemar_test(y_true, preds_a, preds_b)

        assert result.n_a_only > 0
        assert result.n_b_only == 0
        assert result.significant  # Should be significant
        assert result.p_value < 0.05

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            mcnemar_test(["A"], ["A"], ["A", "B"])

    def test_to_dict(self):
        y_true = ["A", "B"] * 50
        preds = ["A", "B"] * 50
        result = mcnemar_test(y_true, preds, preds)
        d = result.to_dict()

        assert "chi2" in d
        assert "p_value" in d
        assert "significant_at_05" in d

    def test_contingency_table_sums(self):
        y_true = ["A", "B"] * 50
        preds_a = ["A", "B"] * 40 + ["B", "A"] * 10
        preds_b = ["A", "B"] * 35 + ["B", "A"] * 15
        result = mcnemar_test(y_true, preds_a, preds_b)

        total = result.n_both_correct + result.n_a_only + result.n_b_only + result.n_both_wrong
        assert total == 100


class TestComputeAllCIs:
    def test_returns_all_metrics(self):
        y_true = ["A", "B"] * 50
        y_pred = ["A", "B"] * 40 + ["B", "A"] * 10
        results = compute_all_cis(y_true, y_pred, n_bootstrap=100, seed=42)

        assert "accuracy" in results
        assert "harmonic_macro_f1" in results
        assert "sklearn_macro_f1" in results

        for name, result in results.items():
            assert isinstance(result, BootstrapResult)
            assert result.metric_name == name


class TestLoadPredictions:
    def test_load_from_file(self, tmp_path):
        data = {
            "benchmark_name": "test",
            "tier": "nli-only",
            "num_examples": 3,
            "f1": 0.667,
            "accuracy": 0.667,
            "predictions": [
                {"gold_label": "A", "predicted_label": "A"},
                {"gold_label": "B", "predicted_label": "B"},
                {"gold_label": "A", "predicted_label": "B"},
            ],
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))

        y_true, y_pred, meta = load_predictions_from_result(path)

        assert y_true == ["A", "B", "A"]
        assert y_pred == ["A", "B", "B"]
        assert meta["benchmark_name"] == "test"
        assert meta["f1"] == 0.667

    def test_empty_predictions_raises(self, tmp_path):
        data = {"predictions": []}
        path = tmp_path / "empty.json"
        path.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="No predictions"):
            load_predictions_from_result(path)
