"""Smoke tests for metric evaluation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ml.core.metrics import (
    ClassificationReport,
    RegressionReport,
    evaluate_classification,
    evaluate_regression,
)


def test_evaluate_classification_returns_expected_scores() -> None:
    """Classification evaluation should aggregate standard metrics correctly."""

    y_true = np.array([0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 2])

    report = evaluate_classification(y_true, y_pred)

    assert isinstance(report, ClassificationReport)
    assert report.accuracy == pytest.approx(0.75)
    assert report.precision == pytest.approx(0.875)
    assert report.recall == pytest.approx(0.75)
    assert report.f1 == pytest.approx(0.75)


def test_evaluate_regression_returns_expected_scores() -> None:
    """Regression evaluation should aggregate error metrics correctly."""

    y_true = np.array([3.0, 4.5, 5.0, 6.5])
    y_pred = np.array([2.5, 4.7, 4.9, 6.2])

    report = evaluate_regression(y_true, y_pred)

    assert isinstance(report, RegressionReport)
    assert report.mse == pytest.approx(0.0975)
    assert report.mae == pytest.approx(0.275)
    assert report.r2 == pytest.approx(0.9376, rel=1e-3)
