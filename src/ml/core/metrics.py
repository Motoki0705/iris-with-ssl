"""Metric aggregation and evaluation helpers for classification and regression.

This module will expose a unified API for computing accuracy, confusion matrices, regression
scores, and SSL reconstruction metrics.

Todo:
    * Provide wrappers around scikit-learn metrics for consistent outputs.
    * Implement aggregation utilities for experiment tracking.
    * Add helpers for formatting metrics for docs/results exports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)


@dataclass(frozen=True)
class ClassificationReport:
    """Container for aggregated classification metrics.

    Attributes:
        accuracy: Share of correctly classified samples.
        precision: Weighted precision across classes.
        recall: Weighted recall across classes.
        f1: Weighted F1 score across classes.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class RegressionReport:
    """Container for aggregated regression metrics.

    Attributes:
        mse: Mean squared error across predictions.
        mae: Mean absolute error across predictions.
        r2: Coefficient of determination.
    """

    mse: float
    mae: float
    r2: float


def _to_1d_array(values: Iterable[int | float]) -> np.ndarray:
    """Convert an iterable of labels or predictions to a NumPy array.

    Args:
        values: Iterable providing numeric labels or predictions.

    Returns:
        numpy.ndarray: Flattened one-dimensional vector.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        return array.ravel()
    return array


def evaluate_classification(
    y_true: Iterable[int | float], y_pred: Iterable[int | float]
) -> ClassificationReport:
    """Evaluate a classification task with standard scalar metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        ClassificationReport: Aggregated accuracy, precision, recall, and F1 score.
    """
    true = _to_1d_array(y_true)
    pred = _to_1d_array(y_pred)
    return ClassificationReport(
        accuracy=accuracy_score(true, pred),
        precision=precision_score(true, pred, average="weighted", zero_division=0),
        recall=recall_score(true, pred, average="weighted", zero_division=0),
        f1=f1_score(true, pred, average="weighted", zero_division=0),
    )


def evaluate_regression(
    y_true: Iterable[float], y_pred: Iterable[float]
) -> RegressionReport:
    """Evaluate a regression task with baseline error metrics.

    Args:
        y_true: Ground-truth numeric targets.
        y_pred: Predicted numeric values.

    Returns:
        RegressionReport: Aggregated mean squared error, mean absolute error, and R².
    """
    true = _to_1d_array(y_true)
    pred = _to_1d_array(y_pred)
    return RegressionReport(
        mse=float(mean_squared_error(true, pred)),
        mae=float(mean_absolute_error(true, pred)),
        r2=float(r2_score(true, pred)),
    )


__all__ = [
    "ClassificationReport",
    "RegressionReport",
    "evaluate_classification",
    "evaluate_regression",
]
