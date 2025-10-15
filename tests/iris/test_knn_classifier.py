"""Tests for the Iris KNN classification utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ml.core.features import IRIS_SCHEMA
from ml.iris.knn_classifier import (
    KnnConfig,
    compute_confusion_matrix,
    evaluate_classifier,
    prepare_dataset,
    train_classifier,
)


def test_prepare_dataset_respects_split_ratio() -> None:
    """Prepared datasets should honour the configured test_size proportion."""

    config = KnnConfig(
        features=["sepal.length (cm)", "sepal.width (cm)"],
        k=3,
        scaler="standard",
        test_size=0.3,
        random_state=7,
    )
    x_train, y_train, x_test, y_test = prepare_dataset(config)

    total_samples = len(x_train) + len(x_test)
    expected_test = total_samples * config.test_size
    assert len(y_train) + len(y_test) == total_samples
    assert len(x_test) == pytest.approx(expected_test, abs=1)
    assert list(x_train.columns) == list(config.features)
    assert list(x_test.columns) == list(config.features)


def test_train_classifier_reaches_minimum_accuracy() -> None:
    """Baseline configuration should achieve acceptable accuracy."""

    config = KnnConfig(
        features=IRIS_SCHEMA["features"],
        k=5,
        scaler="standard",
        test_size=0.2,
        random_state=42,
    )
    _, _, x_test, y_test = prepare_dataset(config)
    model = train_classifier(config)
    report = evaluate_classifier(model, x_test, y_test)

    assert report.accuracy >= 0.8
    assert math.isclose(report.accuracy, report.recall, rel_tol=0.2)


def test_compute_confusion_matrix_returns_square_matrix() -> None:
    """Confusion matrix should be 3x3 with positive diagonals for Iris species."""

    config = KnnConfig(
        features=["petal.length (cm)", "petal.width (cm)"],
        k=3,
        scaler="minmax",
        test_size=0.25,
        random_state=2,
    )
    _, _, x_test, y_test = prepare_dataset(config)
    model = train_classifier(config)
    matrix = compute_confusion_matrix(model, x_test, y_test)

    assert matrix.shape == (3, 3)
    assert np.all(matrix.values.diagonal() > 0)
