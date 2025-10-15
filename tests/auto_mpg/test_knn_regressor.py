from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml.auto_mpg.knn_regressor import (
    build_knn_regressor,
    evaluate_knn_models,
)


def _build_sample_frame() -> pd.DataFrame:
    """Create a deterministic Auto MPG-like dataset for testing."""
    frame = pd.DataFrame(
        {
            "mpg": [21, 22, 24, 19, 18, 30, 28, 35, 33, 26],
            "horsepower": [110, 115, 120, 105, 100, 85, 90, 70, 75, 95],
            "weight": [2600, 2700, 2800, 3000, 3100, 2200, 2300, 2000, 2100, 2400],
            "displacement": [160, 165, 170, 180, 190, 140, 145, 120, 125, 150],
        }
    )
    # Store horsepower as strings to exercise numeric coercion logic.
    frame["horsepower"] = frame["horsepower"].astype(str)
    return frame


def test_build_knn_regressor_returns_pipeline() -> None:
    pipeline = build_knn_regressor(3, ["horsepower", "weight"])
    assert isinstance(pipeline, Pipeline)
    assert pipeline.named_steps["regressor"].n_neighbors == 3


@pytest.mark.parametrize(
    ("feature_sets", "k_values"),
    [
        ([("horsepower",), ("horsepower", "weight")], [1, 3]),
        ([("displacement", "weight")], [2, 4]),
    ],
)
def test_evaluate_knn_models_returns_expected_rows(
    feature_sets: list[tuple[str, ...]],
    k_values: list[int],
) -> None:
    frame = _build_sample_frame()
    results = evaluate_knn_models(frame, feature_sets, k_values)
    assert len(results) == len(feature_sets) * len(k_values)
    sample = results[0]
    assert sample["model_type"] == "knn"
    assert sample["train_samples"] + sample["test_samples"] == len(frame.dropna())
    for metric in ("mse", "mae", "r2"):
        assert metric in sample
        assert isinstance(sample[metric], float)
