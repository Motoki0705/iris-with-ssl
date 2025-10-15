from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from ml.auto_mpg.linear_models import (
    build_linear_regressor,
    build_polynomial_regressor,
    evaluate_linear_models,
)


def _sample_frame() -> pd.DataFrame:
    """Return a lightweight dataset mirroring Auto MPG structure."""
    frame = pd.DataFrame(
        {
            "mpg": [21, 22, 24, 19, 18, 30, 28, 35, 33, 26],
            "horsepower": [110, 115, 120, 105, 100, 85, 90, 70, 75, 95],
            "weight": [2600, 2700, 2800, 3000, 3100, 2200, 2300, 2000, 2100, 2400],
            "displacement": [160, 165, 170, 180, 190, 140, 145, 120, 125, 150],
        }
    )
    frame["horsepower"] = frame["horsepower"].astype(str)
    return frame


def test_build_linear_regressor_configures_linear_regression() -> None:
    pipeline = build_linear_regressor(["horsepower", "weight"], include_bias=False)
    assert isinstance(pipeline, Pipeline)
    regressor = pipeline.named_steps["regressor"]
    assert isinstance(regressor, LinearRegression)
    assert regressor.fit_intercept is False


def test_build_polynomial_regressor_embeds_polynomial_features() -> None:
    frame = _sample_frame()
    pipeline = build_polynomial_regressor(
        ["horsepower", "weight"], degree=3, include_bias=True
    )
    pipeline.fit(frame, frame["mpg"])
    preprocessor = pipeline.named_steps["preprocess"]
    assert isinstance(preprocessor, ColumnTransformer)
    inner_pipeline = preprocessor.named_transformers_["features"]
    assert "poly" in inner_pipeline.named_steps
    assert inner_pipeline.named_steps["poly"].degree == 3


def test_evaluate_linear_models_returns_metrics_for_each_config() -> None:
    frame = _sample_frame()
    configs = [
        {
            "model_type": "linear",
            "feature_names": ["horsepower"],
            "include_bias": True,
            "name": "linear_hp",
        },
        {
            "model_type": "polynomial",
            "feature_names": ["horsepower"],
            "degree": 2,
            "name": "poly_hp_deg2",
        },
        {
            "model_type": "polynomial",
            "feature_names": ["horsepower", "weight"],
            "degree": 3,
            "include_bias": False,
            "regressor": {"type": "ridge", "alpha": 0.5},
            "name": "poly_ridge",
        },
    ]
    results = evaluate_linear_models(frame, configs)
    assert len(results) == len(configs)
    ridge_result = next(
        item for item in results if item["name"] == "poly_ridge"
    )
    assert ridge_result["parameters"]["regressor"]["type"] == "ridge"
    assert isinstance(ridge_result["parameters"]["regressor"]["alpha"], float)
    for record in results:
        assert record["model_type"] in {"linear", "polynomial"}
        assert isinstance(record["mse"], float)
        assert isinstance(record["r2"], float)
