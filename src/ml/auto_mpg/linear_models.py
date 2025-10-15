"""Linear and polynomial regression utilities for Auto MPG analyses.

Ticket M-02 requires reusable builders for linear and polynomial regressors as
well as an evaluation helper that mirrors the behaviour of the KNN utilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from ml.core.metrics import evaluate_regression
from ml.core.features import (
    PolynomialFeatureConfig,
    build_polynomial_features,
    build_scaler,
)
from ml.core.datasets import train_test_split_frame

from .knn_regressor import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    _coerce_numeric,
    _ensure_columns,
    _normalise_feature_names,
)


def _resolve_regressor(
    config: Mapping[str, Any] | None = None
) -> LinearRegression | Ridge:
    """Instantiate a linear model according to configuration hints.

    Args:
        config: Optional mapping describing the regressor type. Supported keys:
            - ``type``: ``\"linear\"`` (default) or ``\"ridge\"``.
            - ``alpha``: Ridge regularisation strength (only if ``type`` is ``\"ridge\"``).

    Returns:
        LinearRegression | Ridge: Scikit-learn regressor instance.
    """
    if not config:
        return LinearRegression()
    regressor_type = config.get("type", "linear").lower()
    if regressor_type == "linear":
        include_bias = bool(config.get("fit_intercept", True))
        return LinearRegression(fit_intercept=include_bias)
    if regressor_type == "ridge":
        alpha = float(config.get("alpha", 1.0))
        fit_intercept = bool(config.get("fit_intercept", True))
        return Ridge(alpha=alpha, fit_intercept=fit_intercept)
    raise ValueError(f"Unsupported regressor type '{regressor_type}'.")


def build_linear_regressor(
    feature_names: Sequence[str], include_bias: bool = True
) -> Pipeline:
    """Construct a linear regression pipeline with standard scaling.

    Args:
        feature_names: Feature columns to include in the regression.
        include_bias: Whether the underlying :class:`LinearRegression` should
            fit an intercept term.

    Returns:
        sklearn.pipeline.Pipeline: Pipeline combining scaling and linear regression.
    """
    features = _normalise_feature_names(feature_names)
    scaler = build_scaler("standard")
    preprocessor = ColumnTransformer(
        transformers=[("scale", scaler, features)],
        remainder="drop",
    )
    regressor = LinearRegression(fit_intercept=include_bias)
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", regressor),
        ]
    )


def build_polynomial_regressor(
    feature_names: Sequence[str],
    degree: int = 2,
    include_bias: bool = False,
    regressor_config: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Construct a polynomial regression pipeline for Auto MPG data.

    Args:
        feature_names: Feature columns to include in the regression.
        degree: Polynomial degree used for feature expansion.
        include_bias: Whether to include a bias column within the polynomial
            features. When ``False`` the downstream regressor handles the bias.
        regressor_config: Optional hints for the terminal regressor. See
            :func:`_resolve_regressor` for supported keys.

    Returns:
        sklearn.pipeline.Pipeline: Pipeline combining scaling, polynomial
        feature expansion, and a linear regressor.
    """
    if degree < 1:
        raise ValueError("degree must be at least 1.")
    features = _normalise_feature_names(feature_names)
    scaler = build_scaler("standard")
    polynomial = build_polynomial_features(
        PolynomialFeatureConfig(
            degree=degree,
            include_bias=include_bias,
            interaction_only=False,
        )
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("scale", scaler),
            ("poly", polynomial),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("features", numeric_pipeline, features)],
        remainder="drop",
    )
    regressor = _resolve_regressor(regressor_config)
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", regressor),
        ]
    )


def _extract_config_value(
    config: Mapping[str, Any], key: str, default: Any | None = None
) -> Any:
    """Retrieve a configuration value while providing a helpful error message.

    Args:
        config: Configuration mapping.
        key: Key that must exist in ``config``.
        default: Optional default value if ``key`` is missing.

    Returns:
        Any: Value stored under ``key`` or ``default`` when provided.

    Raises:
        KeyError: If ``key`` is missing and no default was specified.
    """
    if key in config:
        return config[key]
    if default is not None:
        return default
    raise KeyError(f"Configuration entry '{key}' is required.")


def evaluate_linear_models(
    frame: pd.DataFrame,
    configs: Sequence[Mapping[str, Any]],
    target: str = "mpg",
) -> list[dict[str, Any]]:
    """Evaluate linear and polynomial regression configurations.

    Args:
        frame: Auto MPG dataframe containing features and target.
        configs: Sequence of configuration mappings. Each mapping must contain
            ``model_type`` (``\"linear\"`` or ``\"polynomial\"``) and
            ``feature_names``. Optional keys include ``degree``, ``include_bias``,
            ``regressor`` (sub-config passed to :func:`_resolve_regressor`), and
            ``name`` for human-readable labelling.
        target: Target column name. Defaults to ``\"mpg\"``.

    Returns:
        list[dict[str, Any]]: Evaluation records with metrics for every config.

    Raises:
        ValueError: If ``configs`` is empty or cleaning removes all samples.
        KeyError: If required keys are absent from the dataframe or configs.
    """
    if not configs:
        raise ValueError("configs must contain at least one configuration.")

    results: list[dict[str, Any]] = []
    for config in configs:
        model_type = _extract_config_value(config, "model_type").lower()
        feature_names = _extract_config_value(config, "feature_names")
        features = _normalise_feature_names(feature_names)
        required_columns = features + [target]
        _ensure_columns(frame, required_columns)

        numeric_frame = _coerce_numeric(frame, required_columns)
        if numeric_frame.empty:
            raise ValueError(
                "No samples remain after cleaning for config "
                f"{config!r} and target '{target}'."
            )

        train_frame, test_frame = train_test_split_frame(
            numeric_frame,
            test_size=DEFAULT_TEST_SIZE,
            random_state=DEFAULT_RANDOM_STATE,
        )
        y_train = train_frame[target]
        y_test = test_frame[target]

        if model_type == "linear":
            include_bias = bool(config.get("include_bias", True))
            pipeline = build_linear_regressor(
                feature_names=features, include_bias=include_bias
            )
            parameter_summary = {"include_bias": include_bias}
        elif model_type == "polynomial":
            degree = int(config.get("degree", 2))
            include_bias = bool(config.get("include_bias", False))
            regressor_conf = config.get("regressor")
            pipeline = build_polynomial_regressor(
                feature_names=features,
                degree=degree,
                include_bias=include_bias,
                regressor_config=regressor_conf,
            )
            parameter_summary = {
                "degree": degree,
                "include_bias": include_bias,
            }
            if isinstance(regressor_conf, Mapping):
                parameter_summary["regressor"] = dict(regressor_conf)
        else:
            raise ValueError(
                "model_type must be either 'linear' or 'polynomial'. "
                f"Received {model_type!r}."
            )

        pipeline.fit(train_frame, y_train)
        predictions = pipeline.predict(test_frame)
        metrics = evaluate_regression(y_test, predictions)

        results.append(
            {
                "model_type": model_type,
                "name": config.get(
                    "name",
                    f"{model_type}_{'_'.join(features)}",
                ),
                "features": tuple(features),
                "target": target,
                "parameters": parameter_summary,
                "train_samples": int(train_frame.shape[0]),
                "test_samples": int(test_frame.shape[0]),
                "mse": metrics.mse,
                "mae": metrics.mae,
                "r2": metrics.r2,
            }
        )
    return results


__all__ = [
    "build_linear_regressor",
    "build_polynomial_regressor",
    "evaluate_linear_models",
]
