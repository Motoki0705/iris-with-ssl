"""K-nearest neighbours regressors for Auto MPG experiments.

This module exposes utilities required by ticket M-01: building parametrised
KNN pipelines and evaluating a grid of ``(feature_set, k)`` combinations with
standard regression metrics.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from ml.core.datasets import train_test_split_frame
from ml.core.features import build_scaler
from ml.core.metrics import evaluate_regression

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


def _ensure_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Validate that the requested columns exist in the dataframe.

    Args:
        frame: Source dataframe to inspect.
        columns: Column labels that must be present.

    Raises:
        KeyError: If any required column is missing.
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        missing_list = ", ".join(missing)
        raise KeyError(f"Columns not found in dataframe: {missing_list}")


def _coerce_numeric(
    frame: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    """Convert the provided columns to numeric dtype and drop NaNs.

    Args:
        frame: Dataframe containing raw Auto MPG samples.
        columns: Column names that should be numeric after coercion.

    Returns:
        pandas.DataFrame: Copy of ``frame`` with selected columns converted
        to float and rows containing NaNs removed.
    """
    numeric_frame = frame.copy()
    for column in columns:
        numeric_frame[column] = pd.to_numeric(
            numeric_frame[column], errors="coerce"
        )
    return numeric_frame.dropna(subset=list(columns))


def _normalise_feature_names(feature_names: Sequence[str]) -> list[str]:
    """Normalise feature names into a unique ordered list.

    Args:
        feature_names: Candidate feature labels.

    Returns:
        list[str]: Ordered list without duplicates.

    Raises:
        ValueError: If the feature list is empty.
    """
    ordered = list(dict.fromkeys(feature_names))
    if not ordered:
        raise ValueError("feature_names must contain at least one column.")
    return ordered


def build_knn_regressor(k: int, feature_names: Sequence[str]) -> Pipeline:
    """Build a standardised KNN regression pipeline for Auto MPG data.

    Args:
        k: Number of neighbours to consider. Must be greater than zero.
        feature_names: Sequence of feature column names to use.

    Returns:
        sklearn.pipeline.Pipeline: Pipeline comprising column selection,
        standard scaling, and :class:`~sklearn.neighbors.KNeighborsRegressor`.

    Raises:
        ValueError: If ``k`` is less than one or ``feature_names`` is empty.
    """
    if k < 1:
        raise ValueError("k must be a positive integer.")
    features = _normalise_feature_names(feature_names)
    scaler = build_scaler("standard")
    preprocessor = ColumnTransformer(
        transformers=[("scale", scaler, features)],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", KNeighborsRegressor(n_neighbors=k)),
        ]
    )


def evaluate_knn_models(
    frame: pd.DataFrame,
    feature_sets: Sequence[Sequence[str]],
    k_values: Sequence[int],
    target: str = "mpg",
) -> list[dict[str, Any]]:
    """Evaluate KNN models across feature subsets and neighbour counts.

    Args:
        frame: Auto MPG dataframe containing the target column and candidate
            feature columns.
        feature_sets: Iterable of feature name sequences to evaluate.
        k_values: Iterable of ``k`` values for the KNN regressor.
        target: Target column name. Defaults to ``\"mpg\"``.

    Returns:
        list[dict[str, Any]]: Flattened evaluation records per model, each
        containing metadata and regression metrics.

    Raises:
        ValueError: If ``feature_sets`` or ``k_values`` are empty, or if the
            cleaned dataframe contains no rows for a configuration.
        KeyError: If requested columns are absent from ``frame``.
    """
    if not feature_sets:
        raise ValueError("feature_sets must contain at least one entry.")
    if not k_values:
        raise ValueError("k_values must contain at least one entry.")

    results: list[dict[str, Any]] = []
    for feature_set in feature_sets:
        features = _normalise_feature_names(feature_set)
        required_columns = features + [target]
        _ensure_columns(frame, required_columns)

        numeric_frame = _coerce_numeric(frame, required_columns)
        if numeric_frame.empty:
            raise ValueError(
                "No samples remain after cleaning for features "
                f"{features} and target '{target}'."
            )

        train_frame, test_frame = train_test_split_frame(
            numeric_frame,
            test_size=DEFAULT_TEST_SIZE,
            random_state=DEFAULT_RANDOM_STATE,
        )
        y_train = train_frame[target]
        y_test = test_frame[target]

        for k in k_values:
            pipeline = build_knn_regressor(k=k, feature_names=features)
            pipeline.fit(train_frame, y_train)
            predictions = pipeline.predict(test_frame)
            metrics = evaluate_regression(y_test, predictions)
            results.append(
                {
                    "model_type": "knn",
                    "name": f"knn_k{k}_{'_'.join(features)}",
                    "features": tuple(features),
                    "target": target,
                    "k": k,
                    "train_samples": int(train_frame.shape[0]),
                    "test_samples": int(test_frame.shape[0]),
                    "mse": metrics.mse,
                    "mae": metrics.mae,
                    "r2": metrics.r2,
                }
            )
    return results


__all__ = [
    "build_knn_regressor",
    "evaluate_knn_models",
]
