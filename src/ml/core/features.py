"""Feature engineering utilities shared by Iris, Auto MPG, and SSL experiments.

This module will define preprocessing pipelines that can be composed by task-specific code.

Todo:
    * Implement scaling, normalization, and encoding transformers.
    * Add reusable feature selection helpers and column schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

IRIS_SCHEMA: Dict[str, List[str]] = {
    "features": [
        "sepal.length (cm)",
        "sepal.width (cm)",
        "petal.length (cm)",
        "petal.width (cm)",
    ],
    "target": ["target"],
}

AUTO_MPG_SCHEMA: Dict[str, List[str]] = {
    "features": [
        "cylinder",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "year",
        "origin",
    ],
    "target": ["mpg"],
}

COLUMN_SCHEMAS: Dict[str, Dict[str, List[str]]] = {
    "iris": IRIS_SCHEMA,
    "auto_mpg": AUTO_MPG_SCHEMA,
}


def build_scaler(transform: str) -> StandardScaler | MinMaxScaler:
    """Instantiate a scaling transformer for numeric features.

    Args:
        transform: Name of the scaler to build. Accepts ``"standard"`` or
            ``"minmax"`` (case-insensitive).

    Returns:
        StandardScaler | MinMaxScaler: Configured scikit-learn scaler instance.

    Raises:
        ValueError: If ``transform`` is not a recognised scaler type.
    """
    transform_normalized = transform.strip().lower()
    if transform_normalized == "standard":
        return StandardScaler()
    if transform_normalized == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unsupported scaler transform '{transform}'.")


@dataclass(frozen=True)
class PolynomialFeatureConfig:
    """Configuration for polynomial feature expansion.

    Attributes:
        degree: Polynomial degree to construct.
        include_bias: Whether to add a bias column of ones.
        interaction_only: Whether to limit expansion to interaction terms.
    """

    degree: int = 2
    include_bias: bool = False
    interaction_only: bool = False


def build_polynomial_features(config: PolynomialFeatureConfig) -> PolynomialFeatures:
    """Create a :class:`~sklearn.preprocessing.PolynomialFeatures` transformer.

    Args:
        config: Dataclass specifying degree, bias term, and interaction behaviour.

    Returns:
        PolynomialFeatures: A configured feature expansion transformer.
    """
    return PolynomialFeatures(
        degree=config.degree,
        include_bias=config.include_bias,
        interaction_only=config.interaction_only,
    )


def select_columns(frame: pd.DataFrame, schema_key: str) -> pd.DataFrame:
    """Project a dataframe onto a schema-defined subset of columns.

    Args:
        frame: Source dataframe to select from.
        schema_key: Selection identifier of the form ``\"<dataset>.<section>\"``.
            Example keys include ``\"iris.features\"`` or ``\"auto_mpg.target\"``.

    Returns:
        pandas.DataFrame: Copy of the selected columns for isolation.

    Raises:
        KeyError: If the schema key or columns are undefined.
    """
    if "." not in schema_key:
        raise KeyError("schema_key must be of the form '<dataset>.<section>'.")
    dataset_key, section_key = schema_key.split(".", maxsplit=1)
    try:
        columns = COLUMN_SCHEMAS[dataset_key][section_key]
    except KeyError as exc:
        raise KeyError(f"Unknown schema key '{schema_key}'.") from exc

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        missing_display = ", ".join(missing)
        raise KeyError(f"Columns {missing_display} were not found in the dataframe.")

    return frame.loc[:, columns].copy()


__all__ = [
    "AUTO_MPG_SCHEMA",
    "COLUMN_SCHEMAS",
    "IRIS_SCHEMA",
    "PolynomialFeatureConfig",
    "build_polynomial_features",
    "build_scaler",
    "select_columns",
]
