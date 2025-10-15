"""Smoke tests for dataset loading utilities in ml.core.datasets."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.core.datasets import (
    load_auto_mpg_dataset,
    load_iris_dataset,
    train_test_split_frame,
)


def test_load_iris_dataset_returns_non_empty_dataframe(iris_frame: pd.DataFrame) -> None:
    """The Iris loader should surface the CSV as a populated DataFrame."""

    assert isinstance(iris_frame, pd.DataFrame)
    assert not iris_frame.empty
    expected_columns = {
        "sepal.length (cm)",
        "sepal.width (cm)",
        "petal.length (cm)",
        "petal.width (cm)",
        "target",
    }
    assert expected_columns.issubset(set(iris_frame.columns))


def test_load_auto_mpg_dataset_returns_non_empty_dataframe(
    auto_mpg_frame: pd.DataFrame,
) -> None:
    """The Auto MPG loader should surface the CSV as a populated DataFrame."""

    assert isinstance(auto_mpg_frame, pd.DataFrame)
    assert not auto_mpg_frame.empty
    expected_columns = {"mpg", "horsepower", "weight", "displacement"}
    assert expected_columns.issubset(set(auto_mpg_frame.columns))


@pytest.mark.parametrize("test_size", [0.2, 0.3])
def test_train_test_split_frame_preserves_total_rows(
    iris_frame: pd.DataFrame, test_size: float
) -> None:
    """Splitting should produce train/test partitions summing to the input size."""

    train_frame, test_frame = train_test_split_frame(
        iris_frame, test_size=test_size, random_state=42
    )
    assert len(train_frame) + len(test_frame) == len(iris_frame)
    assert train_frame.index.is_monotonic_increasing
    assert test_frame.index.is_monotonic_increasing
