"""Dataset management helpers for the machine learning toolkit.

This module will expose dataset download, validation, and train/test splitting utilities
common to every task in the project.

Todo:
    * Resolve canonical dataset paths and download sources.
    * Implement reusable loaders for Iris and Auto MPG datasets.
    * Provide stratified and random splitting helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


def _require_data_root() -> Path:
    """Ensure the canonical data directory exists before attempting reads.

    Returns:
        Path: Absolute path to the data directory.

    Raises:
        FileNotFoundError: If the expected data directory cannot be located.
    """
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Expected dataset root at {DATA_ROOT!s}, but the directory does not exist."
        )
    return DATA_ROOT


def _load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file from the project data directory.

    Args:
        filename: Name of the CSV file relative to the ``data`` directory.

    Returns:
        pandas.DataFrame: Loaded dataset with default dtype inference.

    Raises:
        FileNotFoundError: If the target CSV is missing.
    """
    root = _require_data_root()
    path = root / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file {filename} was not found under {root!s}."
        )
    return pd.read_csv(path)


def load_iris_dataset() -> pd.DataFrame:
    """Return the canonical Iris dataset packed with the repository.

    Returns:
        pandas.DataFrame: Iris measurements and target labels.

    Raises:
        FileNotFoundError: If ``data/iris.csv`` cannot be located.
    """
    return _load_csv("iris.csv")


def load_auto_mpg_dataset() -> pd.DataFrame:
    """Return the Auto MPG dataset used by the regression track.

    Returns:
        pandas.DataFrame: Auto MPG features and mileage target.

    Raises:
        FileNotFoundError: If ``data/auto_mpg.csv`` cannot be located.
    """
    return _load_csv("auto_mpg.csv")


def train_test_split_frame(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int | None = None,
    stratify: Sequence[Any] | pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a :class:`~pandas.DataFrame` into train and test partitions.

    Args:
        frame: Source dataframe containing all samples.
        test_size: Fraction of data reserved for the test split.
        random_state: Deterministic random seed passed to scikit-learn.
        stratify: Optional labels used to preserve class balance.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: Train and test subsets with
        index reset for downstream ergonomics.

    Raises:
        ValueError: If ``test_size`` is not within the open interval (0, 1).
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    train_frame, test_frame = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


__all__ = [
    "DATA_ROOT",
    "load_auto_mpg_dataset",
    "load_iris_dataset",
    "train_test_split_frame",
]
