"""Pytest fixtures shared across F-04 smoke tests."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pandas as pd
import pytest

from ml.core.datasets import load_auto_mpg_dataset, load_iris_dataset


@pytest.fixture(scope="session")
def iris_frame() -> pd.DataFrame:
    """Load the Iris dataset via the core datasets API.

    Returns:
        pd.DataFrame: Canonical Iris measurements including class labels.
    """

    return load_iris_dataset()


@pytest.fixture(scope="session")
def auto_mpg_frame() -> pd.DataFrame:
    """Load the Auto MPG dataset via the core datasets API.

    Returns:
        pd.DataFrame: Auto MPG records with target and feature columns.
    """

    return load_auto_mpg_dataset()


@pytest.fixture()
def tmp_results_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Create a temporary results directory under docs/results/tmp/F-04.

    Args:
        tmp_path_factory: Pytest factory for generating unique temporary paths.

    Yields:
        Path: Unique directory for persisting artefacts during a test.
    """

    base_dir = Path.cwd() / "docs" / "results" / "tmp" / "F-04"
    base_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = base_dir / f"run-{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        try:
            if base_dir.exists() and not any(base_dir.iterdir()):
                base_dir.rmdir()
        except OSError:
            pass
