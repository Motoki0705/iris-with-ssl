"""Smoke tests for visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ml.core.visualization import residual_plot, save_fig, scatter_plot


def _close_figure(fig: Figure) -> None:
    """Ensure figures are released after assertions.

    Args:
        fig: Matplotlib figure to close.
    """

    plt.close(fig)


def test_scatter_plot_creates_figure(tmp_results_dir: Path) -> None:
    """Scatter plot helper should return a Figure and support saving."""

    x_values = np.linspace(0, 1, 5)
    y_values = np.linspace(1, 2, 5)
    hue: Iterable[int] = [0, 1, 0, 1, 0]

    fig = scatter_plot(x_values, y_values, hue=hue, title="demo", xlabel="x", ylabel="y")
    try:
        assert isinstance(fig, Figure)
        target_path = tmp_results_dir / "scatter.png"
        saved_path = save_fig(fig, target_path)
        assert saved_path.is_file()
        assert saved_path == target_path
    finally:
        _close_figure(fig)


def test_residual_plot_creates_figure(tmp_results_dir: Path) -> None:
    """Residual plot helper should return a Figure and support saving."""

    y_true = np.array([10.0, 11.5, 13.0])
    y_pred = np.array([9.5, 11.0, 13.5])

    fig = residual_plot(y_true, y_pred, title="residuals")
    try:
        assert isinstance(fig, Figure)
        target_path = tmp_results_dir / "residual.png"
        saved_path = save_fig(fig, target_path)
        assert saved_path.is_file()
        assert saved_path == target_path
    finally:
        _close_figure(fig)
