"""Visualization utilities for exploratory analysis and reporting.

This module will house common plotting routines (scatter plots, decision boundaries,
residual diagnostics) that task modules can compose.

Todo:
    * Implement reusable figure factories and styling conventions.
    * Add helpers for saving plots into docs/results directories.
    * Integrate logging hooks for experiment traceability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

FIGURES_ROOT = Path(__file__).resolve().parents[3] / "docs" / "results" / "figures"


def save_fig(fig: Figure, path: str | Path) -> Path:
    """Persist a figure under ``docs/results/figures`` and return the file path.

    Args:
        fig: Matplotlib figure to serialise.
        path: Destination path relative to the figures directory or an absolute path.

    Returns:
        Path: Absolute path to the saved figure on disk.
    """
    target = Path(path)
    if not target.is_absolute():
        target = FIGURES_ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight")
    return target


def scatter_plot(
    x: Sequence[float],
    y: Sequence[float],
    *,
    hue: Iterable[int | float] | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend: bool = True,
    cmap: str = "viridis",
) -> Figure:
    """Create a scatter plot optionally coloured by class labels.

    Args:
        x: Iterable of x-axis values.
        y: Iterable of y-axis values.
        hue: Optional class labels used for colouring points.
        title: Plot title shown above the axes.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        legend: Whether to render a legend when labels are supplied.
        cmap: Matplotlib colormap name for class colouring.

    Returns:
        matplotlib.figure.Figure: Generated figure instance.
    """
    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    fig, ax = plt.subplots()
    if hue is None:
        ax.scatter(x_values, y_values, s=40, alpha=0.8)
    else:
        labels = np.asarray(list(hue))
        classes = np.unique(labels)
        colormap = plt.get_cmap(cmap, len(classes))
        for index, klass in enumerate(classes):
            mask = labels == klass
            ax.scatter(
                x_values[mask],
                y_values[mask],
                s=40,
                alpha=0.8,
                label=str(klass),
                color=colormap(index),
            )
        if legend:
            ax.legend(title="Class")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig


def residual_plot(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    title: str = "",
    xlabel: str = "Predicted",
    ylabel: str = "Residual",
) -> Figure:
    """Create a residual scatter plot for regression diagnostics.

    Args:
        y_true: Ground-truth numeric targets.
        y_pred: Predicted numeric targets.
        title: Plot title shown above the axes.
        xlabel: Label for the x-axis (predictions).
        ylabel: Label for the y-axis (residuals).

    Returns:
        matplotlib.figure.Figure: Generated figure instance.
    """
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    residuals = true - pred

    fig, ax = plt.subplots()
    ax.scatter(pred, residuals, s=40, alpha=0.8)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig


__all__ = ["FIGURES_ROOT", "residual_plot", "save_fig", "scatter_plot"]
