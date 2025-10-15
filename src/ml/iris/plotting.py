"""Plotting utilities tailored to the Iris dataset experiments.

This module provides reusable scatter-plot helpers that apply consistent styling
across notebooks and training scripts. The API builds on the shared visualization
utilities and focuses on Iris-specific needs such as mapping species to marker /
colour combinations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from ml.core.visualization import FIGURES_ROOT, save_fig

IRIS_TARGET_TO_SPECIES: dict[int, str] = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}
"""Mapping from numeric target codes to human-readable species names."""

SPECIES_STYLE_BANK: list[dict[str, Any]] = [
    {"color": "#1f77b4", "marker": "o", "s": 60, "alpha": 0.85, "edgecolor": "black"},
    {"color": "#ff7f0e", "marker": "s", "s": 70, "alpha": 0.8, "edgecolor": "black"},
    {"color": "#2ca02c", "marker": "^", "s": 65, "alpha": 0.8, "edgecolor": "black"},
    {"color": "#d62728", "marker": "D", "s": 72, "alpha": 0.78, "edgecolor": "black"},
    {"color": "#9467bd", "marker": "P", "s": 68, "alpha": 0.82, "edgecolor": "black"},
    {"color": "#8c564b", "marker": "X", "s": 75, "alpha": 0.85, "edgecolor": "black"},
]
"""Default palette of marker styles cycled across Iris species."""


def build_species_style_map(
    species: Sequence[str],
    style_bank: Sequence[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Construct a species-to-style mapping with cyclic assignment.

    Args:
        species: Ordered sequence of species labels used to determine mapping order.
        style_bank: Optional override for the available marker styles. Must contain
            at least five entries to satisfy visual diversity requirements.

    Returns:
        dict[str, dict[str, Any]]: Mapping from each unique species label to a style
        dictionary compatible with :func:`matplotlib.axes.Axes.scatter`.

    Raises:
        ValueError: If the species sequence is empty or the style bank is undersized.
    """

    unique_labels: list[str] = []
    seen: set[str] = set()
    for label in species:
        if label not in seen:
            unique_labels.append(label)
            seen.add(label)

    if not unique_labels:
        raise ValueError("species must contain at least one label.")

    bank = list(style_bank if style_bank is not None else SPECIES_STYLE_BANK)
    if len(bank) < 5:
        raise ValueError("style_bank must provide at least five style definitions.")

    style_map: dict[str, dict[str, Any]] = {}
    for index, label in enumerate(unique_labels):
        style_map[label] = dict(bank[index % len(bank)])
    return style_map


def _resolve_species_labels(frame: pd.DataFrame) -> pd.Series:
    """Extract or derive species labels from the Iris dataframe."""

    if "species" in frame.columns:
        return frame["species"].astype(str)
    if "target" in frame.columns:
        raw = frame["target"]
        mapped = raw.map(IRIS_TARGET_TO_SPECIES)
        if mapped.isna().any():
            return raw.astype(str)
        return mapped
    raise KeyError("Frame must include either a 'species' or 'target' column.")


def scatter_feature_pair(
    frame: pd.DataFrame,
    x: str,
    y: str,
    style_map: Mapping[str, dict[str, Any]],
    title: str,
    save_path: Path | str | None = None,
) -> Figure:
    """Render a scatter plot for a pair of Iris features grouped by species.

    Args:
        frame: Source dataframe containing the feature columns and ``species`` labels.
        x: Column name for the x-axis.
        y: Column name for the y-axis.
        style_map: Mapping from species labels to style dictionaries that include
            marker configuration accepted by :meth:`matplotlib.axes.Axes.scatter`.
        title: Plot title rendered above the axes.
        save_path: Optional path (relative to ``docs/results/figures`` or absolute)
            used to persist the figure via :func:`ml.core.visualization.save_fig`.

    Returns:
        matplotlib.figure.Figure: Generated scatter plot figure.

    Raises:
        KeyError: If required columns or species labels are missing.
    """

    for column in (x, y):
        if column not in frame.columns:
            raise KeyError(f"Column '{column}' is required but not present.")

    labels = _resolve_species_labels(frame)
    working = frame.assign(__iris_species=labels)

    fig, ax = plt.subplots()
    try:
        for species, group in working.groupby("__iris_species", sort=False):
            if species not in style_map:
                raise KeyError(
                    f"Species '{species}' missing from provided style_map keys."
                )
            style = style_map[species]
            marker_style = {
                "marker": style.get("marker", "o"),
                "s": style.get("s", style.get("size", 60)),
                "alpha": style.get("alpha", 0.85),
                "edgecolor": style.get("edgecolor", "black"),
                "linewidths": style.get("linewidths", 0.5),
            }
            color = style.get("color")
            label = style.get("label", species)

            ax.scatter(
                group[x].to_numpy(),
                group[y].to_numpy(),
                label=label,
                c=color,
                **marker_style,
            )

        ax.set_title(title)
        ax.set_xlabel(x.replace("_", " ").title())
        ax.set_ylabel(y.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        ax.legend(title="Species")
        fig.tight_layout()

        if save_path is not None:
            saved_path = save_fig(fig, save_path)
            saved_path = Path(saved_path)
            setattr(fig, "_saved_path", saved_path)
    except Exception:
        plt.close(fig)
        raise

    return fig


def plot_feature_grid(
    frame: pd.DataFrame,
    feature_pairs: Sequence[tuple[str, str]],
    style_map: Mapping[str, dict[str, Any]],
    ticket_id: str,
    output_dir: Path | None = None,
) -> list[Path]:
    """Generate and save scatter plots for a collection of feature pairs.

    Args:
        frame: Dataframe containing the feature columns and ``species`` labels.
        feature_pairs: Iterable of feature column pairs (x, y) to visualise.
        style_map: Mapping of species labels to marker styles.
        ticket_id: Prefix used when naming generated figure files.
        output_dir: Optional relative directory (under ``docs/results/figures``)
            or absolute path where figures should be written.

    Returns:
        list[pathlib.Path]: Absolute file paths to the saved figures.
    """

    saved_paths: list[Path] = []
    for x, y in feature_pairs:
        filename = f"{ticket_id}_{x}_{y}".replace(" ", "_") + ".png"
        if output_dir is not None:
            target: Path | str = Path(output_dir) / filename
        else:
            target = Path(filename)

        fig = scatter_feature_pair(
            frame=frame,
            x=x,
            y=y,
            style_map=style_map,
            title=f"{x.replace('_', ' ').title()} vs {y.replace('_', ' ').title()}",
            save_path=target,
        )
        try:
            saved_path = getattr(fig, "_saved_path", None)
            if saved_path is None:
                resolved = Path(target)
                if not resolved.is_absolute():
                    resolved = FIGURES_ROOT / resolved
                saved_path = resolved
            saved_paths.append(Path(saved_path))
        finally:
            plt.close(fig)

    return saved_paths


__all__ = [
    "IRIS_TARGET_TO_SPECIES",
    "SPECIES_STYLE_BANK",
    "build_species_style_map",
    "plot_feature_grid",
    "scatter_feature_pair",
]
