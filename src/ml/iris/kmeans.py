"""K-means clustering workflows for Iris exploratory analysis.

The helpers in this module streamline feature preparation, clustering, evaluation,
and plotting so that experimentation code can focus on interpreting results rather
than orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from ml.core.datasets import load_iris_dataset
from ml.core.features import build_scaler
from ml.core.visualization import FIGURES_ROOT, save_fig
from ml.iris.plotting import (
    IRIS_TARGET_TO_SPECIES,
    build_species_style_map,
    scatter_feature_pair,
)

CLUSTER_MARKERS = ["o", "s", "^", "D", "P", "X"]


def _slugify_feature(name: str) -> str:
    """Create a filesystem-safe slug from a feature name."""

    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(".", "_")
    )


def _ensure_directory(path: Path) -> Path:
    """Ensure the parent directory exists for the given path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class KMeansConfig:
    """Configuration describing a single Iris K-means experiment.

    Attributes:
        features: Columns to include during clustering.
        n_clusters: Number of clusters for scikit-learn ``KMeans``.
        scaler: Scaling strategy name compatible with :func:`build_scaler`, or
            ``None`` to disable scaling.
        random_state: Deterministic seed for centroid initialisation.
        n_init: Number of centroid initialisation runs.
    """

    features: Sequence[str]
    n_clusters: int
    scaler: str | None = "standard"
    random_state: int = 42
    n_init: int = 10
    _scaler: Any = field(init=False, default=None, repr=False)
    _scaled_features: pd.DataFrame | None = field(init=False, default=None, repr=False)
    _assignments: np.ndarray | None = field(init=False, default=None, repr=False)


def run_clustering(config: KMeansConfig) -> tuple[pd.DataFrame, KMeans]:
    """Fit a K-means model and return assignments alongside the estimator.

    Args:
        config: Experiment configuration describing the feature subset and
            hyperparameters to apply.

    Returns:
        tuple[pandas.DataFrame, sklearn.cluster.KMeans]: A dataframe containing the
        original feature values, numeric targets, and cluster assignments, together
        with the fitted KMeans estimator.

    Raises:
        KeyError: If requested feature columns are missing.
    """

    frame = load_iris_dataset()
    feature_list = list(config.features)
    missing = [feature for feature in feature_list if feature not in frame.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing feature columns: {missing_str}.")

    features_original = frame.loc[:, feature_list].copy()
    if config.scaler:
        scaler = build_scaler(config.scaler)
        features_scaled = pd.DataFrame(
            scaler.fit_transform(features_original),
            columns=feature_list,
        )
    else:
        scaler = None
        features_scaled = features_original.copy()

    model = KMeans(
        n_clusters=config.n_clusters,
        random_state=config.random_state,
        n_init=config.n_init,
    )
    assignments = model.fit_predict(features_scaled)

    output_frame = features_original.copy()
    output_frame["cluster"] = assignments
    output_frame["target"] = frame["target"].astype(int)

    config._scaler = scaler
    config._scaled_features = features_scaled
    config._assignments = assignments

    return output_frame, model


def evaluate_clustering(
    model: KMeans,
    feature_matrix: pd.DataFrame,
    labels: Iterable[int] | None,
) -> dict[str, float]:
    """Evaluate clustering quality using inertia, silhouette, and ARI metrics.

    Args:
        model: Fitted KMeans estimator.
        feature_matrix: Feature matrix (typically scaled) used for fitting.
        labels: Optional iterable of ground-truth labels for adjusted Rand score.

    Returns:
        dict[str, float]: Dictionary containing ``inertia``, ``silhouette_score``,
        and ``adjusted_rand_score`` keys.
    """

    metrics: dict[str, float] = {"inertia": float(model.inertia_)}

    if feature_matrix.shape[1] >= 2 and len(np.unique(model.labels_)) > 1:
        silhouette = float(silhouette_score(feature_matrix, model.labels_))
    else:
        silhouette = float("nan")
    metrics["silhouette_score"] = silhouette

    if labels is None:
        metrics["adjusted_rand_score"] = float("nan")
    else:
        metrics["adjusted_rand_score"] = float(
            adjusted_rand_score(list(labels), model.labels_)
        )
    return metrics


def plot_clusters(
    frame: pd.DataFrame,
    config: KMeansConfig,
    assignments: Sequence[int],
    model: KMeans,
    style_map: dict[str, dict[str, Any]],
    ticket_id: str,
) -> Path:
    """Plot clustering assignments overlaid on the Iris feature plane.

    Args:
        frame: Dataframe containing the original feature values and ``target`` column.
        config: Experiment configuration being visualised.
        assignments: Cluster assignments aligned with ``frame`` rows.
        model: Fitted KMeans estimator (used to plot centroids).
        style_map: Mapping of species labels to marker/colour styles.
        ticket_id: Ticket identifier used when deriving the output filename.

    Returns:
        Path: Path to the persisted figure.

    Raises:
        ValueError: If ``config.features`` does not contain exactly two columns.
    """

    if len(config.features) != 2:
        raise ValueError("Cluster plotting requires exactly two features.")

    feature_x, feature_y = list(config.features)
    plot_frame = frame.loc[:, [feature_x, feature_y]].copy()
    plot_frame["target"] = frame["target"].astype(int)

    figure = scatter_feature_pair(
        frame=plot_frame,
        x=feature_x,
        y=feature_y,
        style_map=style_map,
        title=f"KMeans Clusters (k={config.n_clusters})",
    )
    ax = figure.axes[0]

    cluster_array = np.asarray(assignments)
    unique_clusters = np.unique(cluster_array)
    for cluster_id in unique_clusters:
        mask = cluster_array == cluster_id
        marker = CLUSTER_MARKERS[cluster_id % len(CLUSTER_MARKERS)]
        ax.scatter(
            plot_frame.loc[mask, feature_x],
            plot_frame.loc[mask, feature_y],
            marker=marker,
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            s=70,
            label=f"cluster {cluster_id}",
        )

    centroids = model.cluster_centers_
    if config._scaler is not None:
        centroids = config._scaler.inverse_transform(centroids)

    cmap = ListedColormap([style_map[name]["color"] for name in style_map])
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        c="black",
        s=120,
        label="centroid",
        edgecolors="white",
        linewidths=1.5,
    )

    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate legend entries while preserving order.
    seen: set[str] = set()
    filtered_handles = []
    filtered_labels = []
    for handle, label in zip(handles, labels, strict=False):
        if label in seen:
            continue
        seen.add(label)
        filtered_handles.append(handle)
        filtered_labels.append(label)
    ax.legend(filtered_handles, filtered_labels, title="Legend", loc="best")
    figure.tight_layout()

    filename = (
        f"{ticket_id}_kmeans_k{config.n_clusters}_{_slugify_feature(feature_x)}_"
        f"{_slugify_feature(feature_y)}.png"
    )
    target_path = _ensure_directory(FIGURES_ROOT / filename)
    saved_path = save_fig(figure, target_path)
    plt.close(figure)
    return saved_path


def compare_feature_pairs(
    configs: Sequence[KMeansConfig],
    ticket_id: str,
) -> list[dict[str, Any]]:
    """Run multiple K-means configurations and aggregate metrics and artefacts.

    Args:
        configs: Iterable of KMeans configurations to evaluate.
        ticket_id: Ticket identifier used when naming generated artefacts.

    Returns:
        list[dict[str, Any]]: Experiment summaries containing configuration,
        model, metrics, and optional figure paths.
    """

    frame = load_iris_dataset()
    species = [IRIS_TARGET_TO_SPECIES[int(target)] for target in frame["target"]]
    style_map = build_species_style_map(species)

    results: list[dict[str, Any]] = []
    for config in configs:
        assignments_frame, model = run_clustering(config)
        metrics = evaluate_clustering(
            model,
            config._scaled_features if config._scaled_features is not None else assignments_frame.loc[:, list(config.features)],
            assignments_frame["target"],
        )

        figure_path: Path | None = None
        if len(config.features) == 2:
            figure_path = plot_clusters(
                assignments_frame,
                config,
                assignments_frame["cluster"],
                model,
                style_map,
                ticket_id,
            )

        results.append(
            {
                "config": config,
                "model": model,
                "assignments": assignments_frame,
                "metrics": metrics,
                "figure_path": figure_path,
            }
        )

    return results


__all__ = [
    "CLUSTER_MARKERS",
    "KMeansConfig",
    "compare_feature_pairs",
    "evaluate_clustering",
    "plot_clusters",
    "run_clustering",
]
