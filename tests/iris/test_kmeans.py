"""Tests for Iris K-means clustering utilities."""

from __future__ import annotations

import math

import numpy as np

from ml.core.datasets import load_iris_dataset
from ml.iris.kmeans import (
    KMeansConfig,
    evaluate_clustering,
    plot_clusters,
    run_clustering,
)
from ml.iris.plotting import IRIS_TARGET_TO_SPECIES, build_species_style_map


def test_run_clustering_generates_expected_cluster_count() -> None:
    """KMeans should allocate the configured number of clusters."""

    config = KMeansConfig(
        features=["sepal.length (cm)", "sepal.width (cm)"],
        n_clusters=3,
        scaler="standard",
        random_state=4,
        n_init=10,
    )
    assignments_frame, model = run_clustering(config)

    assert len(assignments_frame) == 150
    assert len(np.unique(assignments_frame["cluster"])) == config.n_clusters
    assert model.n_clusters == config.n_clusters


def test_evaluate_clustering_scores_within_expected_bounds() -> None:
    """Evaluation metrics should stay within theoretical ranges."""

    config = KMeansConfig(
        features=[
            "sepal.length (cm)",
            "sepal.width (cm)",
            "petal.length (cm)",
            "petal.width (cm)",
        ],
        n_clusters=3,
        scaler="standard",
        random_state=1,
        n_init=15,
    )
    assignments_frame, model = run_clustering(config)
    metrics = evaluate_clustering(
        model,
        config._scaled_features,  # type: ignore[arg-type]
        assignments_frame["target"],
    )

    assert metrics["inertia"] > 0
    if not math.isnan(metrics["silhouette_score"]):
        assert -1.0 <= metrics["silhouette_score"] <= 1.0
    if not math.isnan(metrics["adjusted_rand_score"]):
        assert 0.0 <= metrics["adjusted_rand_score"] <= 1.0


def test_plot_clusters_creates_figure(monkeypatch, tmp_results_dir) -> None:  # type: ignore[override]
    """Plot helper should persist a figure for two-feature configurations."""

    config = KMeansConfig(
        features=["petal.length (cm)", "petal.width (cm)"],
        n_clusters=3,
        scaler="minmax",
        random_state=3,
        n_init=10,
    )
    assignments_frame, model = run_clustering(config)

    species = [
        IRIS_TARGET_TO_SPECIES[int(target)]
        for target in assignments_frame["target"].tolist()
    ]
    style_map = build_species_style_map(species)

    monkeypatch.setattr("ml.iris.kmeans.FIGURES_ROOT", tmp_results_dir, raising=False)
    path = plot_clusters(
        assignments_frame,
        config,
        assignments_frame["cluster"],
        model,
        style_map,
        ticket_id="TEST",
    )

    assert path.exists()
    assert path.parent == tmp_results_dir
