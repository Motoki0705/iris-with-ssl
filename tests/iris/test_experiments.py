"""Tests for the Iris experiment orchestration module."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest

from ml.iris.experiments import (
    IrisExperimentReport,
    build_default_kmeans_configs,
    build_default_knn_configs,
    generate_report,
    main,
)
from ml.iris.knn_classifier import KnnConfig
from ml.iris.kmeans import KMeansConfig


def test_build_default_configs_have_expected_length() -> None:
    """Default configuration factories should return at least four configs."""

    knn_configs = build_default_knn_configs()
    kmeans_configs = build_default_kmeans_configs()

    assert len(knn_configs) >= 4
    assert len(kmeans_configs) >= 4
    assert isinstance(knn_configs[0], KnnConfig)
    assert isinstance(kmeans_configs[0], KMeansConfig)


def _patch_figures_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr("ml.core.visualization.FIGURES_ROOT", root, raising=False)
    monkeypatch.setattr("ml.iris.knn_classifier.FIGURES_ROOT", root, raising=False)
    monkeypatch.setattr("ml.iris.kmeans.FIGURES_ROOT", root, raising=False)


def test_generate_report_creates_artifacts(monkeypatch, tmp_results_dir) -> None:  # type: ignore[override]
    """Report generation should return artefact paths when runs are executed."""

    _patch_figures_root(monkeypatch, tmp_results_dir)

    minimal_knn_configs: Iterable[KnnConfig] = build_default_knn_configs()[:1]
    report = generate_report(
        ticket_id="TEST-I-04",
        output_dir=tmp_results_dir,
        include_knn=True,
        include_kmeans=False,
        knn_configs=minimal_knn_configs,
    )

    assert isinstance(report, IrisExperimentReport)
    assert len(report.knn_results) == 1
    assert report.kmeans_results == []
    assert report.generated_artifacts, "Expected at least one artefact path."


def test_cli_knn_only_execution(monkeypatch, tmp_results_dir, capsys) -> None:  # type: ignore[override]
    """CLI entry point should support running only the KNN suite."""

    _patch_figures_root(monkeypatch, tmp_results_dir)
    monkeypatch.setattr(
        "ml.iris.experiments.build_default_knn_configs",
        lambda: build_default_knn_configs()[:1],
    )

    report = main(
        [
            "--ticket-id",
            "CLI-I-04",
            "--knn-only",
            "--output-dir",
            str(tmp_results_dir),
        ]
    )

    captured = capsys.readouterr()
    assert "knn_runs=1" in captured.out
    assert isinstance(report, IrisExperimentReport)
    assert len(report.knn_results) == 1
    assert report.kmeans_results == []
