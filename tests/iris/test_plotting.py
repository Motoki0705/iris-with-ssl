"""Tests for Iris-specific plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ml.iris.plotting import (
    IRIS_TARGET_TO_SPECIES,
    SPECIES_STYLE_BANK,
    build_species_style_map,
    plot_feature_grid,
    scatter_feature_pair,
)


def _close(fig: Figure) -> None:
    """Release matplotlib resources after assertions."""

    plt.close(fig)


def test_build_species_style_map_rotates_styles() -> None:
    """Species style map should keep keys unique and cover five style combos."""

    species = [f"species_{index}" for index in range(6)]
    style_map = build_species_style_map(species)

    assert list(style_map.keys()) == species
    combinations = {(style["color"], style["marker"]) for style in style_map.values()}
    assert len(combinations) >= 5

    bank_combinations = {
        (style["color"], style["marker"]) for style in SPECIES_STYLE_BANK[:5]
    }
    assert len(bank_combinations) == 5


def test_scatter_feature_pair_saves_fig(
    iris_frame, tmp_results_dir: Path  # type: ignore[arg-type]
) -> None:
    """Scatter helper should return a Figure and persist to disk when requested."""

    species = [IRIS_TARGET_TO_SPECIES[target] for target in iris_frame["target"]]
    style_map = build_species_style_map(species)
    target_path = tmp_results_dir / "scatter_iris.png"

    fig = scatter_feature_pair(
        frame=iris_frame,
        x="sepal.length (cm)",
        y="sepal.width (cm)",
        style_map=style_map,
        title="Sepal Length vs Width",
        save_path=target_path,
    )
    try:
        assert isinstance(fig, Figure)
        assert target_path.exists()
        assert getattr(fig, "_saved_path") == target_path
    finally:
        _close(fig)


def test_plot_feature_grid_writes_expected_files(
    iris_frame, tmp_results_dir: Path  # type: ignore[arg-type]
) -> None:
    """Grid helper should return absolute paths for each saved figure."""

    species = [IRIS_TARGET_TO_SPECIES[target] for target in iris_frame["target"]]
    style_map = build_species_style_map(species)
    pairs = [
        ("sepal.length (cm)", "sepal.width (cm)"),
        ("petal.length (cm)", "petal.width (cm)"),
    ]

    saved = plot_feature_grid(
        frame=iris_frame,
        feature_pairs=pairs,
        style_map=style_map,
        ticket_id="I-01",
        output_dir=tmp_results_dir,
    )

    assert len(saved) == len(pairs)
    for path, (x, y) in zip(saved, pairs, strict=True):
        assert path.is_absolute()
        assert path.exists()
        expected_name = f"I-01_{x}_{y}".replace(" ", "_") + ".png"
        assert path.name == expected_name
