from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.auto_mpg.comparison import (
    compile_regression_results,
    generate_discussion_report,
    rank_models,
)


def _mock_knn_results() -> list[dict[str, object]]:
    return [
        {
            "model_type": "knn",
            "name": "knn_k3_hp",
            "features": ("horsepower",),
            "target": "mpg",
            "k": 3,
            "mse": 3.1,
            "mae": 1.2,
            "r2": 0.78,
            "train_samples": 80,
            "test_samples": 20,
        },
        {
            "model_type": "knn",
            "name": "knn_k5_hp_wt",
            "features": ("horsepower", "weight"),
            "target": "mpg",
            "k": 5,
            "mse": 2.9,
            "mae": 1.1,
            "r2": 0.80,
            "train_samples": 80,
            "test_samples": 20,
        },
    ]


def _mock_linear_results() -> list[dict[str, object]]:
    return [
        {
            "model_type": "linear",
            "name": "linear_hp",
            "features": ("horsepower",),
            "target": "mpg",
            "parameters": {"include_bias": True},
            "mse": 3.5,
            "mae": 1.3,
            "r2": 0.75,
            "train_samples": 80,
            "test_samples": 20,
        },
        {
            "model_type": "polynomial",
            "name": "poly_hp_deg2",
            "features": ("horsepower",),
            "target": "mpg",
            "parameters": {"degree": 2, "include_bias": False},
            "mse": 2.5,
            "mae": 1.0,
            "r2": 0.84,
            "train_samples": 80,
            "test_samples": 20,
        },
    ]


def test_compile_regression_results_combines_records() -> None:
    results = compile_regression_results(_mock_knn_results(), _mock_linear_results())
    assert results.shape[0] == 4
    assert "features" in results.columns
    assert all(isinstance(item, str) for item in results["features"])


def test_rank_models_returns_top_tables() -> None:
    frame = compile_regression_results(_mock_knn_results(), _mock_linear_results())
    rankings = rank_models(frame, top_n=2)
    assert set(rankings.keys()) == {"mse", "r2"}
    assert len(rankings["mse"]) == 2
    assert len(rankings["r2"]) == 2
    assert rankings["mse"]["mse"].is_monotonic_increasing
    assert rankings["r2"]["r2"].is_monotonic_decreasing


def test_generate_discussion_report_writes_markdown(tmp_path: Path) -> None:
    frame = compile_regression_results(_mock_knn_results(), _mock_linear_results())
    output = tmp_path / "report.md"
    generated_path = generate_discussion_report(frame, output, top_n=2)
    assert generated_path.exists()
    content = generated_path.read_text(encoding="utf-8")
    assert "Auto MPG Regression Comparison" in content
    assert "Best Models by Mean Squared Error" in content
