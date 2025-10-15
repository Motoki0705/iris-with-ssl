"""Model comparison utilities for Auto MPG regression experiments.

Ticket M-03 aggregates evaluation outputs from tickets M-01 and M-02 to
generate ranked tables and Markdown reports that support the No4 discussion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_TOP_N = 3


def _stringify_features(features: Any) -> str:
    """Convert a feature specification into a readable string."""
    if isinstance(features, str):
        return features
    if isinstance(features, Sequence):
        return ", ".join(str(item) for item in features)
    return str(features)


def _stringify_parameters(parameters: Any) -> str:
    """Convert parameter dictionaries into a deterministic string."""
    if parameters is None:
        return ""
    if isinstance(parameters, Mapping):
        items = sorted(parameters.items())
        return ", ".join(f"{key}={value}" for key, value in items)
    return str(parameters)


def _prepare_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise a single result dictionary for DataFrame construction."""
    model_type = record.get("model_type", "unknown")
    parameters = record.get("parameters")
    degree = None
    if isinstance(parameters, Mapping):
        degree = parameters.get("degree")
    return {
        "model_type": model_type,
        "name": record.get("name"),
        "features": _stringify_features(record.get("features")),
        "target": record.get("target", "mpg"),
        "k": record.get("k"),
        "degree": record.get("degree", degree),
        "parameters": _stringify_parameters(parameters),
        "mse": record.get("mse"),
        "mae": record.get("mae"),
        "r2": record.get("r2"),
        "train_samples": record.get("train_samples"),
        "test_samples": record.get("test_samples"),
    }


def compile_regression_results(
    knn_results: Sequence[Mapping[str, Any]],
    linear_results: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Combine KNN and linear-model evaluation results into a single table.

    Args:
        knn_results: Sequence of evaluation dictionaries produced by
            :func:`ml.auto_mpg.knn_regressor.evaluate_knn_models`.
        linear_results: Sequence of evaluation dictionaries produced by
            :func:`ml.auto_mpg.linear_models.evaluate_linear_models`.

    Returns:
        pandas.DataFrame: Tidy table containing metrics and metadata for all
        provided results. Columns are standardised to support ranking and
        reporting tasks.
    """
    records = [
        _prepare_record(record) for record in (*knn_results, *linear_results)
    ]
    columns = [
        "model_type",
        "name",
        "features",
        "target",
        "k",
        "degree",
        "parameters",
        "mse",
        "mae",
        "r2",
        "train_samples",
        "test_samples",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(records, columns=columns)
    return frame.sort_values(
        by=["model_type", "name"], kind="stable"
    ).reset_index(drop=True)


def rank_models(results: pd.DataFrame, top_n: int = DEFAULT_TOP_N) -> dict[str, pd.DataFrame]:
    """Identify top-performing models under different ranking schemes.

    Args:
        results: DataFrame produced by :func:`compile_regression_results`.
        top_n: Number of models to surface for each ranking.

    Returns:
        dict[str, pandas.DataFrame]: Mapping with ``\"mse\"`` and ``\"r2\"``
        keys containing top-k tables ranked by error (ascending) and R²
        (descending) respectively.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if results.empty:
        raise ValueError("results dataframe is empty; cannot compute rankings.")
    if top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    mse_rank = results.nsmallest(top_n, "mse").reset_index(drop=True)
    r2_rank = results.nlargest(top_n, "r2").reset_index(drop=True)
    return {
        "mse": mse_rank,
        "r2": r2_rank,
    }


def _format_table_for_markdown(frame: pd.DataFrame) -> str:
    """Render a dataframe into Markdown while ensuring readability."""
    if frame.empty:
        return "_No models to display._"
    display_frame = frame.copy()
    for column in ("features", "parameters"):
        if column in display_frame.columns:
            display_frame[column] = display_frame[column].astype(str)
    try:
        return display_frame.to_markdown(index=False)
    except ImportError:
        # Fall back to a plain-text table when tabulate is unavailable.
        return "```\n" + display_frame.to_string(index=False) + "\n```"


def generate_discussion_report(
    results: pd.DataFrame,
    output_path: Path | str,
    top_n: int = DEFAULT_TOP_N,
) -> Path:
    """Write a Markdown report summarising Auto MPG regression experiments.

    Args:
        results: Aggregated metrics from :func:`compile_regression_results`.
        output_path: Destination path (file) for the generated Markdown.
        top_n: Number of top models to include in each ranking section.

    Returns:
        Path: Absolute path to the Markdown report.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if results.empty:
        raise ValueError("Cannot generate a report from an empty dataframe.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rankings = rank_models(results, top_n=top_n)
    mse_table = _format_table_for_markdown(rankings["mse"])
    r2_table = _format_table_for_markdown(rankings["r2"])

    summary_lines = [
        f"# Auto MPG Regression Comparison — Top {top_n}",
        "",
        "This report summarises the strongest-performing models across KNN, "
        "linear, and polynomial regressors evaluated in tickets M-01 and M-02.",
        "",
        "## Best Models by Mean Squared Error",
        "",
        mse_table,
        "",
        "## Best Models by R²",
        "",
        r2_table,
        "",
        "## Notes",
        "",
        "- Metrics computed on hold-out splits with consistent random seeds.",
        "- Feature lists and parameters are provided for reproducibility.",
    ]
    output.write_text("\n".join(summary_lines), encoding="utf-8")
    return output.resolve()


__all__ = [
    "compile_regression_results",
    "generate_discussion_report",
    "rank_models",
]
