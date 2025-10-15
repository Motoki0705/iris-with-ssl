"""Experiment orchestration for Iris classification and clustering tasks.

This module ties together the reusable utilities implemented across I-01〜I-03
so that notebooks and CLI flows can trigger the full suite of experiments with
a single entry point. Outputs (metrics, figures, and trace logs) are written to
the ``docs`` hierarchy for downstream review.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from ml.core.features import IRIS_SCHEMA
from ml.core.logging import ResultArtifact, write_result_markdown, write_trace
from ml.iris.knn_classifier import KnnConfig, run_knn_experiments
from ml.iris.kmeans import KMeansConfig, compare_feature_pairs


@dataclass
class IrisExperimentReport:
    """Aggregate report summarising Iris KNN and K-means experiment runs.

    Attributes:
        ticket_id: Ticket identifier used for artefact namespacing.
        knn_results: Results list returned by :func:`run_knn_suite`.
        kmeans_results: Results list returned by :func:`run_kmeans_suite`.
        generated_artifacts: Collection of filesystem paths produced during the run.
    """

    ticket_id: str
    knn_results: list[dict[str, Any]]
    kmeans_results: list[dict[str, Any]]
    generated_artifacts: list[Path]


def build_default_knn_configs() -> list[KnnConfig]:
    """Return the canonical suite of KNN configurations for Iris experiments."""

    full_features = IRIS_SCHEMA["features"]
    sepal_pair = ["sepal.length (cm)", "sepal.width (cm)"]
    petal_pair = ["petal.length (cm)", "petal.width (cm)"]
    base_kwargs = {"scaler": "standard", "test_size": 0.2, "random_state": 42}

    return [
        KnnConfig(features=full_features, k=1, **base_kwargs),
        KnnConfig(features=full_features, k=3, **base_kwargs),
        KnnConfig(features=full_features, k=5, **base_kwargs),
        KnnConfig(features=full_features, k=10, **base_kwargs),
        KnnConfig(features=sepal_pair, k=5, **base_kwargs),
        KnnConfig(features=petal_pair, k=5, **base_kwargs),
    ]


def build_default_kmeans_configs() -> list[KMeansConfig]:
    """Return the canonical set of KMeans configurations for Iris experiments."""

    full_features = IRIS_SCHEMA["features"]
    sepal_pair = ["sepal.length (cm)", "sepal.width (cm)"]
    petal_pair = ["petal.length (cm)", "petal.width (cm)"]
    base_kwargs = {"scaler": "standard", "random_state": 42, "n_init": 20}

    return [
        KMeansConfig(features=full_features, n_clusters=3, **base_kwargs),
        KMeansConfig(features=full_features, n_clusters=4, **base_kwargs),
        KMeansConfig(features=full_features, n_clusters=5, **base_kwargs),
        KMeansConfig(features=sepal_pair, n_clusters=3, scaler="standard", random_state=42, n_init=20),
        KMeansConfig(features=petal_pair, n_clusters=3, scaler="standard", random_state=42, n_init=20),
    ]


def run_knn_suite(
    configs: Sequence[KnnConfig], ticket_id: str
) -> list[dict[str, Any]]:
    """Execute the KNN suite and return the raw results collection."""

    return run_knn_experiments(configs, ticket_id)


def run_kmeans_suite(
    configs: Sequence[KMeansConfig], ticket_id: str
) -> list[dict[str, Any]]:
    """Execute the KMeans suite and return the raw results collection."""

    return compare_feature_pairs(configs, ticket_id)


def _aggregate_knn_metrics(results: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        config: KnnConfig = result["config"]
        metrics: dict[str, float] = result["metrics"]
        rows.append(
            {
                "features": ", ".join(config.features),
                "k": config.k,
                "scaler": config.scaler,
                "test_size": config.test_size,
                **metrics,
                "confusion_matrix_path": str(result["confusion_matrix_path"]),
                "figure_path": str(result.get("figure_path", "")),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_kmeans_metrics(results: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        config: KMeansConfig = result["config"]
        metrics: dict[str, float] = result["metrics"]
        rows.append(
            {
                "features": ", ".join(config.features),
                "n_clusters": config.n_clusters,
                "scaler": config.scaler,
                "random_state": config.random_state,
                "n_init": config.n_init,
                **metrics,
                "figure_path": str(result.get("figure_path", "")),
            }
        )
    return pd.DataFrame(rows)


def _default_tables_root() -> Path:
    from ml.core.visualization import FIGURES_ROOT

    return FIGURES_ROOT.parent / "tables"


def generate_report(
    ticket_id: str,
    output_dir: Path | None = None,
    *,
    include_knn: bool = True,
    include_kmeans: bool = True,
    knn_configs: Sequence[KnnConfig] | None = None,
    kmeans_configs: Sequence[KMeansConfig] | None = None,
) -> IrisExperimentReport:
    """Run the configured experiment suites and write summary artefacts.

    Args:
        ticket_id: Ticket identifier used in artefact filenames.
        output_dir: Optional directory for tabular summaries (defaults to
            ``docs/results/tables``).
        include_knn: Whether to execute KNN experiments.
        include_kmeans: Whether to execute KMeans experiments.
        knn_configs: Optional override for the KNN configuration list.
        kmeans_configs: Optional override for the KMeans configuration list.

    Returns:
        IrisExperimentReport: Aggregate report capturing suite outcomes.
    """

    if not include_knn and not include_kmeans:
        raise ValueError("At least one of include_knn or include_kmeans must be True.")

    resolved_output_dir = output_dir or _default_tables_root()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    knn_results: list[dict[str, Any]] = []
    if include_knn:
        knn_results = run_knn_suite(
            knn_configs or build_default_knn_configs(), ticket_id
        )

    kmeans_results: list[dict[str, Any]] = []
    if include_kmeans:
        kmeans_results = run_kmeans_suite(
            kmeans_configs or build_default_kmeans_configs(), ticket_id
        )

    artifacts: list[ResultArtifact] = []
    artifact_paths: list[Path] = []

    summary_lines: list[str] = []
    if knn_results:
        knn_table = _aggregate_knn_metrics(knn_results)
        knn_table_path = resolved_output_dir / f"{ticket_id}_knn_summary.csv"
        knn_table.to_csv(knn_table_path, index=False)
        artifacts.append(
            ResultArtifact(
                ticket_id=ticket_id,
                artifact_path=knn_table_path,
                description="KNN configuration performance summary.",
            )
        )
        artifact_paths.append(knn_table_path)
        best_knn = max(knn_results, key=lambda item: item["metrics"]["accuracy"])
        summary_lines.append(
            "KNN best accuracy "
            f"{best_knn['metrics']['accuracy']:.3f} achieved with "
            f"k={best_knn['config'].k} over features "
            f"{', '.join(best_knn['config'].features)}."
        )
        for entry in knn_results:
            confusion_path = Path(entry["confusion_matrix_path"])
            artifacts.append(
                ResultArtifact(
                    ticket_id=ticket_id,
                    artifact_path=confusion_path,
                    description=f"KNN confusion matrix (k={entry['config'].k}).",
                )
            )
            artifact_paths.append(confusion_path)
            if entry.get("figure_path"):
                fig_path = Path(entry["figure_path"])
                artifacts.append(
                    ResultArtifact(
                        ticket_id=ticket_id,
                        artifact_path=fig_path,
                        description=f"KNN decision boundary (k={entry['config'].k}).",
                    )
                )
                artifact_paths.append(fig_path)

    if kmeans_results:
        kmeans_table = _aggregate_kmeans_metrics(kmeans_results)
        kmeans_table_path = resolved_output_dir / f"{ticket_id}_kmeans_summary.csv"
        kmeans_table.to_csv(kmeans_table_path, index=False)
        artifacts.append(
            ResultArtifact(
                ticket_id=ticket_id,
                artifact_path=kmeans_table_path,
                description="KMeans configuration performance summary.",
            )
        )
        artifact_paths.append(kmeans_table_path)
        best_kmeans = max(
            kmeans_results,
            key=lambda item: item["metrics"]["adjusted_rand_score"]
            if not pd.isna(item["metrics"]["adjusted_rand_score"])
            else -1,
        )
        summary_lines.append(
            "KMeans best adjusted Rand score "
            f"{best_kmeans['metrics']['adjusted_rand_score']:.3f} with "
            f"k={best_kmeans['config'].n_clusters} over features "
            f"{', '.join(best_kmeans['config'].features)}."
        )
        for entry in kmeans_results:
            if entry.get("figure_path"):
                fig_path = Path(entry["figure_path"])
                artifacts.append(
                    ResultArtifact(
                        ticket_id=ticket_id,
                        artifact_path=fig_path,
                        description=f"KMeans clusters (k={entry['config'].n_clusters}).",
                    )
                )
                artifact_paths.append(fig_path)

    summary_text = "\n".join(summary_lines) if summary_lines else "No experiments executed."
    write_result_markdown(ticket_id, summary_text, artifacts)

    trace_lines: list[str] = []
    if knn_results:
        trace_lines.append("### KNN configurations")
        for entry in knn_results:
            config: KnnConfig = entry["config"]
            trace_lines.append(
                f"- k={config.k}, features={', '.join(config.features)}, "
                f"accuracy={entry['metrics']['accuracy']:.3f}"
            )
    if kmeans_results:
        trace_lines.append("")
        trace_lines.append("### KMeans configurations")
        for entry in kmeans_results:
            config: KMeansConfig = entry["config"]
            trace_lines.append(
                f"- k={config.n_clusters}, features={', '.join(config.features)}, "
                f"ARI={entry['metrics']['adjusted_rand_score']:.3f}"
            )
    write_trace(ticket_id, "\n".join(trace_lines))

    # Deduplicate artefact paths while preserving order.
    seen_paths: set[Path] = set()
    unique_artifacts: list[Path] = []
    for path in artifact_paths:
        resolved = Path(path)
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        unique_artifacts.append(resolved)

    return IrisExperimentReport(
        ticket_id=ticket_id,
        knn_results=list(knn_results),
        kmeans_results=list(kmeans_results),
        generated_artifacts=unique_artifacts,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Iris KNN/KMeans experiment suites and persist artefacts."
    )
    parser.add_argument(
        "--ticket-id",
        default="I-04",
        help="Ticket identifier used to namespace generated artefacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for summary tables (defaults to docs/results/tables).",
    )
    parser.add_argument(
        "--knn-only",
        action="store_true",
        help="Run only the KNN experiment suite.",
    )
    parser.add_argument(
        "--kmeans-only",
        action="store_true",
        help="Run only the KMeans experiment suite.",
    )
    return parser


def main(args: argparse.Namespace | None = None) -> IrisExperimentReport:
    """CLI entry point compatible with ``python -m ml.iris.experiments``."""

    parser = _build_parser()
    parsed = parser.parse_args(args=args)

    include_knn = not parsed.kmeans_only
    include_kmeans = not parsed.knn_only
    report = generate_report(
        ticket_id=parsed.ticket_id,
        output_dir=parsed.output_dir,
        include_knn=include_knn,
        include_kmeans=include_kmeans,
    )

    print(
        f"[Iris Experiments] ticket={parsed.ticket_id} "
        f"knn_runs={len(report.knn_results)} "
        f"kmeans_runs={len(report.kmeans_results)} "
        f"artifacts={len(report.generated_artifacts)}"
    )
    return report


if __name__ == "__main__":
    main()
