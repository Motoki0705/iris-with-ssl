"""K-nearest neighbours classifier pipelines for Iris experiments.

The utilities in this module wrap dataset preparation, model training, evaluation,
and visualisation so that notebooks and orchestration scripts can compose reusable
workflows. Each helper leans on the shared ``ml.core`` packages to keep behaviour
consistent across tickets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from ml.core.datasets import load_iris_dataset, train_test_split_frame
from ml.core.features import build_scaler
from ml.core.metrics import ClassificationReport, evaluate_classification
from ml.core.visualization import FIGURES_ROOT, save_fig
from ml.iris.plotting import (
    IRIS_TARGET_TO_SPECIES,
    build_species_style_map,
    scatter_feature_pair,
)


def _slugify_feature(name: str) -> str:
    """Create a filesystem-friendly slug for a feature name."""

    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(".", "_")
    )


def _ensure_directory(path: Path) -> Path:
    """Create the parent directory for ``path`` if necessary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class KnnConfig:
    """Configuration describing a single KNN experiment.

    Attributes:
        features: Ordered collection of feature names extracted from the Iris frame.
        k: Number of neighbours considered by the classifier.
        scaler: Scaling strategy passed to :func:`ml.core.features.build_scaler`.
        test_size: Fraction of samples reserved for evaluation.
        random_state: Deterministic seed for dataset splitting.
    """

    features: Sequence[str]
    k: int = 5
    scaler: str = "standard"
    test_size: float = 0.2
    random_state: int = 42
    _scaler: Any = field(init=False, default=None, repr=False)
    _dataset_cache: tuple[
        pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
    ] | None = field(init=False, default=None, repr=False)


def prepare_dataset(
    config: KnnConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Prepare train/test splits for a specific KNN configuration.

    Args:
        config: Experiment configuration describing the feature subset and split
            parameters to apply.

    Returns:
        tuple[pandas.DataFrame, pandas.Series, pandas.DataFrame, pandas.Series]:
        Scaled training features, training labels, scaled test features, and
        test labels.

    Raises:
        KeyError: If any requested feature columns are absent.
    """

    frame = load_iris_dataset()
    feature_list = list(config.features)
    missing = [feature for feature in feature_list if feature not in frame.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing feature columns: {missing_str}.")

    dataset = frame.loc[:, feature_list + ["target"]].copy()
    train_frame, test_frame = train_test_split_frame(
        dataset,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=dataset["target"],
    )

    x_train = train_frame.loc[:, feature_list]
    y_train = train_frame["target"].astype(int).reset_index(drop=True)
    x_test = test_frame.loc[:, feature_list]
    y_test = test_frame["target"].astype(int).reset_index(drop=True)

    scaler = build_scaler(config.scaler)
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=feature_list,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=feature_list,
    )

    config._scaler = scaler
    config._dataset_cache = (x_train_scaled, y_train, x_test_scaled, y_test)
    return config._dataset_cache


def train_classifier(config: KnnConfig) -> KNeighborsClassifier:
    """Fit a :class:`~sklearn.neighbors.KNeighborsClassifier` for the given config.

    Args:
        config: Experiment configuration describing feature subset and hyperparameters.

    Returns:
        KNeighborsClassifier: Fitted classifier ready for inference.
    """

    if config._dataset_cache is None:
        prepare_dataset(config)
    x_train, y_train, _, _ = config._dataset_cache  # type: ignore[misc]
    model = KNeighborsClassifier(n_neighbors=config.k)
    model.fit(x_train, y_train)
    setattr(model, "_iris_config", config)
    return model


def evaluate_classifier(
    model: KNeighborsClassifier, x_test: pd.DataFrame, y_test: pd.Series
) -> ClassificationReport:
    """Compute classification metrics against a hold-out split.

    Args:
        model: Trained KNN classifier.
        x_test: Test feature matrix, typically produced by :func:`prepare_dataset`.
        y_test: Ground-truth labels aligned with ``x_test``.

    Returns:
        ClassificationReport: Aggregated accuracy, precision, recall, and F1.
    """

    predictions = model.predict(x_test)
    return evaluate_classification(y_test, predictions)


def compute_confusion_matrix(
    model: KNeighborsClassifier, x_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """Compute a confusion matrix against the Iris test labels.

    Args:
        model: Trained classifier generating predictions.
        x_test: Test feature matrix.
        y_test: Ground-truth class labels.

    Returns:
        pandas.DataFrame: Confusion matrix using species names for indices/columns.
    """

    ordered_labels = sorted(IRIS_TARGET_TO_SPECIES.keys())
    pred_labels = model.predict(x_test)
    matrix = confusion_matrix(
        y_test,
        pred_labels,
        labels=ordered_labels,
    )
    species_names = [IRIS_TARGET_TO_SPECIES[label] for label in ordered_labels]
    return pd.DataFrame(matrix, index=species_names, columns=species_names)


def plot_decision_boundary(
    model: KNeighborsClassifier,
    frame: pd.DataFrame,
    config: KnnConfig,
    style_map: Mapping[str, dict[str, Any]],
    ticket_id: str,
    grid_resolution: int = 200,
) -> Path:
    """Render and persist a decision boundary plot for a two-feature configuration.

    Args:
        model: Trained KNN classifier that supports ``predict``.
        frame: Dataframe containing the original (unscaled) feature values and ``target``.
        config: Experiment configuration describing feature subset and scaling.
        style_map: Mapping of species names to plotting styles.
        ticket_id: Identifier used when constructing the output filename.
        grid_resolution: Number of steps per axis when constructing the mesh grid.

    Returns:
        Path: Filesystem path to the generated figure.

    Raises:
        ValueError: If the configuration does not correspond to a two-feature plot.
    """

    if len(config.features) != 2:
        raise ValueError("Decision boundary plotting requires exactly two features.")

    if config._scaler is None:
        prepare_dataset(config)
    scaler = config._scaler

    feature_x, feature_y = list(config.features)
    frame_subset = frame.loc[:, [feature_x, feature_y]].copy()
    frame_subset["target"] = frame["target"].astype(int)

    figure = scatter_feature_pair(
        frame=frame_subset,
        x=feature_x,
        y=feature_y,
        style_map=style_map,
        title=f"KNN Decision Boundary (k={config.k})",
    )
    ax = figure.axes[0]

    x_min, x_max = frame_subset[feature_x].min(), frame_subset[feature_x].max()
    y_min, y_max = frame_subset[feature_y].min(), frame_subset[feature_y].max()
    padding_x = (x_max - x_min) * 0.05 or 0.5
    padding_y = (y_max - y_min) * 0.05 or 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min - padding_x, x_max + padding_x, grid_resolution),
        np.linspace(y_min - padding_y, y_max + padding_y, grid_resolution),
    )
    mesh_points = pd.DataFrame(
        np.c_[xx.ravel(), yy.ravel()], columns=[feature_x, feature_y]
    )
    if scaler is not None:
        scaled = scaler.transform(mesh_points)
        mesh_points = pd.DataFrame(scaled, columns=[feature_x, feature_y])
    predictions = model.predict(mesh_points).reshape(xx.shape)

    ordered_labels = sorted(IRIS_TARGET_TO_SPECIES.keys())
    colour_order = [
        style_map[IRIS_TARGET_TO_SPECIES[label]]["color"] for label in ordered_labels
    ]
    cmap = ListedColormap(colour_order)
    levels = np.arange(len(ordered_labels) + 1) - 0.5
    ax.contourf(xx, yy, predictions, levels=levels, alpha=0.2, cmap=cmap)

    filename = (
        f"{ticket_id}_knn_k{config.k}_{_slugify_feature(feature_x)}_"
        f"{_slugify_feature(feature_y)}.png"
    )
    target_path = _ensure_directory(FIGURES_ROOT / filename)
    saved_path = save_fig(figure, target_path)
    plt.close(figure)
    return saved_path


def run_knn_experiments(
    configs: Sequence[KnnConfig], ticket_id: str
) -> list[dict[str, Any]]:
    """Execute a batch of KNN experiments and collect artefact metadata.

    Args:
        configs: Iterable of experiment configurations to evaluate.
        ticket_id: Ticket identifier used to namespace saved artefacts.

    Returns:
        list[dict[str, Any]]: Collection of experiment outcomes including metrics,
        confusion matrices, and optional figure paths.
    """

    frame = load_iris_dataset()
    species = [IRIS_TARGET_TO_SPECIES[int(target)] for target in frame["target"]]
    style_map = build_species_style_map(species)

    results: list[dict[str, Any]] = []
    tables_root = Path(FIGURES_ROOT).parent / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    for config in configs:
        x_train, y_train, x_test, y_test = prepare_dataset(config)
        model = train_classifier(config)
        metrics = evaluate_classifier(model, x_test, y_test)
        confusion = compute_confusion_matrix(model, x_test, y_test)

        feature_descriptor = "_".join(_slugify_feature(name) for name in config.features)
        confusion_filename = (
            f"{ticket_id}_knn_confusion_k{config.k}_{feature_descriptor or 'full'}.csv"
        )
        confusion_path = _ensure_directory(tables_root / confusion_filename)
        confusion.to_csv(confusion_path)

        figure_path: Path | None = None
        if len(config.features) == 2:
            figure_path = plot_decision_boundary(
                model,
                frame.loc[:, list(config.features) + ["target"]],
                config,
                style_map,
                ticket_id,
            )

        result_payload: dict[str, Any] = {
            "config": config,
            "model": model,
            "metrics": {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
            },
            "confusion_matrix": confusion,
            "confusion_matrix_path": confusion_path,
        }
        if figure_path is not None:
            result_payload["figure_path"] = figure_path
        results.append(result_payload)

    return results


__all__ = [
    "KnnConfig",
    "compute_confusion_matrix",
    "evaluate_classifier",
    "plot_decision_boundary",
    "prepare_dataset",
    "run_knn_experiments",
    "train_classifier",
]
