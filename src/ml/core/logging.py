"""Experiment logging utilities for metrics, artifacts, and traces.

This module will coordinate writing structured logs to docs/results and docs/trace so that
experiments remain reproducible across agents.

Todo:
    * Implement JSON/CSV log writers and structured trace outputs.
    * Provide helper routines for recording environment metadata.
    * Integrate artifact saving hooks for plots and tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "docs" / "results"
TRACE_ROOT = PROJECT_ROOT / "docs" / "trace"


@dataclass(frozen=True)
class ResultArtifact:
    """Record describing an artifact generated during a ticket.

    Attributes:
        ticket_id: Identifier of the ticket that produced the artifact.
        artifact_path: Filesystem path to the generated artifact.
        description: Human-friendly description of the artifact contents.
        created_at: Timestamp of artifact creation in UTC.
    """

    ticket_id: str
    artifact_path: Path
    description: str
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        """Normalise the artifact path field to a :class:`pathlib.Path` instance.

        Returns:
            None: This method mutates the dataclass field in-place.
        """
        object.__setattr__(self, "artifact_path", Path(self.artifact_path))


def _format_artifacts_markdown(artifacts: Sequence[ResultArtifact]) -> list[str]:
    """Render a list of artifacts into Markdown bullet points.

    Args:
        artifacts: Collection of artifacts to translate into Markdown.

    Returns:
        list[str]: Lines ready to be written to a Markdown document.
    """
    if not artifacts:
        return ["(No artifacts recorded.)"]

    lines: list[str] = []
    for artifact in artifacts:
        relative = artifact.artifact_path
        try:
            relative = artifact.artifact_path.relative_to(PROJECT_ROOT)
        except ValueError:
            pass
        lines.append(
            f"- `{relative.as_posix()}` — {artifact.description}"
        )
    return lines


def write_result_markdown(
    ticket_id: str, summary: str, artifacts: Sequence[ResultArtifact]
) -> Path:
    """Append a results entry to ``docs/results/<ticket_id>.md``.

    Args:
        ticket_id: Identifier of the F-ticket being summarised.
        summary: Short paragraph describing the outcomes of the work.
        artifacts: Iterable of artifacts generated during the task.

    Returns:
        Path: Absolute path to the Markdown file that was updated.
    """
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    destination = RESULTS_ROOT / f"{ticket_id}.md"
    timestamp = datetime.now(timezone.utc).isoformat()
    content_lines = [
        f"## {ticket_id} — {timestamp}",
        "",
        summary.strip(),
        "",
        "### Artifacts",
        "",
        *_format_artifacts_markdown(list(artifacts)),
        "",
    ]
    with destination.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(content_lines))
    return destination


def write_trace(ticket_id: str, content: str) -> Path:
    """Append execution trace content to ``docs/trace/<ticket_id>.md``.

    Args:
        ticket_id: Ticket identifier that owns the trace entry.
        content: Free-form text to append to the trace log.

    Returns:
        Path: Absolute path to the trace Markdown file that was updated.
    """
    TRACE_ROOT.mkdir(parents=True, exist_ok=True)
    destination = TRACE_ROOT / f"{ticket_id}.md"
    entry = [
        f"## {datetime.now(timezone.utc).isoformat()}",
        "",
        content.strip(),
        "",
    ]
    with destination.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(entry))
    return destination


__all__ = [
    "PROJECT_ROOT",
    "RESULTS_ROOT",
    "TRACE_ROOT",
    "ResultArtifact",
    "write_result_markdown",
    "write_trace",
]
