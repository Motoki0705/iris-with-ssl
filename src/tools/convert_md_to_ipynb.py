"""Convert Markdown notebooks into Jupyter .ipynb files.

The Markdown format assumes that code cells are enclosed in triple backticks.
All other content is treated as Markdown cells.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def to_source_lines(text: str) -> list[str]:
    if not text:
        return []
    lines = text.splitlines()
    if text.endswith("\n"):
        return [line + "\n" for line in lines]
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def parse_markdown_cells(md_text: str) -> list[dict]:
    cells: list[dict] = []
    buffer: list[str] = []
    in_code_block = False

    def flush_markdown(lines: Iterable[str]) -> None:
        text = "\n".join(lines).strip("\n")
        if text:
            cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": to_source_lines(text + "\n"),
                }
            )

    def flush_code(lines: Iterable[str]) -> None:
        text = "\n".join(lines)
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": to_source_lines(text + "\n" if text else ""),
            }
        )

    for line in md_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_code_block:
                flush_markdown(buffer)
                buffer = []
                in_code_block = True
            else:
                flush_code(buffer)
                buffer = []
                in_code_block = False
            continue

        buffer.append(line)

    if buffer:
        if in_code_block:
            flush_code(buffer)
        else:
            flush_markdown(buffer)

    return cells


def convert_markdown_to_notebook(md_path: Path, output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = md_path.with_suffix(".ipynb")

    text = md_path.read_text(encoding="utf-8")
    cells = parse_markdown_cells(text)

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
                "language": "python",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "cells": cells,
    }

    output_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="*",
        type=Path,
        default=[Path.cwd()],
        help="Markdown files or directories containing them.",
    )
    args = parser.parse_args()

    markdown_files: list[Path] = []
    for source in args.sources:
        if source.is_dir():
            markdown_files.extend(sorted(source.glob("*.md")))
        elif source.suffix.lower() == ".md":
            markdown_files.append(source)
        else:
            raise ValueError(f"Unsupported source: {source}")

    if not markdown_files:
        raise SystemExit("No Markdown files found to convert.")

    for md_file in markdown_files:
        nb_path = convert_markdown_to_notebook(md_file)
        print(f"Converted {md_file} -> {nb_path}")


if __name__ == "__main__":
    main()
