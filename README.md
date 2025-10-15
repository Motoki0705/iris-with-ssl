# iris-with-ssl

## Overview
This repository solves the No3/No4 exercise assignments and extends Auto MPG with an SSL masked autoencoder. Runtime code sits under `src/ml/`, and agents follow [AGENTS.md](AGENTS.md) plus the specs in `docs/spec/`. Tickets coordinate implementation so every artifact lands in `docs/results/` and `docs/trace/`.

## Environment Setup
1. Create the managed environment: `uv venv .venv-wsl`
2. Activate before every command: `source .venv-wsl/bin/activate`
3. Install the project in editable mode: `uv pip install -e .`
4. Sync dependencies after updates: `uv sync --active`

## Key Commands
- `uv run --active pytest` — execute the test suite (smoke tests live in `tests/`)
- `uv run --active python -m ml.<module>` — run task modules directly (e.g., `ml.iris.experiments`)
- `uv run --active python -m compileall src/ml` — sanity check syntax across the codebase

## Directory Highlights
- `src/ml/` — core utilities plus `iris/`, `auto_mpg/`, `ssl/` packages
- `docs/spec/` — requirements, architecture, tasks, agent workflow
- `docs/results/`, `docs/trace/` — ticket-scoped outputs and logs (see naming rules in `AGENTS.md`)
- `docs/tickets/` — planner-owned tickets and queue overview (start with `overview.md`)

## Workflow Reference
Follow the context reload checklist in `AGENTS.md`, then open `docs/tickets/overview.md` to pick the next ticket. Each ticket (`docs/tickets/<ticket-id>.md`) explains goals, acceptance criteria, and required artifacts—complete these before requesting judge review.
