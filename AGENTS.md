# Repository Guidelines

## Context Reload Checklist
- Begin every session by revisiting `docs/spec/requrements.md`, `docs/spec/architecture.md`, and `docs/spec/implementation-tasks.md` to restore project scope and current priorities.
- Review `docs/spec/agents.md` to confirm your role expectations, then open `docs/tickets/overview.md` to see active assignments and sequencing.
- For the ticket you are handling, read `docs/tickets/<ticket-id>.md` and the companion artifacts under `docs/results/` and `docs/trace/` to understand prior work before you begin.

## Project Structure & Key References
- Runtime code lives in `src/ml/` with feature areas split into `core/`, `iris/`, `auto_mpg/`, and `ssl/`. Add new modules within these namespaces.
- Experiments, metrics, and logs must use ticket-prefixed filenames inside `docs/results/` and `docs/trace/` (example: `docs/trace/F-01-training.log`).
- Shared notebooks and scripts belong in `notebooks/` and `scripts/`; keep them thin wrappers over `src/ml` APIs.

## Environment & Commands
- Activate the managed environment before running anything: `source .venv-wsl/bin/activate`.
- Install or refresh dependencies with `uv sync --active`. Execute commands through uv to reuse the active environment, e.g.:
  - `uv run --active pytest` — run the full test suite.
  - `uv run --active python scripts/run_iris_tasks.py --ticket I-01` — invoke automation for a ticket.
- Format and lint with `uv run --active black src/` and `uv run --active ruff check src/`.

## Coding Style & Documentation
- Follow PEP 8 with 4-space indentation, descriptive type hints, and Google-style docstrings on every public function, class, and module. Example:
  ```python
  def load_iris_dataset() -> pd.DataFrame:
      """Load the Iris dataset from data/iris.csv.

      Returns:
          pd.DataFrame: Cleaned dataframe ready for modeling.
      """
  ```
- Prefer explicit naming (`iris_knn_experiment.py`) and keep ticket IDs out of code files.

## Testing & Evidence Workflow
- Mirror source layout under `tests/` (e.g., `tests/iris/test_knn_classifier.py`) and keep tests deterministic with fixed seeds.
- After running `uv run --active pytest`, capture key metrics or plots in `docs/results/<ticket-id>-summary.md` and log execution details in `docs/trace/<ticket-id>.md`.
- Judge reviews rely on these artifacts; do not mark a ticket ready without them.

## Commit & Review Etiquette
- Write imperative, scoped commits (`Implement SSL masking utilities`) and reference ticket IDs in bodies (`Refs docs/tickets/S-02.md`).
- Pull requests must include a concise summary, executed commands, and links to the latest results/trace files; attach screenshots for visual outputs.
- Update `docs/spec/implementation-tasks.md` checkboxes once judge approval is secured, keeping planner and orchestrator in sync.
