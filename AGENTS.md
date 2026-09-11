# Repository Guidelines

## Project Structure & Module Organization

Shared Python code lives in `src/pertinence/`; keep routing, training, metrics, cache, and configuration logic in focused modules there. Tests mirror those concerns in `tests/test_*.py`. Dataset-specific configuration, catalogs, scripts, figures, and documentation belong under `experiments/cifar10/` or `experiments/cifar100/`. Shared command wrappers live in `scripts/`. Generated datasets, checkpoints, caches, and run outputs belong in the ignored `artifacts/<dataset>/` tree—never commit them or recreate root-level `configs/`, `data/`, `docs/`, or `figures/` directories.

## Build, Test, and Development Commands

Use Python 3.11 as the canonical interpreter (3.12 is also supported).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest
.venv/bin/ruff check .
```

The editable install exposes the `pertinence` CLI. Run a focused test with `pytest tests/test_routing.py`, or select a case with `pytest -k directional`. Asset preparation and experiment-stage commands are documented in the root README; production inference and searches require the expected assets and CUDA setup.

## Coding Style & Naming Conventions

Follow standard Python conventions: four-space indentation, `snake_case` functions and modules, `PascalCase` classes, and `UPPER_CASE` constants. Add type annotations to public and internal APIs where practical, use `pathlib.Path` for paths, and keep module docstrings concise. Ruff targets Python 3.11 with a 100-character line limit; run it before submitting changes. Preserve deterministic seeds, fingerprints, immutable configurations, and atomic output behavior in reproducibility-sensitive code.

## Testing Guidelines

Tests use pytest and should be named `test_<behavior>`. Add tests alongside every behavioral change, including failure paths for validation and cache integrity. Use `pytest.mark.parametrize` for input variants. Tests marked `integration` require downloaded assets and PyTorch; keep ordinary unit tests independent of network access. No numeric coverage threshold is configured, so prioritize meaningful regression coverage.

## Commit & Pull Request Guidelines

History is currently sparse and uses short lowercase summaries (for example, `experiment finish`). Prefer concise, imperative commit subjects that describe one logical change. Pull requests should explain the motivation, summarize implementation and experiment impact, list commands run, and link relevant issues. Include updated figures or result excerpts when outputs change, and call out new assets, configuration changes, CUDA requirements, or reproducibility implications.
