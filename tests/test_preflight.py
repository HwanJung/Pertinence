"""Environment contract tests for the production preflight."""

from importlib import metadata
from pathlib import Path

import pytest

from pertinence.preflight import PreflightError, _check_python_and_dependencies


def _write_project(path: Path, dependencies: list[str]) -> None:
    values = ", ".join(repr(value) for value in dependencies)
    (path / "pyproject.toml").write_text(
        f"[project]\ndependencies = [{values}]\n",
        encoding="utf-8",
    )


def test_dependency_check_uses_matching_environment_markers(tmp_path: Path) -> None:
    packaging_version = metadata.version("packaging")
    _write_project(
        tmp_path,
        [
            f"packaging=={packaging_version}; python_version >= '3.11'",
            "package-that-must-not-exist==1; python_version < '3.0'",
        ],
    )

    _check_python_and_dependencies(tmp_path)


def test_dependency_check_reports_active_marker_mismatch(tmp_path: Path) -> None:
    _write_project(
        tmp_path,
        ["packaging==0; python_version >= '3.11'"],
    )

    with pytest.raises(PreflightError, match=r"packaging: .*expected ==0"):
        _check_python_and_dependencies(tmp_path)
