"""Shared parsing and modelcar helpers for serving commands."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, TypeVar

from ...reproducibility import sha256_file
from ..contracts import PromotionManifestV1


_STARTUP_ARTIFACT_TIMEOUT_SECONDS = 120.0
_STARTUP_ARTIFACT_POLL_SECONDS = 0.1
_T = TypeVar("_T")


def _wait_for_modelcar(
    loader: Callable[[], _T],
    *,
    timeout_seconds: float = _STARTUP_ARTIFACT_TIMEOUT_SECONDS,
    poll_seconds: float = _STARTUP_ARTIFACT_POLL_SECONDS,
) -> _T:
    """Wait for KServe's modelcar sidecar to finish publishing verified files."""

    if timeout_seconds < 0 or poll_seconds <= 0:
        raise ValueError("modelcar wait durations are invalid")
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            return loader()
        except (OSError, ValueError) as error:
            if time.monotonic() >= deadline:
                raise TimeoutError("modelcar artifacts were not ready before startup timeout") from error
            time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def _verified_model_path(path: Path, expected_sha256: str) -> Path:
    """Return a model only after the sidecar-visible bytes match the promotion."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("modelcar model is not a regular file")
    if sha256_file(path) != expected_sha256:
        raise ValueError("modelcar model SHA-256 differs from the promotion")
    return path


def _key_values(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item or key in result:
            raise ValueError(f"{label} entries must be unique NAME=VALUE pairs")
        result[key] = item
    return result


def _load_promotion(path: Path) -> dict[str, Any]:
    return PromotionManifestV1.from_text(path.read_text(encoding="utf-8")).to_dict()
