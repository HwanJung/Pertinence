"""Portable artifact integrity, safe local paths, and atomic publication."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import unquote, urlparse

from .contracts.common import digest
from .reproducibility import canonical_json_hash, sha256_file


def local_path(uri: str | Path) -> Path:
    parsed = urlparse(str(uri))
    if parsed.scheme and (parsed.scheme != "file" or parsed.netloc not in ("", "localhost")):
        raise ValueError(
            "local components accept paths or file:// URIs; stage remote artifacts first"
        )
    if parsed.query or parsed.fragment:
        raise ValueError("input URI must not contain a query or fragment")
    path = Path(unquote(parsed.path)) if parsed.scheme else Path(uri)
    if not path.exists() or path.is_symlink():
        raise ValueError(f"input must exist and must not be a symlink: {path}")
    return path.resolve()


def safe_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or "\\" in relative
        or ":" in relative
    ):
        raise ValueError(f"unsafe artifact path: {relative}")
    candidate = root / path
    for parent in [candidate, *candidate.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"symlink inside artifact: {relative}")
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"artifact path escapes bundle: {relative}")
    return candidate


def file_inventory(root: Path, *, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    inventory = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        safe_path(root, relative)
        if path.is_file() and relative not in exclude:
            inventory[relative] = sha256_file(path)
        elif not path.is_file() and not path.is_dir():
            raise ValueError(f"unsupported bundle entry: {relative}")
    return inventory


def artifact_sha256(path: Path) -> str:
    """Directories hash the canonical relative-path -> file-SHA256 inventory."""
    return canonical_json_hash(file_inventory(path)) if path.is_dir() else sha256_file(path)


def verify_artifact(path: Path, expected: str, *, max_bytes: int | None = None) -> None:
    digest(expected)
    if max_bytes is not None:
        paths = path.rglob("*") if path.is_dir() else [path]
        size = 0
        for item in paths:
            if item.is_symlink():
                raise ValueError("symlinks are not allowed in input artifacts")
            if item.is_file():
                size += item.stat().st_size
            if size > max_bytes:
                raise ValueError("bundle exceeds platform byte limit")
    if artifact_sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path.name}")


@contextmanager
def atomic_directory(output: Path):
    """Publish a complete new directory; never overwrite an existing artifact."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"output artifact already exists: {output}")
    if output.is_symlink():
        raise ValueError("output artifact cannot be a symlink")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        yield temporary
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(filename, path)
    finally:
        Path(filename).unlink(missing_ok=True)


def seal(value: dict) -> dict:
    return {**value, "fingerprint": canonical_json_hash(value)}


def read_sealed(path: Path) -> dict:
    from .contracts.common import parse_manifest

    value = parse_manifest(path.read_text(encoding="utf-8"))
    fingerprint = value.pop("fingerprint", None)
    if fingerprint != canonical_json_hash(value):
        raise ValueError(f"artifact fingerprint mismatch: {path.name}")
    return {**value, "fingerprint": fingerprint}


def code_fingerprint() -> str:
    root = Path(__file__).parent
    return canonical_json_hash(
        {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in sorted(root.rglob("*.py"))
        }
    )
