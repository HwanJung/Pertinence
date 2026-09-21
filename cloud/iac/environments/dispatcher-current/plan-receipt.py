#!/usr/bin/env python3
"""Create and verify create-only receipts for benchmark Terraform saved plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2
GUARD_MODE = "benchmark-pools-create"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _run_git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    ).stdout


def _repo_root() -> Path:
    script_directory = Path(__file__).resolve().parent
    output = subprocess.run(
        ["git", "-C", str(script_directory), "rev-parse", "--show-toplevel"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return Path(output).resolve()


def _relative_to_repo(path: Path, repo: Path) -> str | None:
    try:
        return path.resolve().relative_to(repo).as_posix()
    except ValueError:
        return None


def _file_record(path: Path, repo: Path) -> dict[str, Any]:
    relative = path.relative_to(repo).as_posix()
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        content = os.readlink(path).encode("utf-8", "surrogateescape")
        kind = "symlink"
    elif stat.S_ISREG(metadata.st_mode):
        content = path.read_bytes()
        kind = "file"
    else:
        raise RuntimeError(f"unsupported source file type: {relative}")
    return {
        "path": relative,
        "kind": kind,
        "mode": stat.S_IMODE(metadata.st_mode),
        "sha256": _sha256_bytes(content),
    }


def _source_snapshot(repo: Path) -> dict[str, Any]:
    iac_root = repo / "cloud" / "iac"
    candidates = {
        path
        for path in iac_root.rglob("*")
        if path.is_file()
        and ".terraform" not in path.parts
        and (path.suffix in {".tf", ".tfvars", ".hcl", ".sh", ".py"})
    }
    candidates.add(repo / "scripts" / "terraform_plan_guard.py")
    files = [_file_record(path, repo) for path in sorted(candidates)]
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": _sha256_bytes(canonical), "files": files}


def _worktree_snapshot(repo: Path, excluded: set[str]) -> dict[str, Any]:
    tracked = _run_git(repo, "diff", "--name-only", "-z", "HEAD", "--").split(b"\0")
    untracked = _run_git(
        repo, "ls-files", "--others", "--exclude-standard", "-z", "--"
    ).split(b"\0")
    paths = {
        item.decode("utf-8", "surrogateescape")
        for item in tracked + untracked
        if item
    } - excluded
    entries: list[dict[str, Any]] = []
    for relative in sorted(paths):
        path = repo / relative
        if not path.exists() and not path.is_symlink():
            entries.append({"path": relative, "kind": "missing", "mode": None, "sha256": None})
            continue
        entries.append(_file_record(path, repo))
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": _sha256_bytes(canonical), "entries": entries}


def _input_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON top level must be an object: {path}")
    return value


def _excluded_paths(repo: Path, *paths: Path) -> set[str]:
    return {
        relative
        for path in paths
        if (relative := _relative_to_repo(path, repo)) is not None
    }


def _common_receipt(
    plan: Path,
    plan_json: Path,
    profile: Path,
    receipt_path: Path,
    guard: Path,
    guard_output: Path,
) -> dict[str, Any]:
    repo = _repo_root()
    excluded = _excluded_paths(repo, plan, plan_json, receipt_path)
    revision = _run_git(repo, "rev-parse", "HEAD").decode().strip()
    output = guard_output.read_text(encoding="utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "guard": {
            "mode": GUARD_MODE,
            "path": guard.resolve().relative_to(repo).as_posix(),
            "sha256": _sha256_file(guard),
            "output": output,
            "output_sha256": _sha256_bytes(output.encode()),
        },
        "terraform_plan": _input_record(plan),
        "plan_json": _input_record(plan_json),
        "profile": _input_record(profile),
        "git": {
            "revision": revision,
            "worktree": _worktree_snapshot(repo, excluded),
        },
        "source_snapshot": _source_snapshot(repo),
    }


def _write_exclusive(path: Path, body: dict[str, Any]) -> None:
    payload = (json.dumps(body, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _seal(args: argparse.Namespace) -> int:
    repo = _repo_root()
    actual_revision = _run_git(repo, "rev-parse", "HEAD").decode().strip()
    if actual_revision != args.git_revision:
        raise RuntimeError("git revision differs from current checkout")
    receipt = _common_receipt(
        args.plan, args.plan_json, args.profile, args.receipt, args.guard, args.guard_output
    )
    _write_exclusive(args.receipt, receipt)
    return 0


def _verify(args: argparse.Namespace) -> int:
    sealed = _load_json_object(args.receipt)
    current = _common_receipt(
        args.plan,
        args.plan_json,
        args.profile,
        args.receipt,
        args.guard,
        args.guard_output,
    )
    if sealed != current:
        keys = sorted(
            key for key in set(sealed) | set(current) if sealed.get(key) != current.get(key)
        )
        raise RuntimeError("sealed Terraform inputs changed: " + ", ".join(keys))
    if current["terraform_plan"]["path"] != str(args.plan.resolve()):
        raise RuntimeError("saved plan path differs from sealed receipt")
    if current["profile"]["path"] != str(args.profile.resolve()):
        raise RuntimeError("profile path differs from sealed receipt")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "verify"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--guard-output", type=Path, required=True)
    parser.add_argument("--git-revision")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "seal":
            if not args.git_revision:
                raise RuntimeError("seal requires --git-revision")
            return _seal(args)
        if args.git_revision:
            raise RuntimeError("verify does not accept --git-revision")
        return _verify(args)
    except (OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"Terraform plan receipt error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
