"""Fail-closed local approval gate for a GKE benchmark execution session."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Mapping, Sequence
from urllib.parse import urlsplit


SHA256_RE = re.compile(r"[0-9a-f]{64}")
DNS_ID_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
BUILD_TAG_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
GSA_RE = re.compile(
    r"[a-z][a-z0-9-]{4,29}@[a-z][a-z0-9-]{4,28}\.iam\.gserviceaccount\.com"
)
PRINCIPAL_RE = re.compile(r"[^\s@]+@[^\s@]+")
APPROVER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+@:-]{0,127}")
PLACEHOLDER_RE = re.compile(r"(?:__[^\s]*__|REPLACE|REQUIRED|CHANGEME|PLACEHOLDER|TODO)", re.I)
RUN_BUCKET = "pertinence-509116-pertinence-runs"
PROJECT = "pertinence-509116"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

CONTROLLER_SOURCES = {
    "quay.io/jetstack/cert-manager-controller:v1.20.4",
    "quay.io/jetstack/cert-manager-webhook:v1.20.4",
    "quay.io/jetstack/cert-manager-cainjector:v1.20.4",
    "quay.io/jetstack/cert-manager-startupapicheck:v1.20.4",
    "quay.io/jetstack/cert-manager-acmesolver:v1.20.4",
    "kserve/kserve-controller:v0.20.0",
    "quay.io/brancz/kube-rbac-proxy:v0.18.0",
    "kserve/storage-initializer:v0.20.0",
    "kserve/agent:v0.20.0",
    "kserve/router:v0.20.0",
    "kserve/art-explainer:v0.20.0",
}
BASE_SOURCES = {"python:3.11-slim", "busybox:1.37.0"}

REQUIRED_INPUTS = {
    "EXPECTED_PRINCIPAL",
    "BENCHMARK_GSA",
    "EVIDENCE_ROOT",
    "RUN_SET_ID",
    "BENCHMARK_ID_FINAL",
    "BENCHMARK_ID_C101",
    "APPROVED_BY",
    "APPROVED_AT",
    "APPROVED_FINAL_LABEL_SHA256",
    "GOLDEN_LOGITS_ATOL",
    "REVIEWED_INTEGRATION_DRIVER",
    "REVIEWED_INTEGRATION_DRIVER_SHA256",
    "REVIEWED_PARITY_DRIVER",
    "REVIEWED_PARITY_DRIVER_SHA256",
    "BUILD_TAG",
    "APPROVED_PROMOTION_ID",
    "IMMUTABLE_BUILD_BUNDLE_PREFIX",
    "EVALUATION_NPZ_URI",
    "CONTROLLER_IMAGE_INVENTORY",
    "CONTROLLER_IMAGE_INVENTORY_SHA256",
    "BASE_IMAGE_INVENTORY",
    "BASE_IMAGE_INVENTORY_SHA256",
    "DIRTY_WORKTREE_APPROVED",
    "WORKTREE_SNAPSHOT_SHA256",
}

EVIDENCE_DIRECTORIES = (
    "00-session",
    "01-artifacts",
    "02-charts",
    "03-cloud-build",
    "04-deployment-images",
    "05-promotion",
    "06-traces",
    "07-iam",
    "08-preflight",
    "09-terraform-plan",
    "10-terraform-apply",
    "11-addons",
    "12-workloads",
    "13-readiness",
    "14-parity",
    "15-cold-start",
    "20-stop",
    "receipts",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_worktree(repository_root: Path) -> dict[str, object]:
    """Hash current tracked changes and untracked bytes against HEAD deterministically."""
    root = repository_root.resolve(strict=True)
    if not (root / ".git").exists():
        raise ValueError(f"repository root has no .git directory: {root}")

    def git_paths(arguments: list[str]) -> set[str]:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("unable to inventory the Git worktree against HEAD") from error
        paths: set[str] = set()
        for encoded in completed.stdout.split(b"\0"):
            if not encoded:
                continue
            path = os.fsdecode(encoded)
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("Git returned an unsafe worktree path")
            paths.add(path)
        return paths

    paths = git_paths(["diff", "--name-only", "-z", "HEAD", "--"])
    paths.update(git_paths(["ls-files", "--others", "--exclude-standard", "-z", "--"]))
    entries: list[dict[str, object]] = []
    for relative in sorted(paths, key=lambda item: item.encode("utf-8", "surrogateescape")):
        path = root / relative
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            entries.append({"path": relative, "type": "deleted"})
            continue
        mode = f"{stat.S_IMODE(metadata.st_mode):04o}"
        if stat.S_ISREG(metadata.st_mode):
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "mode": mode,
                    "size": metadata.st_size,
                    "sha256": _sha256(path),
                }
            )
        elif stat.S_ISLNK(metadata.st_mode):
            target = os.readlink(path)
            encoded_target = os.fsencode(target)
            entries.append(
                {
                    "path": relative,
                    "type": "symlink",
                    "mode": mode,
                    "size": len(encoded_target),
                    "sha256": hashlib.sha256(encoded_target).hexdigest(),
                    "target": target,
                }
            )
        else:
            raise ValueError(f"unsupported dirty worktree entry type: {relative}")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "repository_root": str(root),
        "entries": entries,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _write_once(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)


def read_env(path: Path) -> dict[str, str]:
    """Read a non-executable strict NAME=value file."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"env must be a regular non-symlink file: {path}")
    values: dict[str, str] = {}
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{number}: expected NAME=value")
        key, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"{path}:{number}: invalid variable name")
        if key in values:
            raise ValueError(f"{path}:{number}: duplicate {key}")
        if not value or value != value.strip() or any(char.isspace() for char in value):
            raise ValueError(f"{path}:{number}: {key} must be one non-empty token")
        values[key] = value
    return values


def parse_assignments(assignments: Sequence[str]) -> dict[str, str]:
    """Parse strict command-line NAME=value assignments."""
    values: dict[str, str] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"expected NAME=value: {assignment}")
        key, value = assignment.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or not value:
            raise ValueError(f"invalid assignment: {assignment}")
        if key in values:
            raise ValueError(f"duplicate command-line input: {key}")
        values[key] = value
    return values


def _validate_sha(value: str, name: str) -> None:
    if not SHA256_RE.fullmatch(value) or value == "0" * 64:
        raise ValueError(f"{name} must be a non-sentinel lowercase SHA-256")


def _validated_file(values: Mapping[str, str], path_key: str, sha_key: str) -> tuple[Path, str]:
    path = Path(values[path_key])
    if not path.is_absolute():
        raise ValueError(f"{path_key} must be absolute")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path_key} must be a regular non-symlink file")
    expected = values[sha_key]
    _validate_sha(expected, sha_key)
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{path_key} SHA-256 differs from {sha_key}")
    return path, actual


def _validate_inventory(path: Path, expected_sources: set[str], label: str) -> None:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} inventory must be valid UTF-8 JSON") from error
    if not isinstance(body, dict) or set(body) != {
        "schema_version",
        "queried_at",
        "crane_version",
        "images",
    }:
        raise ValueError(f"{label} inventory has missing or unexpected root fields")
    if body["schema_version"] != 1:
        raise ValueError(f"{label} inventory schema_version must be 1")
    try:
        queried_at = datetime.strptime(body["queried_at"], "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} inventory queried_at must be RFC3339 UTC") from error
    if queried_at.strftime("%Y-%m-%dT%H:%M:%SZ") != body["queried_at"]:
        raise ValueError(f"{label} inventory queried_at must be canonical RFC3339 UTC")
    crane_version = body["crane_version"]
    if (
        not isinstance(crane_version, str)
        or not crane_version
        or any(character.isspace() for character in crane_version)
        or PLACEHOLDER_RE.search(crane_version)
    ):
        raise ValueError(f"{label} inventory crane_version is invalid")
    images = body["images"]
    if not isinstance(images, list):
        raise ValueError(f"{label} inventory must contain an images list")
    sources: set[str] = set()
    for image in images:
        if (
            not isinstance(image, dict)
            or set(image) != {"source", "digest", "os", "architecture"}
            or not isinstance(image.get("source"), str)
        ):
            raise ValueError(f"{label} inventory contains a malformed image")
        source = image["source"]
        if source in sources:
            raise ValueError(f"{label} inventory contains duplicate source: {source}")
        sources.add(source)
        digest = image.get("digest")
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError(f"{label} image {source} lacks a SHA-256 digest")
        if digest == "sha256:" + "0" * 64:
            raise ValueError(f"{label} image {source} uses a sentinel digest")
        if image.get("os") != "linux" or image.get("architecture") != "amd64":
            raise ValueError(f"{label} image {source} is not linux/amd64")
    if sources != expected_sources:
        missing = sorted(expected_sources - sources)
        extra = sorted(sources - expected_sources)
        raise ValueError(f"{label} inventory source mismatch; missing={missing}, extra={extra}")


def _validate_gcs(value: str, name: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "gs"
        or parsed.netloc != RUN_BUCKET
        or not parsed.path.startswith("/pertinence/")
        or parsed.query
        or parsed.fragment
        or "//" in parsed.path
        or any(part in {".", ".."} for part in parsed.path.split("/"))
        or any(character.isspace() or ord(character) < 32 for character in value)
        or any(character in value for character in "*?[]")
    ):
        raise ValueError(
            f"{name} must be an exact gs://{RUN_BUCKET}/pertinence/... path without a glob"
        )


def _validate_new_root(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError("EVIDENCE_ROOT must be absolute")
    if path.exists() or path.is_symlink():
        raise ValueError("EVIDENCE_ROOT must not already exist")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("EVIDENCE_ROOT parent must be an existing non-symlink directory")
    current = parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError("EVIDENCE_ROOT ancestors must not be symlinks")
        current = current.parent


def validate_inputs(
    values: Mapping[str, str], *, repository_root: Path = REPOSITORY_ROOT
) -> dict[str, object]:
    """Validate every approval without creating files or making cloud calls."""
    missing = sorted(REQUIRED_INPUTS - values.keys())
    extra = sorted(values.keys() - REQUIRED_INPUTS)
    if missing:
        raise ValueError(f"missing inputs: {', '.join(missing)}")
    if extra:
        raise ValueError(f"unexpected inputs: {', '.join(extra)}")
    for key, value in values.items():
        if PLACEHOLDER_RE.search(value):
            raise ValueError(f"{key} contains a placeholder")

    if not PRINCIPAL_RE.fullmatch(values["EXPECTED_PRINCIPAL"]):
        raise ValueError("EXPECTED_PRINCIPAL must be an account identity")
    if not GSA_RE.fullmatch(values["BENCHMARK_GSA"]):
        raise ValueError("BENCHMARK_GSA must be a valid service-account address")
    if not values["BENCHMARK_GSA"].endswith(f"@{PROJECT}.iam.gserviceaccount.com"):
        raise ValueError("BENCHMARK_GSA must belong to the approved project")
    if not APPROVER_RE.fullmatch(values["APPROVED_BY"]):
        raise ValueError("APPROVED_BY must be a compact human identity")
    try:
        approved_at = datetime.strptime(values["APPROVED_AT"], "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError("APPROVED_AT must be an RFC3339 UTC timestamp ending in Z") from error
    if approved_at.strftime("%Y-%m-%dT%H:%M:%SZ") != values["APPROVED_AT"]:
        raise ValueError("APPROVED_AT must be a canonical RFC3339 UTC timestamp")

    ids = {
        key: values[key]
        for key in ("RUN_SET_ID", "BENCHMARK_ID_FINAL", "BENCHMARK_ID_C101")
    }
    for key, value in ids.items():
        if not DNS_ID_RE.fullmatch(value):
            raise ValueError(f"{key} must be a DNS-safe label")
    if len(set(ids.values())) != len(ids):
        raise ValueError("run and benchmark IDs must be distinct")
    if not BUILD_TAG_RE.fullmatch(values["BUILD_TAG"]):
        raise ValueError("BUILD_TAG is not a valid unique image tag")
    if not DNS_ID_RE.fullmatch(values["APPROVED_PROMOTION_ID"]):
        raise ValueError("APPROVED_PROMOTION_ID must be a DNS-safe label")
    all_identifiers = [*ids.values(), values["BUILD_TAG"], values["APPROVED_PROMOTION_ID"]]
    if len(set(all_identifiers)) != len(all_identifiers):
        raise ValueError("run, benchmark, build, and promotion IDs must be distinct")

    _validate_sha(values["APPROVED_FINAL_LABEL_SHA256"], "APPROVED_FINAL_LABEL_SHA256")
    try:
        golden_logits_atol = float(values["GOLDEN_LOGITS_ATOL"])
    except ValueError as error:
        raise ValueError("GOLDEN_LOGITS_ATOL must be a finite non-negative number") from error
    if not math.isfinite(golden_logits_atol) or golden_logits_atol < 0:
        raise ValueError("GOLDEN_LOGITS_ATOL must be a finite non-negative number")
    _validate_sha(values["WORKTREE_SNAPSHOT_SHA256"], "WORKTREE_SNAPSHOT_SHA256")
    if values["DIRTY_WORKTREE_APPROVED"] != "true":
        raise ValueError("DIRTY_WORKTREE_APPROVED must be exactly true")
    worktree = snapshot_worktree(repository_root)
    if worktree["sha256"] != values["WORKTREE_SNAPSHOT_SHA256"]:
        raise ValueError("current dirty worktree differs from WORKTREE_SNAPSHOT_SHA256")
    _validate_gcs(values["IMMUTABLE_BUILD_BUNDLE_PREFIX"], "IMMUTABLE_BUILD_BUNDLE_PREFIX")
    _validate_gcs(values["EVALUATION_NPZ_URI"], "EVALUATION_NPZ_URI")
    if values["IMMUTABLE_BUILD_BUNDLE_PREFIX"] == values["EVALUATION_NPZ_URI"]:
        raise ValueError("immutable build bundle and evaluation NPZ GCS paths must differ")
    build_parts = urlsplit(values["IMMUTABLE_BUILD_BUNDLE_PREFIX"]).path.split("/")
    evaluation_parts = urlsplit(values["EVALUATION_NPZ_URI"]).path.split("/")
    if values["BUILD_TAG"] not in build_parts:
        raise ValueError("IMMUTABLE_BUILD_BUNDLE_PREFIX must be scoped by BUILD_TAG")
    if values["RUN_SET_ID"] not in evaluation_parts:
        raise ValueError("EVALUATION_NPZ_URI must be scoped by RUN_SET_ID")

    integration_path, integration_sha = _validated_file(
        values, "REVIEWED_INTEGRATION_DRIVER", "REVIEWED_INTEGRATION_DRIVER_SHA256"
    )
    parity_path, parity_sha = _validated_file(
        values, "REVIEWED_PARITY_DRIVER", "REVIEWED_PARITY_DRIVER_SHA256"
    )
    controller_path, controller_sha = _validated_file(
        values, "CONTROLLER_IMAGE_INVENTORY", "CONTROLLER_IMAGE_INVENTORY_SHA256"
    )
    base_path, base_sha = _validated_file(
        values, "BASE_IMAGE_INVENTORY", "BASE_IMAGE_INVENTORY_SHA256"
    )
    _validate_inventory(controller_path, CONTROLLER_SOURCES, "controller")
    _validate_inventory(base_path, BASE_SOURCES, "base")

    final_worktree = snapshot_worktree(repository_root)
    if final_worktree["sha256"] != values["WORKTREE_SNAPSHOT_SHA256"]:
        raise ValueError("dirty worktree changed while session inputs were being validated")
    worktree = final_worktree

    root = Path(values["EVIDENCE_ROOT"])
    _validate_new_root(root)
    return {
        "approved_at": values["APPROVED_AT"],
        "golden_logits_atol": golden_logits_atol,
        "worktree_snapshot": worktree,
        "reviewed_files": {
            "integration_driver": {"path": str(integration_path), "sha256": integration_sha},
            "parity_driver": {"path": str(parity_path), "sha256": parity_sha},
            "controller_image_inventory": {
                "path": str(controller_path),
                "sha256": controller_sha,
            },
            "base_image_inventory": {"path": str(base_path), "sha256": base_sha},
        },
    }


def create_session(
    values: Mapping[str, str], *, repository_root: Path = REPOSITORY_ROOT
) -> Path:
    """Validate approvals, then create the complete session skeleton once."""
    validated = validate_inputs(values, repository_root=repository_root)
    root = Path(values["EVIDENCE_ROOT"])
    os.mkdir(root, 0o700)
    root.chmod(0o700)
    for relative in EVIDENCE_DIRECTORIES:
        destination = root / relative
        destination.mkdir(mode=0o700)
        destination.chmod(0o700)
    datasets_root = root / "datasets"
    datasets_root.mkdir(mode=0o700)
    datasets_root.chmod(0o700)
    for benchmark_id in (values["BENCHMARK_ID_FINAL"], values["BENCHMARK_ID_C101"]):
        dataset = datasets_root / benchmark_id
        dataset.mkdir(mode=0o700)
        dataset.chmod(0o700)
        for child in ("raw", "external", "analysis", "publish"):
            directory = dataset / child
            directory.mkdir(mode=0o700)
            directory.chmod(0o700)

    receipt = {
        "schema_version": 1,
        "valid": True,
        "cloud_calls_performed": False,
        "approved_inputs": dict(sorted(values.items())),
        **validated,
        "evidence_directories": sorted(
            str(path.relative_to(root)) for path in root.rglob("*") if path.is_dir()
        ),
    }
    receipt_path = root / "00-session" / "session-gate-receipt.json"
    _write_once(receipt_path, json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt_path
