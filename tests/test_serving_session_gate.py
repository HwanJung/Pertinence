"""Tests for the local, fail-closed benchmark session gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from pertinence.serving.session_gate import (
    BASE_SOURCES,
    CONTROLLER_SOURCES,
    create_session,
    read_env,
    snapshot_worktree,
)

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(path: Path, sources: set[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "queried_at": "2026-09-21T01:00:00Z",
                "crane_version": "v0.20.7",
                "images": [
                    {
                        "source": source,
                        "digest": "sha256:" + "a" * 64,
                        "os": "linux",
                        "architecture": "amd64",
                    }
                    for source in sorted(sources)
                ],
            }
        ),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path) -> dict[str, str]:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    tracked = repository / "tracked.txt"
    tracked.write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "baseline",
        ],
        check=True,
    )
    tracked.write_text("approved dirty bytes\n", encoding="utf-8")
    (repository / "untracked.txt").write_text("approved untracked bytes\n", encoding="utf-8")
    integration = tmp_path / "integration.py"
    parity = tmp_path / "parity.py"
    controller = tmp_path / "controllers.json"
    base = tmp_path / "bases.json"
    integration.write_text("# reviewed integration\n", encoding="utf-8")
    parity.write_text("# reviewed parity\n", encoding="utf-8")
    _inventory(controller, CONTROLLER_SOURCES)
    _inventory(base, BASE_SOURCES)
    return {
        "EXPECTED_PRINCIPAL": "human@example.com",
        "BENCHMARK_GSA": "benchmark-runner@pertinence-509116.iam.gserviceaccount.com",
        "EVIDENCE_ROOT": str(tmp_path / "evidence"),
        "RUN_SET_ID": "run-20260921t010203z",
        "BENCHMARK_ID_FINAL": "final-20260921t010203z",
        "BENCHMARK_ID_C101": "c101-20260921t010203z",
        "APPROVED_BY": "reviewer@example.com",
        "APPROVED_AT": "2026-09-21T01:02:03Z",
        "APPROVED_FINAL_LABEL_SHA256": "b" * 64,
        "GOLDEN_LOGITS_ATOL": "1e-5",
        "REVIEWED_INTEGRATION_DRIVER": str(integration),
        "REVIEWED_INTEGRATION_DRIVER_SHA256": _sha256(integration),
        "REVIEWED_PARITY_DRIVER": str(parity),
        "REVIEWED_PARITY_DRIVER_SHA256": _sha256(parity),
        "BUILD_TAG": "build-20260921t010203z",
        "APPROVED_PROMOTION_ID": "promotion-20260921t010203z",
        "IMMUTABLE_BUILD_BUNDLE_PREFIX": (
            "gs://pertinence-509116-pertinence-runs/pertinence/builds/build-20260921t010203z"
        ),
        "EVALUATION_NPZ_URI": (
            "gs://pertinence-509116-pertinence-runs/pertinence/evaluations/"
            "run-20260921t010203z/evaluation-final.npz"
        ),
        "CONTROLLER_IMAGE_INVENTORY": str(controller),
        "CONTROLLER_IMAGE_INVENTORY_SHA256": _sha256(controller),
        "BASE_IMAGE_INVENTORY": str(base),
        "BASE_IMAGE_INVENTORY_SHA256": _sha256(base),
        "DIRTY_WORKTREE_APPROVED": "true",
        "WORKTREE_SNAPSHOT_SHA256": str(snapshot_worktree(repository)["sha256"]),
    }


def test_create_session_builds_private_create_only_skeleton(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    receipt_path = create_session(values, repository_root=tmp_path / "repo")

    root = Path(values["EVIDENCE_ROOT"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["valid"] is True
    assert receipt["cloud_calls_performed"] is False
    assert receipt["approved_inputs"]["BENCHMARK_ID_FINAL"] == values["BENCHMARK_ID_FINAL"]
    assert receipt["worktree_snapshot"]["sha256"] == values["WORKTREE_SNAPSHOT_SHA256"]
    assert {item["path"] for item in receipt["worktree_snapshot"]["entries"]} == {
        "tracked.txt",
        "untracked.txt",
    }
    assert (root.stat().st_mode & 0o777) == 0o700
    assert (receipt_path.stat().st_mode & 0o777) == 0o600
    assert all((path.stat().st_mode & 0o777) == 0o700 for path in root.rglob("*") if path.is_dir())
    assert (root / "datasets" / values["BENCHMARK_ID_FINAL"] / "publish").is_dir()
    assert (root / "datasets" / values["BENCHMARK_ID_C101"] / "analysis").is_dir()

    with pytest.raises(ValueError, match="must not already exist"):
        create_session(values, repository_root=tmp_path / "repo")


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("EXPECTED_PRINCIPAL", "__REQUIRED__", "placeholder"),
        ("BENCHMARK_ID_C101", "final-20260921t010203z", "must be distinct"),
        ("APPROVED_AT", "2026-09-21T01:02:03+00:00", "RFC3339 UTC"),
        ("DIRTY_WORKTREE_APPROVED", "yes", "exactly true"),
        ("APPROVED_FINAL_LABEL_SHA256", "0" * 64, "non-sentinel"),
        ("GOLDEN_LOGITS_ATOL", "nan", "finite non-negative"),
        ("EVALUATION_NPZ_URI", "gs://bucket/path/*.npz", "must be an exact"),
    ],
)
def test_rejected_input_leaves_no_evidence_root(
    tmp_path: Path, key: str, value: str, message: str
) -> None:
    values = _inputs(tmp_path)
    values[key] = value

    with pytest.raises(ValueError, match=message):
        create_session(values, repository_root=tmp_path / "repo")
    assert not Path(values["EVIDENCE_ROOT"]).exists()


def test_reviewed_file_hash_mismatch_is_rejected_before_creation(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    values["REVIEWED_PARITY_DRIVER_SHA256"] = "d" * 64

    with pytest.raises(ValueError, match="SHA-256 differs"):
        create_session(values, repository_root=tmp_path / "repo")
    assert not Path(values["EVIDENCE_ROOT"]).exists()


def test_symlinked_reviewed_file_and_env_are_rejected(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    link = tmp_path / "integration-link.py"
    link.symlink_to(values["REVIEWED_INTEGRATION_DRIVER"])
    values["REVIEWED_INTEGRATION_DRIVER"] = str(link)

    with pytest.raises(ValueError, match="non-symlink"):
        create_session(values, repository_root=tmp_path / "repo")
    assert not Path(values["EVIDENCE_ROOT"]).exists()

    env = tmp_path / "session.env"
    env.write_text("EXPECTED_PRINCIPAL=human@example.com\n", encoding="utf-8")
    env_link = tmp_path / "session-link.env"
    env_link.symlink_to(env)
    with pytest.raises(ValueError, match="non-symlink"):
        read_env(env_link)


def test_inventory_rejects_wrong_platform_and_sentinel(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    path = Path(values["BASE_IMAGE_INVENTORY"])
    inventory = json.loads(path.read_text(encoding="utf-8"))
    inventory["images"][0]["architecture"] = "arm64"
    inventory["images"][1]["digest"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(inventory), encoding="utf-8")
    values["BASE_IMAGE_INVENTORY_SHA256"] = _sha256(path)

    with pytest.raises(ValueError, match="not linux/amd64|sentinel digest"):
        create_session(values, repository_root=tmp_path / "repo")
    assert not Path(values["EVIDENCE_ROOT"]).exists()


def test_changed_worktree_is_rejected_before_creation(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    (tmp_path / "repo" / "untracked.txt").write_text("changed after approval\n", encoding="utf-8")

    with pytest.raises(ValueError, match="current dirty worktree differs"):
        create_session(values, repository_root=tmp_path / "repo")
    assert not Path(values["EVIDENCE_ROOT"]).exists()
