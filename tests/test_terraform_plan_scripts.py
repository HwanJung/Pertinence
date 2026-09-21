"""Subprocess tests for sealing and applying reviewed Terraform saved plans."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = Path("cloud/iac/environments/dispatcher-current")
SERVING_POOL = 'module.gke.google_container_node_pool.this["serving-pool"]'
LOADGEN_POOL = 'module.gke.google_container_node_pool.this["loadgen-pool"]'


def _pool(address: str, machine_type: str, taint_value: str) -> dict[str, object]:
    return {
        "address": address,
        "type": "google_container_node_pool",
        "change": {
            "actions": ["create"],
            "before": None,
            "after": {
                "initial_node_count": 0,
                "deletion_policy": "PREVENT",
                "autoscaling": [{"min_node_count": 0, "max_node_count": 1}],
                "node_config": [
                    {
                        "machine_type": machine_type,
                        "image_type": "COS_CONTAINERD",
                        "service_account": (
                            "pertinence-gke-nodes@pertinence-509116.iam.gserviceaccount.com"
                        ),
                        "spot": False,
                        "disk_size_gb": 50,
                        "disk_type": "pd-standard",
                        "workload_metadata_config": [{"mode": "GKE_METADATA"}],
                        "taint": [
                            {
                                "key": "workload",
                                "value": taint_value,
                                "effect": "NO_SCHEDULE",
                            }
                        ],
                    }
                ],
            },
        },
    }


def _plan_json() -> str:
    value = {
        "format_version": "1.2",
        "terraform_version": "1.16.3",
        "resource_changes": [
            _pool(SERVING_POOL, "n2-standard-8", "pertinence-serving"),
            _pool(LOADGEN_POOL, "n2-standard-4", "pertinence-loadgen"),
        ],
    }
    return json.dumps(value, sort_keys=True) + "\n"


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)


@pytest.fixture
def sealed_repo(tmp_path: Path) -> dict[str, object]:
    repo = tmp_path / "repo"
    environment = repo / ENVIRONMENT
    scripts = repo / "scripts"
    evidence = tmp_path / "evidence"
    fake_bin = tmp_path / "bin"
    environment.mkdir(parents=True)
    scripts.mkdir()
    evidence.mkdir()
    fake_bin.mkdir()

    for name in ("seal-plan.sh", "apply-saved-plan.sh", "plan-receipt.py"):
        shutil.copy2(ROOT / ENVIRONMENT / name, environment / name)
    shutil.copy2(ROOT / "scripts/terraform_plan_guard.py", scripts / "terraform_plan_guard.py")
    (environment / "main.tf").write_text('terraform { required_version = ">= 1.5" }\n')
    profile = environment / "benchmark.tfvars"
    profile.write_text('project_id = "test"\n')
    (repo / "README.md").write_text("sealed worktree\n")

    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()

    plan = evidence / "benchmark.tfplan"
    plan.write_bytes(b"opaque saved plan\n")
    plan_json = evidence / "benchmark.tfplan.json"
    plan_json.write_text(_plan_json())
    receipt = evidence / "receipt.json"
    apply_log = evidence / "apply.log"
    terraform = fake_bin / "terraform"
    terraform.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"$1\" == show && \"$2\" == -json ]]; then\n"
        "  cat \"$FAKE_TERRAFORM_JSON\"\n"
        "elif [[ \"$1\" == apply && $# -eq 2 ]]; then\n"
        "  printf '%s\\n' \"$2\" >>\"$FAKE_APPLY_LOG\"\n"
        "else\n"
        "  exit 91\n"
        "fi\n"
    )
    terraform.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_TERRAFORM_JSON": str(plan_json),
        "FAKE_APPLY_LOG": str(apply_log),
    }
    return {
        "repo": repo,
        "environment": environment,
        "profile": profile,
        "plan": plan,
        "plan_json": plan_json,
        "receipt": receipt,
        "apply_log": apply_log,
        "revision": revision,
        "env": env,
    }


def _seal(fixture: dict[str, object]) -> subprocess.CompletedProcess[str]:
    environment = fixture["environment"]
    assert isinstance(environment, Path)
    return _run(
        [
            str(environment / "seal-plan.sh"),
            str(fixture["plan"]),
            str(fixture["plan_json"]),
            str(fixture["profile"]),
            str(fixture["revision"]),
            str(fixture["receipt"]),
        ],
        cwd=environment.parent,
        env=fixture["env"],  # type: ignore[arg-type]
    )


def _apply(fixture: dict[str, object]) -> subprocess.CompletedProcess[str]:
    environment = fixture["environment"]
    assert isinstance(environment, Path)
    return _run(
        [
            str(environment / "apply-saved-plan.sh"),
            str(fixture["plan"]),
            str(fixture["profile"]),
            str(fixture["receipt"]),
        ],
        cwd=environment.parent,
        env=fixture["env"],  # type: ignore[arg-type]
    )


def test_seal_records_guard_sources_and_worktree_then_applies_exact_plan(
    sealed_repo: dict[str, object],
) -> None:
    sealed = _seal(sealed_repo)
    assert sealed.returncode == 0, sealed.stderr

    receipt_path = sealed_repo["receipt"]
    assert isinstance(receipt_path, Path)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["schema_version"] == 2
    assert receipt["guard"]["mode"] == "benchmark-pools-create"
    assert receipt["guard"]["output"] == (
        "Terraform plan accepted in benchmark-pools-create mode\n"
    )
    assert len(receipt["guard"]["sha256"]) == 64
    assert receipt["git"]["revision"] == sealed_repo["revision"]
    assert receipt["git"]["worktree"]["entries"] == []
    source_paths = {item["path"] for item in receipt["source_snapshot"]["files"]}
    assert "scripts/terraform_plan_guard.py" in source_paths
    assert f"{ENVIRONMENT.as_posix()}/seal-plan.sh" in source_paths
    assert f"{ENVIRONMENT.as_posix()}/apply-saved-plan.sh" in source_paths

    applied = _apply(sealed_repo)
    assert applied.returncode == 0, applied.stderr
    apply_log = sealed_repo["apply_log"]
    assert isinstance(apply_log, Path)
    assert apply_log.read_text().splitlines() == [str(sealed_repo["plan"])]


def test_seal_preserves_dirty_and_untracked_content_snapshot(
    sealed_repo: dict[str, object],
) -> None:
    repo = sealed_repo["repo"]
    assert isinstance(repo, Path)
    (repo / "README.md").write_text("approved dirty content\n")
    (repo / "review-note.txt").write_text("approved untracked content\n")

    sealed = _seal(sealed_repo)

    assert sealed.returncode == 0, sealed.stderr
    receipt_path = sealed_repo["receipt"]
    assert isinstance(receipt_path, Path)
    receipt = json.loads(receipt_path.read_text())
    entries = receipt["git"]["worktree"]["entries"]
    assert {entry["path"] for entry in entries} == {"README.md", "review-note.txt"}
    assert all(len(entry["sha256"]) == 64 for entry in entries)
    assert _apply(sealed_repo).returncode == 0


def test_seal_rejects_json_not_regenerated_from_saved_plan(
    sealed_repo: dict[str, object],
) -> None:
    other_json = Path(str(sealed_repo["plan_json"]) + ".other")
    other_json.write_text("{}\n")
    sealed_repo["plan_json"] = other_json

    result = _seal(sealed_repo)

    assert result.returncode != 0
    assert "differs from terraform show -json" in result.stderr
    assert not Path(sealed_repo["receipt"]).exists()


def test_seal_uses_exclusive_receipt_creation(sealed_repo: dict[str, object]) -> None:
    receipt = sealed_repo["receipt"]
    assert isinstance(receipt, Path)
    receipt.write_text("do not replace\n")

    result = _seal(sealed_repo)

    assert result.returncode != 0
    assert receipt.read_text() == "do not replace\n"


@pytest.mark.parametrize("changed_path", ["README.md", str(ENVIRONMENT / "main.tf")])
def test_apply_rejects_changed_worktree_or_iac_source_before_terraform_apply(
    sealed_repo: dict[str, object], changed_path: str
) -> None:
    sealed = _seal(sealed_repo)
    assert sealed.returncode == 0, sealed.stderr
    repo = sealed_repo["repo"]
    assert isinstance(repo, Path)
    (repo / changed_path).write_text("changed after review\n")

    applied = _apply(sealed_repo)

    assert applied.returncode != 0
    assert "sealed Terraform inputs changed" in applied.stderr
    assert not Path(sealed_repo["apply_log"]).exists()


@pytest.mark.parametrize("input_name", ["plan", "profile"])
def test_apply_rejects_changed_sealed_input(
    sealed_repo: dict[str, object], input_name: str
) -> None:
    sealed = _seal(sealed_repo)
    assert sealed.returncode == 0, sealed.stderr
    changed = sealed_repo[input_name]
    assert isinstance(changed, Path)
    changed.write_bytes(changed.read_bytes() + b"changed after sealing\n")

    applied = _apply(sealed_repo)

    assert applied.returncode != 0
    assert "sealed Terraform inputs changed" in applied.stderr
    assert not Path(sealed_repo["apply_log"]).exists()
