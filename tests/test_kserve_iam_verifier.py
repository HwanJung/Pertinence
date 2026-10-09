"""Checks for the fail-closed benchmark IAM policy verifier."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "cloud" / "kserve" / "scripts" / "verify-benchmark-iam.py"
PROJECT = "pertinence-509116"
BUCKET = "pertinence-509116-pertinence-runs"
GSA = "benchmark-runner@pertinence-509116.iam.gserviceaccount.com"
RUN_SET = "run-20260921t120000z"
BENCHMARK_ID = "comparison-20260921t120000z"
KSA = f"serviceAccount:{PROJECT}.svc.id.goog[pertinence-serving/benchmark-runner]"
GSA_MEMBER = f"serviceAccount:{GSA}"


def _condition(title: str, prefix: str) -> dict[str, str]:
    resource = f"projects/_/buckets/{BUCKET}/objects/pertinence/benchmarks/{prefix}/"
    return {
        "title": title,
        "expression": f"resource.name.startsWith('{resource}')",
    }


def _read_condition() -> dict[str, str]:
    bucket_resource = f"projects/_/buckets/{BUCKET}"
    trace = f"{bucket_resource}/objects/pertinence/benchmarks/traces/cifar10-final-20260824/"
    run = f"{bucket_resource}/objects/pertinence/benchmarks/runs/{BENCHMARK_ID}/"
    return {
        "title": f"trace-final-{RUN_SET}",
        "expression": (
            f"resource.name == '{bucket_resource}' || "
            f"resource.name.startsWith('{trace}') || "
            f"resource.name.startsWith('{run}')"
        ),
    }


def _policies() -> tuple[dict[str, object], dict[str, object]]:
    gsa_policy = {
        "etag": "BwYAAABase64=",
        "bindings": [{"role": "roles/iam.workloadIdentityUser", "members": [KSA]}],
    }
    bucket_policy = {
        "version": 3,
        "etag": "CAE=",
        "bindings": [
            {
                "role": "roles/storage.objectViewer",
                "members": [GSA_MEMBER],
                "condition": _read_condition(),
            },
            {
                "role": "roles/storage.objectCreator",
                "members": [GSA_MEMBER],
                "condition": _condition(f"run-{RUN_SET}", f"runs/{BENCHMARK_ID}"),
            },
            {
                "role": "roles/storage.admin",
                "members": ["group:platform@example.com"],
            },
        ],
    }
    return gsa_policy, bucket_policy


def _run(
    tmp_path: Path,
    *,
    gsa_policy: dict[str, object] | None = None,
    bucket_policy: dict[str, object] | None = None,
    keys: list[object] | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    valid_gsa, valid_bucket = _policies()
    paths = {
        "gsa": tmp_path / "gsa.json",
        "bucket": tmp_path / "bucket.json",
        "keys": tmp_path / "keys.json",
        "output": tmp_path / "receipt.json",
    }
    paths["gsa"].write_text(json.dumps(gsa_policy or valid_gsa), encoding="utf-8")
    paths["bucket"].write_text(json.dumps(bucket_policy or valid_bucket), encoding="utf-8")
    command = [
        sys.executable,
        str(SCRIPT),
        "--project",
        PROJECT,
        "--bucket",
        BUCKET,
        "--gsa",
        GSA,
        "--run-set-id",
        RUN_SET,
        "--benchmark-id",
        BENCHMARK_ID,
        "--gsa-policy",
        str(paths["gsa"]),
        "--bucket-policy",
        str(paths["bucket"]),
        "--output",
        str(paths["output"]),
    ]
    paths["keys"].write_text(json.dumps(keys or []), encoding="utf-8")
    command.extend(["--service-account-keys", str(paths["keys"])])
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    receipt = json.loads(paths["output"].read_text()) if paths["output"].exists() else {}
    return result, receipt


def test_accepts_exact_boundary_and_ignores_unrelated_principals(tmp_path: Path) -> None:
    result, receipt = _run(
        tmp_path,
        keys=[{"name": "projects/x/serviceAccounts/y/keys/z", "keyType": "SYSTEM_MANAGED"}],
    )

    assert result.returncode == 0, result.stderr
    assert receipt["valid"] is True
    assert receipt["observed"] == {
        "benchmark_bucket_bindings": 2,
        "bucket_policy_etag": "CAE=",
        "gsa_policy_etag": "BwYAAABase64=",
        "user_managed_key_count": 0,
    }


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("missing", "exactly the two expected conditional bindings"),
        ("wrong-expression", "exactly the two expected conditional bindings"),
        ("trace-only", "exactly the two expected conditional bindings"),
        ("broad", "forbidden broad bucket role roles/storage.objectAdmin"),
        ("extra-viewer", "exactly the two expected conditional bindings"),
    ],
)
def test_rejects_missing_mutated_or_excess_bucket_permissions(
    tmp_path: Path, case: str, expected: str
) -> None:
    _, bucket = _policies()
    bindings = bucket["bindings"]
    assert isinstance(bindings, list)
    if case == "missing":
        del bindings[0]
    elif case == "wrong-expression":
        bindings[0]["condition"]["expression"] += " "
    elif case == "trace-only":
        bindings[0]["condition"] = _condition(
            f"trace-final-{RUN_SET}", "traces/cifar10-final-20260824"
        )
    elif case == "broad":
        bindings.append({"role": "roles/storage.objectAdmin", "members": [GSA_MEMBER]})
    else:
        bindings.append(
            {
                "role": "roles/storage.objectViewer",
                "members": [GSA_MEMBER],
                "condition": _condition("unexpected", "traces/unapproved"),
            }
        )

    result, receipt = _run(tmp_path, bucket_policy=bucket)

    assert result.returncode == 2
    assert receipt["valid"] is False
    assert any(expected in reason for reason in receipt["invalid_reasons"])


def test_rejects_non_exact_workload_identity_binding(tmp_path: Path) -> None:
    gsa, _ = _policies()
    gsa["bindings"][0]["members"].append("serviceAccount:another.example")

    result, receipt = _run(tmp_path, gsa_policy=gsa)

    assert result.returncode == 2
    assert any("exactly one unconditional" in reason for reason in receipt["invalid_reasons"])


def test_rejects_user_managed_service_account_key(tmp_path: Path) -> None:
    result, receipt = _run(
        tmp_path,
        keys=[
            {
                "name": "projects/-/serviceAccounts/x/keys/user-key",
                "keyType": "USER_MANAGED",
            }
        ],
    )

    assert result.returncode == 2
    assert receipt["observed"]["user_managed_key_count"] == 1
    assert any("user-managed" in reason for reason in receipt["invalid_reasons"])


def test_receipt_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    output.write_text("owned\n", encoding="utf-8")
    gsa, bucket = _policies()
    (tmp_path / "gsa.json").write_text(json.dumps(gsa), encoding="utf-8")
    (tmp_path / "bucket.json").write_text(json.dumps(bucket), encoding="utf-8")
    (tmp_path / "keys.json").write_text("[]", encoding="utf-8")
    command = [
        sys.executable,
        str(SCRIPT),
        "--project",
        PROJECT,
        "--bucket",
        BUCKET,
        "--gsa",
        GSA,
        "--run-set-id",
        RUN_SET,
        "--benchmark-id",
        BENCHMARK_ID,
        "--gsa-policy",
        str(tmp_path / "gsa.json"),
        "--bucket-policy",
        str(tmp_path / "bucket.json"),
        "--service-account-keys",
        str(tmp_path / "keys.json"),
        "--output",
        str(output),
    ]

    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode != 0
    assert output.read_text(encoding="utf-8") == "owned\n"
