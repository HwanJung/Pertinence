#!/usr/bin/env python3
"""Verify the sealed benchmark IAM boundary from read-only policy snapshots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any


PROJECT_RE = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$")
GSA_RE = re.compile(
    r"^[a-z][a-z0-9-]{4,29}@[a-z][a-z0-9-]{4,28}[a-z0-9]\.iam\.gserviceaccount\.com$"
)
DNS_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")

WORKLOAD_IDENTITY_ROLE = "roles/iam.workloadIdentityUser"
TRACE_VIEWER_ROLE = "roles/storage.objectViewer"
RUN_CREATOR_ROLE = "roles/storage.objectCreator"
BROAD_BUCKET_ROLES = frozenset(
    {
        "roles/editor",
        "roles/owner",
        "roles/storage.admin",
        "roles/storage.objectAdmin",
        "roles/storage.legacyBucketOwner",
        "roles/storage.legacyBucketWriter",
        "roles/storage.legacyObjectOwner",
    }
)


class DuplicateKeyError(ValueError):
    """Raised when a JSON object contains duplicate keys."""


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _load_json(path: Path, expected: type[dict[str, Any]] | type[list[Any]]) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"input must be a regular non-symlink file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON input {path}: {error}") from error
    if not isinstance(value, expected):
        raise ValueError(f"{path}: expected a JSON {expected.__name__}")
    return value


def _require_text(value: object, field: str, violations: list[str]) -> str | None:
    if not isinstance(value, str) or not value:
        violations.append(f"{field} must be a non-empty string")
        return None
    return value


def _bindings(policy: dict[str, Any], name: str, violations: list[str]) -> list[dict[str, Any]]:
    raw = policy.get("bindings")
    if not isinstance(raw, list):
        violations.append(f"{name} policy bindings must be a list")
        return []
    bindings: list[dict[str, Any]] = []
    for index, binding in enumerate(raw):
        if not isinstance(binding, dict):
            violations.append(f"{name} policy binding {index} must be an object")
            continue
        role = binding.get("role")
        members = binding.get("members")
        if not isinstance(role, str) or not role:
            violations.append(f"{name} policy binding {index} has an invalid role")
        if (
            not isinstance(members, list)
            or not all(isinstance(member, str) and member for member in members)
            or len(members) != len(set(members))
        ):
            violations.append(f"{name} policy binding {index} has invalid or duplicate members")
        bindings.append(binding)
    return bindings


def _expected_bucket_bindings(
    *, bucket: str, gsa_member: str, run_set_id: str, final_id: str, c101_id: str
) -> set[tuple[str, tuple[str, ...], str, str]]:
    base = f"projects/_/buckets/{bucket}/objects/pertinence/benchmarks"
    entries = (
        (
            TRACE_VIEWER_ROLE,
            f"trace-final-{run_set_id}",
            f"{base}/traces/cifar10-final-20260824/",
        ),
        (
            TRACE_VIEWER_ROLE,
            f"trace-c101-{run_set_id}",
            f"{base}/traces/cifar10.1-v6/",
        ),
        (RUN_CREATOR_ROLE, f"run-final-{run_set_id}", f"{base}/runs/{final_id}/"),
        (RUN_CREATOR_ROLE, f"run-c101-{run_set_id}", f"{base}/runs/{c101_id}/"),
    )
    return {
        (role, (gsa_member,), title, f"resource.name.startsWith('{prefix}')")
        for role, title, prefix in entries
    }


def _bucket_projection(
    binding: dict[str, Any], index: int, violations: list[str]
) -> tuple[str, tuple[str, ...], str, str] | None:
    role = binding.get("role")
    members = binding.get("members")
    condition = binding.get("condition")
    if not isinstance(role, str) or not isinstance(members, list):
        return None
    if not isinstance(condition, dict):
        violations.append(f"bucket policy binding {index} for benchmark GSA is unconditional")
        return None
    extra = set(condition) - {"title", "expression", "description"}
    if extra:
        violations.append(
            f"bucket policy binding {index} condition has unexpected fields: {sorted(extra)}"
        )
    if condition.get("description") not in (None, ""):
        violations.append(f"bucket policy binding {index} condition description must be absent")
    title = condition.get("title")
    expression = condition.get("expression")
    if not isinstance(title, str) or not isinstance(expression, str):
        violations.append(f"bucket policy binding {index} condition is malformed")
        return None
    return role, tuple(members), title, expression


def verify(
    *,
    project: str,
    bucket: str,
    gsa: str,
    run_set_id: str,
    benchmark_id_final: str,
    benchmark_id_c101: str,
    gsa_policy: dict[str, Any],
    bucket_policy: dict[str, Any],
    service_account_keys: list[Any],
) -> dict[str, object]:
    """Return a deterministic receipt for the supplied IAM policy snapshots."""
    violations: list[str] = []
    if not PROJECT_RE.fullmatch(project):
        violations.append("project is not a valid Google Cloud project ID")
    if not BUCKET_RE.fullmatch(bucket):
        violations.append("bucket is not a valid bucket name")
    if not GSA_RE.fullmatch(gsa):
        violations.append("gsa is not a valid service-account email")
    for name, value in (
        ("run-set-id", run_set_id),
        ("benchmark-id-final", benchmark_id_final),
        ("benchmark-id-c101", benchmark_id_c101),
    ):
        if not DNS_ID_RE.fullmatch(value):
            violations.append(f"{name} is not DNS-label safe")
    if benchmark_id_final == benchmark_id_c101:
        violations.append("benchmark IDs must differ")

    gsa_etag = _require_text(gsa_policy.get("etag"), "GSA policy etag", violations)
    bucket_etag = _require_text(bucket_policy.get("etag"), "bucket policy etag", violations)
    gsa_bindings = _bindings(gsa_policy, "GSA", violations)
    bucket_bindings = _bindings(bucket_policy, "bucket", violations)

    ksa_member = (
        f"serviceAccount:{project}.svc.id.goog[pertinence-serving/benchmark-runner]"
    )
    workload_bindings = [
        binding for binding in gsa_bindings if binding.get("role") == WORKLOAD_IDENTITY_ROLE
    ]
    expected_workload = {"role": WORKLOAD_IDENTITY_ROLE, "members": [ksa_member]}
    if workload_bindings != [expected_workload]:
        violations.append(
            "GSA policy must contain exactly one unconditional workloadIdentityUser binding "
            "whose sole member is the benchmark KSA"
        )
    if any(
        ksa_member in binding.get("members", []) and binding.get("role") != WORKLOAD_IDENTITY_ROLE
        for binding in gsa_bindings
    ):
        violations.append("benchmark KSA has an unexpected role on the GSA")

    try:
        bucket_version = int(bucket_policy.get("version", 0))
    except (TypeError, ValueError):
        bucket_version = 0
    if bucket_version < 3:
        violations.append("conditional bucket policy must have version 3 or newer")

    gsa_member = f"serviceAccount:{gsa}"
    selected: list[tuple[str, tuple[str, ...], str, str]] = []
    for index, binding in enumerate(bucket_bindings):
        members = binding.get("members", [])
        if gsa_member not in members:
            continue
        role = binding.get("role")
        if role in BROAD_BUCKET_ROLES:
            violations.append(f"benchmark GSA has forbidden broad bucket role {role}")
        projected = _bucket_projection(binding, index, violations)
        if projected is not None:
            selected.append(projected)

    expected = _expected_bucket_bindings(
        bucket=bucket,
        gsa_member=gsa_member,
        run_set_id=run_set_id,
        final_id=benchmark_id_final,
        c101_id=benchmark_id_c101,
    )
    if len(selected) != 4 or set(selected) != expected:
        violations.append(
            "bucket policy must contain exactly the four expected conditional bindings for "
            "the benchmark GSA"
        )

    user_managed_keys: list[str] = []
    for index, item in enumerate(service_account_keys):
        if not isinstance(item, dict):
            violations.append(f"service-account key entry {index} must be an object")
            continue
        key_type = item.get("keyType")
        if key_type == "USER_MANAGED":
            name = item.get("name")
            user_managed_keys.append(name if isinstance(name, str) else f"entry-{index}")
        elif key_type != "SYSTEM_MANAGED":
            violations.append(f"service-account key entry {index} has unknown keyType")
    if user_managed_keys:
        violations.append("benchmark GSA has user-managed service-account keys")

    return {
        "schema_version": 1,
        "valid": not violations,
        "invalid_reasons": violations,
        "boundary": {
            "project": project,
            "bucket": bucket,
            "gsa": gsa,
            "ksa_member": ksa_member,
            "run_set_id": run_set_id,
            "benchmark_ids": {
                "cifar10_final": benchmark_id_final,
                "cifar10_1_v6": benchmark_id_c101,
            },
        },
        "observed": {
            "gsa_policy_etag": gsa_etag,
            "bucket_policy_etag": bucket_etag,
            "benchmark_bucket_bindings": len(selected),
            "user_managed_key_count": len(user_managed_keys),
        },
    }


def _write_once(path: Path, receipt: dict[str, object]) -> None:
    if not path.parent.is_dir():
        raise ValueError(f"output parent does not exist: {path.parent}")
    data = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--gsa", required=True)
    parser.add_argument("--run-set-id", required=True)
    parser.add_argument("--benchmark-id-final", required=True)
    parser.add_argument("--benchmark-id-c101", required=True)
    parser.add_argument("--gsa-policy", type=Path, required=True)
    parser.add_argument("--bucket-policy", type=Path, required=True)
    parser.add_argument("--service-account-keys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        gsa_policy = _load_json(args.gsa_policy, dict)
        bucket_policy = _load_json(args.bucket_policy, dict)
        keys = _load_json(args.service_account_keys, list)
        receipt = verify(
            project=args.project,
            bucket=args.bucket,
            gsa=args.gsa,
            run_set_id=args.run_set_id,
            benchmark_id_final=args.benchmark_id_final,
            benchmark_id_c101=args.benchmark_id_c101,
            gsa_policy=gsa_policy,
            bucket_policy=bucket_policy,
            service_account_keys=keys,
        )
        _write_once(args.output, receipt)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not receipt["valid"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
