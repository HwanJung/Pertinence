"""Reject unsafe Terraform plans before they can be applied.

The guard consumes the JSON emitted by ``terraform show -json``.  It does not
run Terraform and is intentionally independent of Google Cloud credentials.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SAFE_PASSIVE_ACTIONS = {("no-op",), ("read",)}
BENCHMARK_POOL_CREATE_ADDRESSES = {
    'module.gke.google_container_node_pool.this["loadgen-pool"]',
    'module.gke.google_container_node_pool.this["serving-pool"]',
}
BENCHMARK_POOL_SPECS = {
    'module.gke.google_container_node_pool.this["serving-pool"]': {
        "machine_type": "n2-standard-8",
        "taint_value": "pertinence-serving",
    },
    'module.gke.google_container_node_pool.this["loadgen-pool"]': {
        "machine_type": "n2-standard-4",
        "taint_value": "pertinence-loadgen",
    },
}
PROTECTION_FIELDS = {
    "google_container_cluster": {"deletion_protection"},
    "google_container_node_pool": {"deletion_policy"},
}
PROTECTED_VALUES = {
    ("google_container_cluster", "deletion_protection"): True,
    ("google_container_node_pool", "deletion_policy"): "PREVENT",
}
EXPECTED_NODE_SERVICE_ACCOUNT = (
    "pertinence-gke-nodes@pertinence-509116.iam.gserviceaccount.com"
)


def _changed_top_level_fields(before: Any, after: Any) -> set[str]:
    """Return changed top-level object keys, treating non-objects as one change."""

    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return {"<value>"} if before != after else set()
    keys = set(before) | set(after)
    return {key for key in keys if before.get(key) != after.get(key)}


def _format_actions(actions: Sequence[str]) -> str:
    return "/".join(actions) if actions else "unknown"


def _one_block(value: Any, label: str) -> tuple[Mapping[str, Any] | None, str | None]:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], Mapping):
        return None, f"{label} must contain exactly one object"
    return value[0], None


def _validate_benchmark_pool(address: str, after: Any) -> list[str]:
    if not isinstance(after, Mapping):
        return [f"{address}: created pool after value must be an object"]
    expected = BENCHMARK_POOL_SPECS[address]
    violations: list[str] = []
    for field, wanted in (("initial_node_count", 0), ("deletion_policy", "PREVENT")):
        if after.get(field) != wanted:
            violations.append(f"{address}: {field} must be {wanted!r}")
    autoscaling, error = _one_block(after.get("autoscaling"), "autoscaling")
    if error:
        violations.append(f"{address}: {error}")
    elif autoscaling is not None and (
        autoscaling.get("min_node_count") != 0 or autoscaling.get("max_node_count") != 1
    ):
        violations.append(f"{address}: autoscaling must be exactly 0-1")
    node, error = _one_block(after.get("node_config"), "node_config")
    if error:
        violations.append(f"{address}: {error}")
        return violations
    assert node is not None
    expected_fields = {
        "machine_type": expected["machine_type"],
        "image_type": "COS_CONTAINERD",
        "spot": False,
        "disk_size_gb": 50,
        "disk_type": "pd-standard",
    }
    for field, wanted in expected_fields.items():
        if node.get(field) != wanted:
            violations.append(f"{address}: node_config.{field} must be {wanted!r}")
    service_account = node.get("service_account")
    if service_account != EXPECTED_NODE_SERVICE_ACCOUNT:
        violations.append(f"{address}: node_config.service_account is not the dedicated node SA")
    metadata, error = _one_block(
        node.get("workload_metadata_config"), "node_config.workload_metadata_config"
    )
    if error or metadata is None or metadata.get("mode") != "GKE_METADATA":
        violations.append(f"{address}: Workload Metadata must be GKE_METADATA")
    taints = node.get("taint")
    expected_taint = {
        "key": "workload",
        "value": expected["taint_value"],
        "effect": "NO_SCHEDULE",
    }
    if not isinstance(taints, list) or expected_taint not in taints:
        violations.append(f"{address}: required workload taint is absent")
    return violations


def validate_plan(
    plan: Mapping[str, Any],
    mode: str,
    live_pool_specs: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return human-readable violations for a Terraform plan JSON object."""

    violations: list[str] = []
    for field in ("format_version", "terraform_version"):
        if not isinstance(plan.get(field), str) or not plan[field]:
            violations.append(f"plan.{field} must be a non-empty string")

    resource_drift = plan.get("resource_drift", [])
    if not isinstance(resource_drift, list):
        violations.append("plan.resource_drift must be a list when present")
    else:
        for resource in resource_drift:
            if not isinstance(resource, Mapping):
                violations.append("resource drift entry must be an object")
                continue
            address = str(resource.get("address", "<unknown>"))
            change = resource.get("change", {})
            actions = change.get("actions", []) if isinstance(change, Mapping) else []
            if tuple(actions) not in SAFE_PASSIVE_ACTIONS:
                violations.append(
                    f"{address}: resource drift {_format_actions(actions)} must be resolved first"
                )

    if "resource_changes" not in plan:
        violations.append("plan.resource_changes is required")
        return violations

    resource_changes = plan["resource_changes"]
    if not isinstance(resource_changes, list):
        violations.append("plan.resource_changes must be a list")
        return violations

    benchmark_pool_creates: set[str] = set()
    for resource in resource_changes:
        if not isinstance(resource, Mapping):
            violations.append("resource change must be an object")
            continue

        address = str(resource.get("address", "<unknown>"))
        resource_type = str(resource.get("type", ""))
        change = resource.get("change", {})
        if not isinstance(change, Mapping):
            violations.append(f"{address}: change must be an object")
            continue
        raw_actions = change.get("actions", [])
        if not isinstance(raw_actions, list) or not all(
            isinstance(action, str) for action in raw_actions
        ):
            violations.append(f"{address}: actions must be a string list")
            continue
        actions = tuple(raw_actions)

        if actions in SAFE_PASSIVE_ACTIONS:
            continue

        if mode == "no-op":
            violations.append(f"{address}: {_format_actions(actions)} is not allowed in no-op mode")
            continue

        if mode == "fresh-create":
            if actions == ("create",):
                continue
            violations.append(
                f"{address}: {_format_actions(actions)} is not allowed in fresh-create mode"
            )
            continue

        if mode in {"benchmark-pools-create", "benchmark-pool-recovery"}:
            if actions == ("create",) and address in BENCHMARK_POOL_CREATE_ADDRESSES:
                if resource_type != "google_container_node_pool":
                    violations.append(
                        f"{address}: benchmark pool create must be a google_container_node_pool"
                    )
                    continue
                if address in benchmark_pool_creates:
                    violations.append(f"{address}: duplicate benchmark pool create")
                    continue
                pool_violations = _validate_benchmark_pool(address, change.get("after"))
                violations.extend(pool_violations)
                if not pool_violations:
                    benchmark_pool_creates.add(address)
                continue
            violations.append(
                f"{address}: {_format_actions(actions)} is not an allowed benchmark pool create"
            )
            continue

        if mode == "protection":
            allowed_fields = PROTECTION_FIELDS.get(resource_type)
            if actions != ("update",) or allowed_fields is None:
                violations.append(
                    f"{address}: {_format_actions(actions)} is not an allowed protection update"
                )
                continue
            changed = _changed_top_level_fields(change.get("before"), change.get("after"))
            unexpected = changed - allowed_fields
            if not changed or unexpected:
                details = ", ".join(sorted(changed)) or "none"
                violations.append(f"{address}: changed fields are not protection-only: {details}")
                continue

            before = change.get("before")
            after = change.get("after")
            if not isinstance(before, Mapping) or not isinstance(after, Mapping):
                violations.append(f"{address}: protection update values must be objects")
                continue

            for field in changed:
                protected_value = PROTECTED_VALUES[(resource_type, field)]
                if after.get(field) != protected_value or before.get(field) == protected_value:
                    violations.append(
                        f"{address}: {field} must change toward protected value "
                        f"{protected_value!r}"
                    )
            continue

        violations.append(f"unsupported mode: {mode}")
        break

    if mode == "benchmark-pools-create":
        missing = BENCHMARK_POOL_CREATE_ADDRESSES - benchmark_pool_creates
        if missing:
            violations.append(
                "benchmark pool plan must create exactly the serving and loadgen pools; missing: "
                + ", ".join(sorted(missing))
            )
    elif mode == "benchmark-pool-recovery":
        if len(benchmark_pool_creates) != 1:
            violations.append("recovery plan must create exactly one missing benchmark pool")
        live_expected = BENCHMARK_POOL_CREATE_ADDRESSES - benchmark_pool_creates
        if not isinstance(live_pool_specs, Mapping) or set(live_pool_specs) != live_expected:
            violations.append("recovery requires the other live pool's exact canonical spec")
        else:
            for address in live_expected:
                violations.extend(_validate_benchmark_pool(address, live_pool_specs[address]))

    return violations


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan_json", type=Path, help="Path produced by terraform show -json")
    parser.add_argument(
        "--mode",
        choices=(
            "no-op",
            "protection",
            "fresh-create",
            "benchmark-pools-create",
            "benchmark-pool-recovery",
        ),
        required=True,
        help="Allowed change class for this plan",
    )
    parser.add_argument(
        "--live-pools-json",
        type=Path,
        help="Required only for one-pool benchmark recovery",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        with args.plan_json.open(encoding="utf-8") as stream:
            plan = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"terraform plan guard: cannot read plan JSON: {exc}", file=sys.stderr)
        return 1

    if not isinstance(plan, Mapping):
        print("terraform plan guard: top-level plan must be an object", file=sys.stderr)
        return 1

    live_pools = None
    if args.live_pools_json is not None:
        try:
            live_pools = json.loads(args.live_pools_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"terraform plan guard: cannot read live pool JSON: {exc}", file=sys.stderr)
            return 1
    if args.mode == "benchmark-pool-recovery" and live_pools is None:
        print("terraform plan guard: recovery mode requires --live-pools-json", file=sys.stderr)
        return 1
    violations = validate_plan(plan, args.mode, live_pools)
    if violations:
        print("Terraform plan rejected:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 2

    print(f"Terraform plan accepted in {args.mode} mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
