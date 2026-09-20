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
PROTECTION_FIELDS = {
    "google_container_cluster": {"deletion_protection"},
    "google_container_node_pool": {"deletion_policy"},
}
PROTECTED_VALUES = {
    ("google_container_cluster", "deletion_protection"): True,
    ("google_container_node_pool", "deletion_policy"): "PREVENT",
}


def _changed_top_level_fields(before: Any, after: Any) -> set[str]:
    """Return changed top-level object keys, treating non-objects as one change."""

    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return {"<value>"} if before != after else set()
    keys = set(before) | set(after)
    return {key for key in keys if before.get(key) != after.get(key)}


def _format_actions(actions: Sequence[str]) -> str:
    return "/".join(actions) if actions else "unknown"


def validate_plan(plan: Mapping[str, Any], mode: str) -> list[str]:
    """Return human-readable violations for a Terraform plan JSON object."""

    violations: list[str] = []
    for field in ("format_version", "terraform_version"):
        if not isinstance(plan.get(field), str) or not plan[field]:
            violations.append(f"plan.{field} must be a non-empty string")

    if "resource_changes" not in plan:
        violations.append("plan.resource_changes is required")
        return violations

    resource_changes = plan["resource_changes"]
    if not isinstance(resource_changes, list):
        violations.append("plan.resource_changes must be a list")
        return violations

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

    return violations


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan_json", type=Path, help="Path produced by terraform show -json")
    parser.add_argument(
        "--mode",
        choices=("no-op", "protection", "fresh-create"),
        required=True,
        help="Allowed change class for this plan",
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

    violations = validate_plan(plan, args.mode)
    if violations:
        print("Terraform plan rejected:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 2

    print(f"Terraform plan accepted in {args.mode} mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
