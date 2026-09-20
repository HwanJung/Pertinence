"""Tests for the non-destructive Terraform plan guard."""

from __future__ import annotations

import pytest

from scripts.terraform_plan_guard import validate_plan


def _plan(resource_changes: object) -> dict[str, object]:
    return {
        "format_version": "1.2",
        "terraform_version": "1.16.3",
        "resource_changes": resource_changes,
    }


def _change(
    actions: list[str],
    *,
    resource_type: str = "google_container_cluster",
    before: dict[str, object] | None = None,
    after: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "address": f"module.gke.{resource_type}.this",
        "type": resource_type,
        "change": {
            "actions": actions,
            "before": before,
            "after": after,
        },
    }


@pytest.mark.parametrize("actions", [["no-op"], ["read"]])
def test_no_op_mode_accepts_passive_actions(actions: list[str]) -> None:
    plan = _plan([_change(actions)])

    assert validate_plan(plan, "no-op") == []


@pytest.mark.parametrize("actions", [["create"], ["update"], ["delete"], ["delete", "create"]])
def test_no_op_mode_rejects_mutations(actions: list[str]) -> None:
    plan = _plan([_change(actions)])

    assert validate_plan(plan, "no-op")


def test_fresh_create_mode_allows_only_creation_and_passive_actions() -> None:
    accepted = _plan([_change(["create"]), _change(["read"])])
    rejected = _plan([_change(["create", "delete"])])

    assert validate_plan(accepted, "fresh-create") == []
    assert validate_plan(rejected, "fresh-create")


@pytest.mark.parametrize(
    ("resource_type", "field"),
    [
        ("google_container_cluster", "deletion_protection"),
        ("google_container_node_pool", "deletion_policy"),
    ],
)
def test_protection_mode_accepts_only_protection_field(
    resource_type: str, field: str
) -> None:
    plan = _plan(
        [
            _change(
                ["update"],
                resource_type=resource_type,
                before={field: False if field == "deletion_protection" else "DELETE"},
                after={field: True if field == "deletion_protection" else "PREVENT"},
            )
        ]
    )

    assert validate_plan(plan, "protection") == []


def test_protection_mode_rejects_mixed_update() -> None:
    plan = _plan(
        [
            _change(
                ["update"],
                before={"deletion_protection": False, "network": "default"},
                after={"deletion_protection": True, "network": "other"},
            )
        ]
    )

    violations = validate_plan(plan, "protection")

    assert violations
    assert "network" in violations[0]


def test_protection_mode_rejects_delete() -> None:
    plan = _plan([_change(["delete"])])

    assert validate_plan(plan, "protection")


def test_invalid_resource_changes_shape_is_rejected() -> None:
    assert validate_plan(_plan({}), "no-op") == ["plan.resource_changes must be a list"]


@pytest.mark.parametrize(
    ("resource_type", "field", "before", "after"),
    [
        ("google_container_cluster", "deletion_protection", True, False),
        ("google_container_node_pool", "deletion_policy", "PREVENT", "DELETE"),
    ],
)
def test_protection_mode_rejects_protection_removal(
    resource_type: str, field: str, before: object, after: object
) -> None:
    plan = _plan(
        [
            _change(
                ["update"],
                resource_type=resource_type,
                before={field: before},
                after={field: after},
            )
        ]
    )

    violations = validate_plan(plan, "protection")

    assert violations
    assert "protected value" in violations[0]


@pytest.mark.parametrize(
    "plan",
    [
        {},
        {"resource_changes": []},
        {"format_version": "1.2", "terraform_version": "1.16.3"},
    ],
)
def test_malformed_plan_is_rejected(plan: dict[str, object]) -> None:
    assert validate_plan(plan, "no-op")
