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
    address: str | None = None,
    before: dict[str, object] | None = None,
    after: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "address": address or f"module.gke.{resource_type}.this",
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


SERVING_POOL_ADDRESS = 'module.gke.google_container_node_pool.this["serving-pool"]'
LOADGEN_POOL_ADDRESS = 'module.gke.google_container_node_pool.this["loadgen-pool"]'


def _pool_change(pool_address: str, actions: list[str]) -> dict[str, object]:
    machine_type = "n2-standard-8" if pool_address == SERVING_POOL_ADDRESS else "n2-standard-4"
    taint_value = "pertinence-serving" if pool_address == SERVING_POOL_ADDRESS else "pertinence-loadgen"
    return _change(
        actions,
        resource_type="google_container_node_pool",
        address=pool_address,
        after={
            "initial_node_count": 0,
            "deletion_policy": "PREVENT",
            "autoscaling": [{"min_node_count": 0, "max_node_count": 1}],
            "node_config": [{
                "machine_type": machine_type,
                "image_type": "COS_CONTAINERD",
                "service_account": (
                    "pertinence-gke-nodes@pertinence-509116.iam.gserviceaccount.com"
                ),
                "spot": False,
                "disk_size_gb": 50,
                "disk_type": "pd-standard",
                "workload_metadata_config": [{"mode": "GKE_METADATA"}],
                "taint": [{"key": "workload", "value": taint_value, "effect": "NO_SCHEDULE"}],
            }],
        },
    )


def test_benchmark_pool_mode_accepts_exact_two_indexed_creates_and_passive_changes() -> None:
    plan = _plan(
        [
            _pool_change(SERVING_POOL_ADDRESS, ["create"]),
            _pool_change(LOADGEN_POOL_ADDRESS, ["create"]),
            _change(["no-op"]),
            _change(
                ["read"],
                resource_type="google_compute_network",
                address="module.gke.data.google_compute_network.selected",
            ),
        ]
    )

    assert validate_plan(plan, "benchmark-pools-create") == []


@pytest.mark.parametrize(
    ("address", "actions"),
    [
        ('module.gke.google_container_node_pool.this["pipeline-pool"]', ["create"]),
        ('module.gke.google_container_node_pool.this["system-pool"]', ["delete", "create"]),
        ("module.gke.google_container_cluster.this", ["update"]),
        ("module.gke.google_service_account.unexpected", ["create"]),
        (SERVING_POOL_ADDRESS, ["update"]),
        (SERVING_POOL_ADDRESS, ["delete"]),
        (SERVING_POOL_ADDRESS, ["delete", "create"]),
        (SERVING_POOL_ADDRESS, ["create", "delete"]),
        ("module.other.google_container_node_pool.this[\"serving-pool\"]", ["create"]),
    ],
)
def test_benchmark_pool_mode_rejects_unexpected_or_mutating_change(
    address: str, actions: list[str]
) -> None:
    plan = _plan(
        [
            _pool_change(address, actions),
            _pool_change(LOADGEN_POOL_ADDRESS, ["create"]),
        ]
    )

    violations = validate_plan(plan, "benchmark-pools-create")

    assert violations
    assert any("not an allowed benchmark pool create" in violation for violation in violations)


def test_benchmark_pool_mode_requires_both_pool_creates() -> None:
    plan = _plan([_pool_change(SERVING_POOL_ADDRESS, ["create"])])

    violations = validate_plan(plan, "benchmark-pools-create")

    assert violations == [
        "benchmark pool plan must create exactly the serving and loadgen pools; missing: "
        + LOADGEN_POOL_ADDRESS
    ]


def test_benchmark_pool_mode_rejects_noncanonical_pool_detail() -> None:
    serving = _pool_change(SERVING_POOL_ADDRESS, ["create"])
    serving["change"]["after"]["node_config"][0]["spot"] = True
    plan = _plan([serving, _pool_change(LOADGEN_POOL_ADDRESS, ["create"])])
    violations = validate_plan(plan, "benchmark-pools-create")
    assert any("spot" in violation for violation in violations)


def test_benchmark_pool_mode_rejects_similar_node_service_account() -> None:
    serving = _pool_change(SERVING_POOL_ADDRESS, ["create"])
    serving["change"]["after"]["node_config"][0]["service_account"] = (
        "pertinence-gke-nodes@other-project.iam.gserviceaccount.com"
    )
    plan = _plan([serving, _pool_change(LOADGEN_POOL_ADDRESS, ["create"])])

    violations = validate_plan(plan, "benchmark-pools-create")

    assert any("dedicated node SA" in violation for violation in violations)


def test_recovery_allows_one_create_only_when_other_live_pool_is_exact() -> None:
    create = _pool_change(SERVING_POOL_ADDRESS, ["create"])
    live = _pool_change(LOADGEN_POOL_ADDRESS, ["create"])["change"]["after"]
    assert validate_plan(
        _plan([create]),
        "benchmark-pool-recovery",
        {LOADGEN_POOL_ADDRESS: live},
    ) == []

    live["node_config"][0]["disk_size_gb"] = 100
    assert validate_plan(
        _plan([create]),
        "benchmark-pool-recovery",
        {LOADGEN_POOL_ADDRESS: live},
    )


def test_benchmark_pool_mode_rejects_wrong_resource_type_at_allowed_address() -> None:
    plan = _plan(
        [
            _change(["create"], resource_type="other", address=SERVING_POOL_ADDRESS),
            _pool_change(LOADGEN_POOL_ADDRESS, ["create"]),
        ]
    )

    violations = validate_plan(plan, "benchmark-pools-create")

    assert violations == [
        f"{SERVING_POOL_ADDRESS}: benchmark pool create must be a google_container_node_pool",
        "benchmark pool plan must create exactly the serving and loadgen pools; missing: "
        + SERVING_POOL_ADDRESS,
    ]


def test_benchmark_pool_mode_rejects_duplicate_indexed_address() -> None:
    plan = _plan(
        [
            _pool_change(SERVING_POOL_ADDRESS, ["create"]),
            _pool_change(SERVING_POOL_ADDRESS, ["create"]),
            _pool_change(LOADGEN_POOL_ADDRESS, ["create"]),
        ]
    )

    assert validate_plan(plan, "benchmark-pools-create") == [
        f"{SERVING_POOL_ADDRESS}: duplicate benchmark pool create"
    ]


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


def test_benchmark_pool_mode_rejects_resource_drift() -> None:
    plan = _plan(
        [
            _pool_change(SERVING_POOL_ADDRESS, ["create"]),
            _pool_change(LOADGEN_POOL_ADDRESS, ["create"]),
        ]
    )
    plan["resource_drift"] = [
        _change(
            ["update"],
            resource_type="google_container_node_pool",
            address='module.gke.google_container_node_pool.this["system-pool"]',
        )
    ]

    violations = validate_plan(plan, "benchmark-pools-create")

    assert any("resource drift" in violation for violation in violations)
