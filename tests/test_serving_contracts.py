import json

import pytest

from pertinence.serving import (
    APPROVED_DISPATCHER_SHA256,
    BenchmarkTraceV1,
    PromotionConflictError,
    PromotionManifestV1,
    create_promotion,
    seal_manifest,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def promotion(**updates):
    value = {
        "schema_version": 1,
        "promotion_id": "cifar10-20260920",
        "evaluated_bundle": {
            "uri": "gs://runs/evaluated.tar",
            "generation": 123,
            "fingerprint": SHA_A,
        },
        "lineage": {
            "contract_fingerprint": SHA_A,
            "search_fingerprint": SHA_B,
            "final_fingerprint": SHA_C,
        },
        "dispatcher": {
            "id": "dispatcher-000733",
            "checkpoint_sha256": APPROVED_DISPATCHER_SHA256,
            "feature_dim": 1024,
            "expert_count": 3,
            "modelcar_image": f"registry/dispatcher@sha256:{SHA_C}",
        },
        "experts": [
            {"id": "shuffle", "model_sha256": SHA_A, "modelcar_image": f"registry/s@sha256:{SHA_A}"},
            {"id": "mobile", "model_sha256": SHA_B, "modelcar_image": f"registry/m@sha256:{SHA_B}"},
            {"id": "resnet", "model_sha256": SHA_C, "modelcar_image": f"registry/r@sha256:{SHA_C}"},
        ],
        "runtime_image": f"registry/runtime@sha256:{SHA_A}",
        "approved_by": "reviewer@example.com",
        "approved_at": "2026-09-20T12:34:56Z",
    }
    value.update(updates)
    return seal_manifest(value)


def test_promotion_is_sealed_and_create_is_idempotent(tmp_path) -> None:
    value = promotion()
    parsed = PromotionManifestV1.from_dict(value)
    path = tmp_path / "promotion.json"

    assert create_promotion(path, parsed)
    assert not create_promotion(path, value)
    assert json.loads(path.read_text()) == value


def test_promotion_conflict_never_replaces_existing_file(tmp_path) -> None:
    path = tmp_path / "promotion.json"
    original = promotion()
    create_promotion(path, original)
    changed = promotion(approved_at="2026-09-20T12:35:00Z")

    with pytest.raises(PromotionConflictError, match="different content"):
        create_promotion(path, changed)

    assert json.loads(path.read_text()) == original


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda body: body["dispatcher"].update(id="dispatcher-000267"), "only dispatcher"),
        (lambda body: body["dispatcher"].update(feature_dim=1000), "feature_dim"),
        (lambda body: body["dispatcher"].update(checkpoint_sha256=SHA_A), "approved checkpoint"),
        (lambda body: body.update(runtime_image="registry/runtime:latest"), "pinned"),
        (lambda body: body.update(approved_by=""), "non-empty identity"),
        (lambda body: body["experts"][0].update(model_sha256="ABC"), "SHA-256"),
    ],
)
def test_promotion_rejects_unapproved_identity_and_unpinned_artifacts(mutation, message) -> None:
    body = promotion()
    body.pop("fingerprint")
    mutation(body)
    with pytest.raises(ValueError, match=message):
        PromotionManifestV1.from_dict(seal_manifest(body))


def test_trace_contains_hashes_and_order_but_cannot_contain_labels() -> None:
    body = {
        "schema_version": 1,
        "kind": "benchmark-trace",
        "trace_id": "cifar10-1-v6",
        "dataset": {"id": "cifar10-1", "version": "v6"},
        "samples": 2,
        "input": {"name": "image", "datatype": "FP32", "shape": [1, 3, 32, 32]},
        "source_objects": [
            {"uri": "gs://bench/data.npy", "sha256": SHA_A, "size_bytes": 1, "generation": 7}
        ],
        "provenance": "CIFAR-10.1 v6, Tiny Images lineage",
        "usage_conditions": "recorded separately from CIFAR-10 terms",
        "sample_order_sha256": SHA_B,
        "label_sha256": SHA_C,
        "image_sha256": SHA_A,
        "class_counts": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "payload": "trace.npz",
    }
    BenchmarkTraceV1.from_dict(seal_manifest(body))

    body["labels"] = [1, 2]
    with pytest.raises(ValueError, match="unknown=.*labels"):
        BenchmarkTraceV1.from_dict(seal_manifest(body))
