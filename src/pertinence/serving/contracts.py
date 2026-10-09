"""Immutable promotion and benchmark contracts for online serving."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit

from ..contracts.common import Manifest, digest, fields, integer, name, number, version
from ..reproducibility import canonical_json_hash


APPROVED_DISPATCHER_ID = "dispatcher-000733"
APPROVED_DISPATCHER_SHA256 = (
    "d82d3243bfec4f0c5889aa7e241735c4f212c849a6bcc2f461f60ca6c1524720"
)
APPROVED_FEATURE_DIM = 1024
APPROVED_EXPERT_COUNT = 3
APPROVED_BENCHMARK_ARMS = (
    "resnet56",
    "dispatcher-000733",
)


class PromotionConflictError(RuntimeError):
    """A promotion ID already exists with different sealed content."""


def _timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an RFC 3339 UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} must be an RFC 3339 UTC timestamp ending in Z") from error
    return value


def _uri(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"(?:gs|oci)://[^\s]+", value):
        raise ValueError(f"{label} must be an immutable gs:// or oci:// URI")
    parsed = urlsplit(value)
    if (
        not parsed.netloc
        or "\\" in value
        or any(segment in {"", ".", ".."} for segment in parsed.path.split("/")[1:])
    ):
        raise ValueError(f"{label} contains an unsafe path")
    return value


def _pinned_image(value: Any, label: str) -> str:
    if not isinstance(value, str) or "@sha256:" not in value:
        raise ValueError(f"{label} must be pinned by sha256 digest")
    repository, separator, image_digest = value.rpartition("@sha256:")
    if not separator or not repository:
        raise ValueError(f"{label} must be pinned by sha256 digest")
    digest(image_digest)
    return value


def _sealed_body(value: dict[str, Any]) -> None:
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    if value.get("fingerprint") != canonical_json_hash(body):
        raise ValueError("manifest fingerprint mismatch")


def seal_manifest(value: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied manifest with its canonical fingerprint populated."""
    body = json.loads(json.dumps(value, allow_nan=False))
    if "fingerprint" in body:
        raise ValueError("cannot seal a manifest that already contains fingerprint")
    body["fingerprint"] = canonical_json_hash(body)
    return body


class PromotionManifestV1(Manifest):
    """The sole authority for promoting the evaluated dispatcher into serving."""

    @staticmethod
    def validate(value: dict) -> None:
        fields(
            value,
            {
                "schema_version",
                "promotion_id",
                "evaluated_bundle",
                "lineage",
                "dispatcher",
                "experts",
                "runtime_image",
                "approved_by",
                "approved_at",
                "fingerprint",
            },
        )
        version(value)
        name(value["promotion_id"], "promotion_id")

        bundle = value["evaluated_bundle"]
        fields(bundle, {"uri", "generation", "fingerprint"})
        _uri(bundle["uri"], "evaluated bundle URI")
        integer(bundle["generation"], "evaluated bundle generation", 0)
        digest(bundle["fingerprint"])

        lineage = value["lineage"]
        fields(lineage, {"contract_fingerprint", "search_fingerprint", "final_fingerprint"})
        for fingerprint in lineage.values():
            digest(fingerprint)

        dispatcher = value["dispatcher"]
        fields(
            dispatcher,
            {"id", "checkpoint_sha256", "feature_dim", "expert_count", "modelcar_image"},
        )
        if dispatcher["id"] != APPROVED_DISPATCHER_ID:
            raise ValueError(f"only {APPROVED_DISPATCHER_ID} may be promoted")
        digest(dispatcher["checkpoint_sha256"])
        if dispatcher["checkpoint_sha256"] != APPROVED_DISPATCHER_SHA256:
            raise ValueError("dispatcher checkpoint SHA-256 is not the approved checkpoint")
        if dispatcher["feature_dim"] != APPROVED_FEATURE_DIM:
            raise ValueError(f"dispatcher feature_dim must be {APPROVED_FEATURE_DIM}")
        if dispatcher["expert_count"] != APPROVED_EXPERT_COUNT:
            raise ValueError(f"dispatcher expert_count must be {APPROVED_EXPERT_COUNT}")
        _pinned_image(dispatcher["modelcar_image"], "dispatcher modelcar image")

        experts = value["experts"]
        if not isinstance(experts, list) or len(experts) != APPROVED_EXPERT_COUNT:
            raise ValueError(f"experts must contain exactly {APPROVED_EXPERT_COUNT} ordered entries")
        expert_ids: list[str] = []
        for expert in experts:
            fields(expert, {"id", "model_sha256", "modelcar_image"})
            expert_ids.append(name(expert["id"], "expert id"))
            digest(expert["model_sha256"])
            _pinned_image(expert["modelcar_image"], f"{expert['id']} modelcar image")
        if len(expert_ids) != len(set(expert_ids)):
            raise ValueError("expert IDs must be unique and ordered")

        _pinned_image(value["runtime_image"], "runtime image")
        if not isinstance(value["approved_by"], str) or not value["approved_by"].strip():
            raise ValueError("approved_by must be a non-empty identity")
        _timestamp(value["approved_at"], "approved_at")
        _sealed_body(value)


class BenchmarkTraceV1(Manifest):
    """Ordered benchmark inputs and hashes, deliberately excluding labels."""

    @staticmethod
    def validate(value: dict) -> None:
        fields(
            value,
            {
                "schema_version",
                "kind",
                "trace_id",
                "dataset",
                "samples",
                "input",
                "source_objects",
                "provenance",
                "usage_conditions",
                "sample_order_sha256",
                "label_sha256",
                "image_sha256",
                "class_counts",
                "payload",
                "fingerprint",
            },
        )
        version(value)
        if value["kind"] != "benchmark-trace":
            raise ValueError("trace kind must be benchmark-trace")
        name(value["trace_id"], "trace_id")
        dataset = value["dataset"]
        fields(dataset, {"id", "version"})
        name(dataset["id"], "dataset id")
        name(dataset["version"], "dataset version")
        samples = integer(value["samples"], "samples")
        input_spec = value["input"]
        fields(input_spec, {"name", "datatype", "shape"})
        if input_spec != {
            "name": "image",
            "datatype": "FP32",
            "shape": [1, 3, 32, 32],
        }:
            raise ValueError("trace input must be FP32 image [1,3,32,32]")
        sources = value["source_objects"]
        if not isinstance(sources, list) or not sources:
            raise ValueError("source_objects must be a non-empty list")
        for source in sources:
            allowed = {"uri", "sha256", "size_bytes", "generation", "revision"}
            if not isinstance(source, dict):
                raise ValueError("source object must be a mapping")
            unknown = set(source) - allowed
            missing = {"uri", "sha256", "size_bytes"} - set(source)
            if unknown or missing:
                raise ValueError(
                    f"source object fields differ; missing={sorted(missing)}, unknown={sorted(unknown)}"
                )
            if ("generation" in source) == ("revision" in source):
                raise ValueError("source object requires exactly one of generation or revision")
            if not isinstance(source["uri"], str) or not source["uri"].strip():
                raise ValueError("source object URI must be non-empty")
            digest(source["sha256"])
            integer(source["size_bytes"], "source object size_bytes", 0)
            if "generation" in source:
                integer(source["generation"], "source object generation")
            else:
                revision = source["revision"]
                if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{7,64}", revision):
                    raise ValueError("source object revision must be a commit-like hex revision")
        if not isinstance(value["provenance"], str) or not value["provenance"].strip():
            raise ValueError("dataset provenance must be non-empty")
        if not isinstance(value["usage_conditions"], str) or not value["usage_conditions"].strip():
            raise ValueError("dataset usage_conditions must be non-empty")
        digest(value["sample_order_sha256"])
        digest(value["label_sha256"])
        digest(value["image_sha256"])
        counts = value["class_counts"]
        if (
            not isinstance(counts, list)
            or len(counts) != 10
            or any(type(count) is not int or count < 0 for count in counts)
            or sum(counts) != samples
        ):
            raise ValueError("class_counts must contain ten counts totaling samples")
        payload = value["payload"]
        if (
            not isinstance(payload, str)
            or not payload
            or payload.startswith("/")
            or "\\" in payload
            or any(part in {"", ".", ".."} for part in payload.split("/"))
        ):
            raise ValueError("trace payload contains an unsafe path")
        _sealed_body(value)


class BenchmarkRunManifestV1(Manifest):
    """Immutable execution settings for a low-load paired benchmark."""

    @staticmethod
    def validate(value: dict) -> None:
        fields(
            value,
            {
                "schema_version",
                "benchmark_id",
                "promotion_fingerprint",
                "trace_fingerprints",
                "arms",
                "warmup_requests",
                "requests_per_arm",
                "target_rps",
                "concurrency",
                "seed",
                "created_at",
                "fingerprint",
            },
        )
        version(value)
        name(value["benchmark_id"], "benchmark_id")
        digest(value["promotion_fingerprint"])
        traces = value["trace_fingerprints"]
        if not isinstance(traces, dict) or not traces:
            raise ValueError("trace_fingerprints must be a non-empty mapping")
        for trace_id, fingerprint in traces.items():
            name(trace_id, "trace id")
            digest(fingerprint)
        arms = value["arms"]
        if not isinstance(arms, list) or len(arms) != 2:
            raise ValueError("benchmark must contain exactly two arms")
        for arm in arms:
            name(arm, "benchmark arm")
        if len(arms) != len(set(arms)):
            raise ValueError("benchmark arms must be unique")
        if arms != list(APPROVED_BENCHMARK_ARMS):
            raise ValueError("benchmark arms must use the exact approved order")
        integer(value["warmup_requests"], "warmup_requests", 0)
        integer(value["requests_per_arm"], "requests_per_arm")
        if value["target_rps"] != 2.0:
            raise ValueError("benchmark target_rps must be exactly 2")
        number(value["target_rps"], "target_rps", 1e-12)
        if value["concurrency"] != 1:
            raise ValueError("benchmark concurrency must be exactly 1")
        integer(value["seed"], "seed", 0)
        _timestamp(value["created_at"], "created_at")
        _sealed_body(value)


class BenchmarkSummaryV1(Manifest):
    """Sealed result summary; raw observations remain separate artifacts."""

    @staticmethod
    def validate(value: dict) -> None:
        fields(
            value,
            {
                "schema_version",
                "benchmark_id",
                "run_fingerprint",
                "valid",
                "invalid_reasons",
                "environment",
                "accuracy",
                "performance",
                "created_at",
                "fingerprint",
            },
        )
        version(value)
        name(value["benchmark_id"], "benchmark_id")
        digest(value["run_fingerprint"])
        if type(value["valid"]) is not bool:
            raise ValueError("valid must be boolean")
        reasons = value["invalid_reasons"]
        if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
            raise ValueError("invalid_reasons must be a list of strings")
        if value["valid"] and reasons:
            raise ValueError("a valid run cannot have invalid reasons")
        for key in ("environment", "accuracy", "performance"):
            if not isinstance(value[key], dict):
                raise ValueError(f"{key} must be a mapping")
        _timestamp(value["created_at"], "created_at")
        _sealed_body(value)


def create_promotion(path: Path, value: dict[str, Any] | PromotionManifestV1) -> bool:
    """Create a promotion once, returning false for an identical existing manifest.

    The caller chooses the destination, while this function enforces a regular JSON
    file and refuses symlink targets. Existing bytes are never replaced.
    """
    path = Path(path)
    if path.is_symlink():
        raise ValueError("promotion path must not be a symlink")
    manifest = value if isinstance(value, PromotionManifestV1) else PromotionManifestV1.from_dict(value)
    encoded = json.dumps(manifest.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    except FileExistsError:
        try:
            existing = PromotionManifestV1.from_text(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise PromotionConflictError(f"existing promotion is invalid: {path}") from error
        if existing.to_dict() == manifest.to_dict():
            return False
        raise PromotionConflictError(f"promotion ID already has different content: {path}")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # The file remains as evidence if a write fails; never replace a possibly
        # observed promotion with a second attempt.
        raise
    return True
