"""Command-line entry points for immutable promotions and KServe benchmarks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from .pipeline_artifacts import write_json
from .reproducibility import sha256_file
from .serving.analysis import analyze_records, derive_validity, write_parquet
from .serving.benchmark import (
    APPROVED_ARMS,
    KServeV2Predictor,
    calibrate,
    latin_square,
    records_to_jsonl,
    run_scheduled_arm,
    validate_benchmark_config,
)
from .serving.contracts import (
    BenchmarkRunManifestV1,
    BenchmarkSummaryV1,
    PromotionManifestV1,
    create_promotion,
    seal_manifest,
)
from .serving.trace import (
    build_cifar101_v6_trace,
    build_final_trace_from_npz,
    create_trace,
    source_object,
    verify_trace,
)


def _key_values(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item or key in result:
            raise ValueError(f"{label} entries must be unique NAME=VALUE pairs")
        result[key] = item
    return result


def _promotion_commands(subparsers: argparse._SubParsersAction) -> None:
    promotion = subparsers.add_parser("promotion", help="create or verify a serving promotion")
    commands = promotion.add_subparsers(dest="promotion_command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--manifest", type=Path, required=True, help="JSON draft or sealed manifest")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--evaluated-bundle", type=Path, required=True)
    create.set_defaults(serving_action="promotion-create")
    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--evaluated-bundle", type=Path, required=True)
    verify.set_defaults(serving_action="promotion-verify")


def _trace_commands(subparsers: argparse._SubParsersAction) -> None:
    trace = subparsers.add_parser("trace", help="create or verify an immutable benchmark trace")
    commands = trace.add_subparsers(dest="trace_command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--input", type=Path, required=True, help="NPZ with images/labels/sample_ids")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--trace-id", required=True)
    create.add_argument("--dataset-id", required=True)
    create.add_argument("--dataset-version", required=True)
    create.add_argument("--source-uri")
    provenance_group = create.add_mutually_exclusive_group(required=True)
    provenance_group.add_argument("--source-generation", type=int)
    provenance_group.add_argument("--source-revision")
    create.add_argument("--provenance", required=True)
    create.add_argument("--usage-conditions", required=True)
    create.add_argument("--require-class-balanced", action="store_true")
    create.set_defaults(serving_action="trace-create")
    final = commands.add_parser("create-final")
    final.add_argument("--evaluation-npz", type=Path, required=True)
    final.add_argument("--split-manifest", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--trace-id", default="cifar10-final-20260824")
    final.add_argument("--expected-label-sha256", required=True)
    final.add_argument("--evaluation-uri", required=True)
    final.add_argument("--evaluation-generation", type=int, required=True)
    final.add_argument("--split-uri", required=True)
    final.add_argument("--split-generation", type=int, required=True)
    final.set_defaults(serving_action="trace-create-final")
    cifar101 = commands.add_parser("create-cifar10.1-v6")
    cifar101.add_argument("--data-npy", type=Path, required=True)
    cifar101.add_argument("--labels-npy", type=Path, required=True)
    cifar101.add_argument("--output", type=Path, required=True)
    cifar101.add_argument("--trace-id", default="cifar10.1-v6")
    cifar101.set_defaults(serving_action="trace-create-cifar101")
    publish = commands.add_parser("publish")
    publish.add_argument("--trace", type=Path, required=True)
    publish.add_argument("--uri", required=True)
    publish.set_defaults(serving_action="trace-publish")
    download = commands.add_parser("download")
    download.add_argument("--uri", required=True)
    download.add_argument("--output", type=Path, required=True)
    download.set_defaults(serving_action="trace-download")
    verify = commands.add_parser("verify")
    verify.add_argument("--trace", type=Path, required=True)
    verify.set_defaults(serving_action="trace-verify")


def _serve_commands(subparsers: argparse._SubParsersAction) -> None:
    serve_parser = subparsers.add_parser("serve", help="run the single-worker KServe V2 runtime")
    commands = serve_parser.add_subparsers(dest="serve_command", required=True)
    expert = commands.add_parser("expert")
    expert.add_argument("--model-name", required=True)
    expert.add_argument("--model-path", type=Path, required=True)
    expert.add_argument("--model-sha256", required=True)
    expert.add_argument("--features-name")
    expert.add_argument("--feature-dim", type=int)
    expert.add_argument("--host", default="0.0.0.0")
    expert.add_argument("--port", type=int, default=8080)
    expert.set_defaults(serving_action="serve-expert")

    router = commands.add_parser("router")
    router.add_argument("--promotion", type=Path, required=True)
    router.add_argument("--bundle-path", type=Path)
    router.add_argument("--router-mode", choices=("dynamic", "static-resnet"), required=True)
    router.add_argument("--expert-url", action="append", default=[], metavar="NAME=URL")
    router.add_argument("--timeout-seconds", type=float, default=10.0)
    router.add_argument("--host", default="0.0.0.0")
    router.add_argument("--port", type=int, default=8080)
    router.set_defaults(serving_action="serve-router")


def _benchmark_commands(subparsers: argparse._SubParsersAction) -> None:
    benchmark = subparsers.add_parser("benchmark", help="run or analyze low-load paired traffic")
    commands = benchmark.add_subparsers(dest="benchmark_command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--trace", type=Path, required=True)
    run.add_argument("--trace-uri")
    run.add_argument("--promotion", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--endpoint", action="append", default=[], metavar="ARM=URL")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--output-uri")
    run.add_argument("--collector-done", type=Path)
    run.add_argument("--collector-completed", type=Path)
    run.add_argument("--warmup", type=int, default=200)
    run.add_argument("--calibration", type=int, default=200)
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--timeout-seconds", type=float, default=10.0)
    run.add_argument("--seed", type=int, default=20260920)
    run.set_defaults(serving_action="benchmark-run")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--records", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--baseline-arm", default="static-resnet-router")
    analyze.add_argument("--comparison-arm", default="dispatcher-000733")
    analyze.add_argument("--repeats", type=int, default=5)
    analyze.add_argument("--samples", type=int, default=2000)
    analyze.add_argument("--seed", type=int, default=20260920)
    analyze.add_argument("--bootstrap-samples", type=int, default=10_000)
    analyze.add_argument("--run-manifest", type=Path)
    analyze.add_argument("--environment-manifest", type=Path)
    analyze.set_defaults(serving_action="benchmark-analyze")
    collect = commands.add_parser("collect-evidence")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--done", type=Path, required=True)
    collect.add_argument("--completed", type=Path, required=True)
    collect.set_defaults(serving_action="benchmark-collect")
    parity = commands.add_parser("parity")
    parity.add_argument("--trace", type=Path, required=True)
    parity.add_argument("--local-predictions", type=Path, required=True)
    parity.add_argument("--online-records", type=Path, required=True)
    parity.add_argument("--output", type=Path, required=True)
    parity.add_argument("--promotion-fingerprint")
    parity.add_argument("--image-digest", action="append", default=[], metavar="NAME=SHA256")
    parity.add_argument("--golden-logits-atol", type=float)
    parity.add_argument("--golden-sample-index", type=int, default=0)
    parity.set_defaults(serving_action="benchmark-parity")
    parity_collect = commands.add_parser("parity-collect")
    parity_collect.add_argument("--trace", type=Path, required=True)
    parity_collect.add_argument("--arm", choices=APPROVED_ARMS, required=True)
    parity_collect.add_argument("--endpoint", required=True)
    parity_collect.add_argument(
        "--expected-identity", action="append", default=[], metavar="NAME=VALUE"
    )
    parity_collect.add_argument("--output", type=Path, required=True)
    parity_collect.add_argument("--timeout-seconds", type=float, default=10.0)
    parity_collect.set_defaults(serving_action="benchmark-parity-collect")
    parity_combine = commands.add_parser("parity-combine")
    parity_combine.add_argument("--trace", type=Path, required=True)
    parity_combine.add_argument("--input", action="append", default=[], metavar="ARM=PATH")
    parity_combine.add_argument("--output", type=Path, required=True)
    parity_combine.set_defaults(serving_action="benchmark-parity-combine")
    evidence = commands.add_parser("validate-evidence")
    evidence.add_argument("--request-validity", type=Path, required=True)
    evidence.add_argument("--parity-receipt", type=Path, required=True)
    evidence.add_argument("--snapshot", type=Path, required=True)
    evidence.add_argument("--metric-summary", type=Path, required=True)
    evidence.add_argument("--output", type=Path, required=True)
    evidence.set_defaults(serving_action="benchmark-validate-evidence")
    metrics = commands.add_parser("summarize-evidence")
    metrics.add_argument("--evidence", type=Path, required=True)
    metrics.add_argument("--requests", type=Path, required=True)
    metrics.add_argument("--markers", type=Path)
    metrics.add_argument("--output", type=Path, required=True)
    metrics.set_defaults(serving_action="benchmark-summarize-evidence")


def add_commands(subparsers: argparse._SubParsersAction) -> None:
    _promotion_commands(subparsers)
    _trace_commands(subparsers)
    _serve_commands(subparsers)
    _benchmark_commands(subparsers)


def _load_promotion(path: Path) -> dict[str, Any]:
    return PromotionManifestV1.from_text(path.read_text(encoding="utf-8")).to_dict()


def _run_promotion(args: argparse.Namespace) -> int:
    from .serving.promotion import validate_promotion_bundle

    value = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.serving_action == "promotion-create":
        if "fingerprint" not in value:
            value = seal_manifest(value)
        validate_promotion_bundle(value, args.evaluated_bundle)
        created = create_promotion(args.output, value)
        print("created" if created else "already exists with identical content")
    else:
        manifest = PromotionManifestV1.from_dict(value)
        validate_promotion_bundle(manifest, args.evaluated_bundle)
        print(manifest.to_dict()["fingerprint"])
    return 0


def _run_trace(args: argparse.Namespace) -> int:
    if args.serving_action == "trace-create-final":
        manifest = build_final_trace_from_npz(
            args.evaluation_npz,
            args.split_manifest,
            args.output,
            expected_label_sha256=args.expected_label_sha256,
            evaluation_uri=args.evaluation_uri,
            evaluation_generation=args.evaluation_generation,
            split_uri=args.split_uri,
            split_generation=args.split_generation,
            trace_id=args.trace_id,
        )
        print(manifest["fingerprint"])
        return 0
    if args.serving_action == "trace-create-cifar101":
        manifest = build_cifar101_v6_trace(
            args.data_npy, args.labels_npy, args.output, trace_id=args.trace_id
        )
        print(manifest["fingerprint"])
        return 0
    if args.serving_action == "trace-publish":
        from .serving.gcs import publish_create_only

        verify_trace(args.trace)
        publish_create_only(args.trace, args.uri, kind="trace")
        return 0
    if args.serving_action == "trace-download":
        from .serving.gcs import download_trace

        download_trace(args.uri, args.output)
        print(verify_trace(args.output)[0]["fingerprint"])
        return 0
    if args.serving_action == "trace-verify":
        manifest, _, _, _ = verify_trace(args.trace)
        print(manifest["fingerprint"])
        return 0
    with np.load(args.input, allow_pickle=False) as archive:
        if set(archive.files) != {"images", "labels", "sample_ids"}:
            raise ValueError("trace input NPZ must contain exactly images, labels, sample_ids")
        images = archive["images"]
        labels = archive["labels"]
        sample_ids = archive["sample_ids"].tolist()
    if args.require_class_balanced:
        counts = np.bincount(labels, minlength=10)
        if len(counts) != 10 or len(set(counts.tolist())) != 1:
            raise ValueError("trace is not class balanced")
    manifest = create_trace(
        args.output,
        trace_id=args.trace_id,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        images=images,
        labels=labels,
        sample_ids=sample_ids,
        source_objects=[
            source_object(
                args.input,
                uri=args.source_uri,
                generation=args.source_generation,
                revision=args.source_revision,
            )
        ],
        provenance=args.provenance,
        usage_conditions=args.usage_conditions,
    )
    print(manifest["fingerprint"])
    return 0


def _load_dispatcher(path: Path):
    from .serving.runtime import load_dispatcher_npz

    return load_dispatcher_npz(path)


def _run_serve(args: argparse.Namespace) -> int:
    from .serving.http import ExpertModel, V2Application, serve
    from .serving.runtime import (
        DynamicRouter,
        HttpExpertClient,
        OnnxExpertRunner,
        StaticRouter,
    )

    if args.serving_action == "serve-expert":
        runner = OnnxExpertRunner(
            args.model_path,
            args.model_sha256,
            features_name=args.features_name,
            feature_dim=args.feature_dim,
        )
        model = ExpertModel(args.model_name, runner, feature_dim=args.feature_dim)
        serve(V2Application(model), args.host, args.port)
        return 0

    promotion = _load_promotion(args.promotion)
    experts = [item["id"] for item in promotion["experts"]]
    endpoints = _key_values(args.expert_url, "expert-url")
    required_experts = {experts[-1]} if args.router_mode == "static-resnet" else set(experts)
    if set(endpoints) != required_experts:
        raise ValueError("expert URLs must exactly match the router's required promoted experts")
    clients = {
        name: HttpExpertClient(
            endpoint,
            name,
            timeout_seconds=args.timeout_seconds,
            feature_dim=1024 if name == experts[0] else None,
        )
        for name, endpoint in ((name, endpoints[name]) for name in experts if name in endpoints)
    }
    if args.router_mode == "static-resnet":
        model = StaticRouter(
            "static-resnet-router",
            promotion["promotion_id"],
            experts[-1],
            len(experts) - 1,
            clients[experts[-1]],
        )
    else:
        from .serving.promotion import validate_promotion_bundle

        if args.bundle_path is None:
            raise ValueError("dynamic router requires --bundle-path")
        verified = validate_promotion_bundle(promotion, args.bundle_path)
        checkpoint = args.bundle_path / verified["checkpoint"]
        expected = promotion["dispatcher"]["checkpoint_sha256"]
        if sha256_file(checkpoint) != expected:
            raise ValueError("dispatcher checkpoint SHA-256 differs from promotion")
        dispatcher = _load_dispatcher(checkpoint)
        model = DynamicRouter(
            "dispatcher-000733",
            promotion["promotion_id"],
            promotion["dispatcher"]["id"],
            experts,
            experts[0],
            clients,
            dispatcher,
            feature_dim=1024,
        )
    serve(V2Application(model, router=True), args.host, args.port)
    return 0


def _write_jsonl(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"output already exists: {path}")
    path.write_text(records_to_jsonl(records), encoding="utf-8")


def _output_inventory(output: Path) -> list[dict[str, object]]:
    """Return a deterministic inventory of regular evidence files already written."""

    inventory: list[dict[str, object]] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        inventory.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size_bytes": size,
                "sha256": digest.hexdigest(),
            }
        )
    return inventory


def _write_benchmark_failure(output: Path, *, stage: str, error: BaseException) -> None:
    """Seal a failed load-generator attempt without replacing any existing evidence."""

    reason = f"execution_exception:{stage}:{type(error).__name__}"
    validity = output / "validity.json"
    if not validity.exists() and not validity.is_symlink():
        write_json(validity, {"valid": False, "invalid_reasons": [reason]})
    complete = output / "loadgen-complete.json"
    if not complete.exists() and not complete.is_symlink():
        write_json(complete, {"complete": False, "failure_stage": stage})
    marker = output / "INVALID.json"
    if marker.exists() or marker.is_symlink():
        return
    body = {
        "schema_version": 1,
        "valid": False,
        "failure_stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "files": _output_inventory(output),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(body, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _signal_collector(done: Path | None, completed: Path | None) -> bool:
    """Stop the evidence sidecar and report whether its completion receipt arrived."""

    if done is None or completed is None:
        return True
    done.touch(exist_ok=False)
    for _ in range(300):
        if completed.exists():
            return True
        time.sleep(0.1)
    return False


def _run_benchmark(args: argparse.Namespace) -> int:
    if args.serving_action == "benchmark-collect":
        from .serving.collector import collect_until_done

        collect_until_done(args.output, args.done, args.completed)
        return 0
    if args.serving_action == "benchmark-parity":
        from .serving.parity import verify_offline_online_parity, write_parity_receipt

        records = [
            json.loads(line)
            for line in args.online_records.read_text(encoding="utf-8").splitlines()
            if line
        ]
        receipt = verify_offline_online_parity(
            args.trace,
            args.local_predictions,
            records,
            promotion_fingerprint=args.promotion_fingerprint,
            image_digests=_key_values(args.image_digest, "image digest"),
            golden_logits_atol=args.golden_logits_atol,
            golden_sample_index=args.golden_sample_index,
        )
        write_parity_receipt(args.output, receipt)
        return 0
    if args.serving_action == "benchmark-parity-collect":
        from .serving.parity_collector import collect_parity_arm

        collect_parity_arm(
            args.trace,
            arm=args.arm,
            endpoint=args.endpoint,
            expected_identity=_key_values(args.expected_identity, "expected identity"),
            output=args.output,
            timeout_seconds=args.timeout_seconds,
        )
        return 0
    if args.serving_action == "benchmark-parity-combine":
        from .serving.parity import ARMS
        from .serving.parity_collector import combine_parity_arms

        raw_inputs = _key_values(args.input, "parity input")
        if list(raw_inputs) != list(ARMS):
            raise ValueError("parity inputs must use the exact approved arm order")
        combine_parity_arms(
            args.trace,
            inputs={arm: Path(path) for arm, path in raw_inputs.items()},
            output=args.output,
        )
        return 0
    if args.serving_action == "benchmark-validate-evidence":
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        sources = {
            key: json.loads(path.read_text(encoding="utf-8"))
            for key, path in (
                ("requests", args.request_validity),
                ("parity", args.parity_receipt),
                ("snapshot", args.snapshot),
                ("metrics", args.metric_summary),
            )
        }
        evidence = {
            "request_invalid_reasons": sources["requests"].get("invalid_reasons", []),
            "parity_receipt": sources["parity"],
            "active_workflows": sources["snapshot"].get("active_workflows"),
            "pod_restart_delta": sources["snapshot"].get("pod_restart_delta"),
            "node_topology_stable": sources["snapshot"].get("node_topology_stable"),
            "pod_identity_stable": sources["snapshot"].get("pod_identity_stable"),
            "metric_gap_count": sources["metrics"].get("metric_gap_count"),
            "sources": sources,
        }
        validity = derive_validity(evidence)
        write_json(args.output, {**evidence, **validity})
        return 0 if validity["valid"] else 2
    if args.serving_action == "benchmark-summarize-evidence":
        from .serving.analysis import summarize_arm_counters, summarize_pod_counters

        evidence_records = [
            json.loads(line)
            for line in args.evidence.read_text(encoding="utf-8").splitlines()
            if line
        ]
        requests = [
            json.loads(line)
            for line in args.requests.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if args.markers is None:
            summary = summarize_pod_counters(
                evidence_records,
                request_count=len(requests),
                request_cpu_ns=sum(int(item.get("process_cpu_ns", 0)) for item in requests),
            )
        else:
            markers = [
                json.loads(line)
                for line in args.markers.read_text(encoding="utf-8").splitlines()
                if line
            ]
            summary = summarize_arm_counters(evidence_records, markers, requests)
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        write_json(args.output, summary)
        return 0
    if args.serving_action == "benchmark-analyze":
        if args.output.exists() or args.output.is_symlink():
            raise ValueError(f"output already exists: {args.output}")
        if (args.run_manifest is None) != (args.environment_manifest is None):
            raise ValueError(
                "run-manifest and environment-manifest must be supplied together"
            )
        records = [json.loads(line) for line in args.records.read_text().splitlines() if line]
        summary = analyze_records(
            records,
            baseline_arm=args.baseline_arm,
            comparison_arm=args.comparison_arm,
            expected_repeats=args.repeats,
            expected_samples=args.samples,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
        )
        if args.run_manifest is not None:
            run_manifest = BenchmarkRunManifestV1.from_text(
                args.run_manifest.read_text(encoding="utf-8")
            ).to_dict()
            environment = json.loads(args.environment_manifest.read_text(encoding="utf-8"))
            if not isinstance(environment, dict):
                raise ValueError("environment manifest must be a JSON object")
            validity = derive_validity(environment)
            invalid_reasons = validity["invalid_reasons"]
            valid = validity["valid"]
            body = {
                "schema_version": 1,
                "benchmark_id": run_manifest["benchmark_id"],
                "run_fingerprint": run_manifest["fingerprint"],
                "valid": valid,
                "invalid_reasons": invalid_reasons,
                "environment": environment,
                # Invalid runs retain raw observations but must not publish numeric effects.
                "accuracy": (
                    {
                        "effect_policy": summary["effect_policy"],
                        "interpretation": summary["interpretation"],
                        **summary["comparison"]["accuracy"],
                    }
                    if valid
                    else {}
                ),
                "performance": (
                    {
                        "arms": summary["arms"],
                        "comparison": summary["comparison"]["latency_ms"],
                        "request_cpu_comparison": summary["comparison"]["request_cpu_ms"],
                        "paired_comparisons": summary["paired_comparisons"],
                        "pod_counters": environment.get("sources", {}).get("metrics", {}),
                        "offline_mflops_note": summary["offline_mflops_note"],
                    }
                    if valid
                    else {}
                ),
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            summary = seal_manifest(body)
            BenchmarkSummaryV1.from_dict(summary)
        write_json(args.output, summary)
        return 0

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("benchmark config must be a mapping")
    validate_benchmark_config(config)
    expected_arms = list(APPROVED_ARMS)
    if (
        args.warmup != config["calibration"]["warmup_requests_per_arm"]
        or args.calibration != config["calibration"]["measured_requests_per_arm"]
        or args.repeats != config["steady_state"]["repeats"]
        or args.seed != config["seed"]
    ):
        raise ValueError("CLI execution parameters must exactly match benchmark.yaml")
    if args.trace_uri:
        from .serving.gcs import download_trace

        download_trace(args.trace_uri, args.trace)
    trace, images, labels, sample_ids = verify_trace(args.trace)
    matching_datasets = [
        item for item in config.get("datasets", []) if item.get("dataset_id") == trace["trace_id"]
    ]
    if len(matching_datasets) != 1 or matching_datasets[0].get("samples") != len(images):
        raise ValueError("trace identity/sample count is not declared exactly once in benchmark.yaml")
    promotion = _load_promotion(args.promotion)
    endpoints = _key_values(args.endpoint, "endpoint")
    if list(endpoints) != expected_arms:
        raise ValueError("benchmark endpoints must use the exact approved arm order")
    if (args.collector_done is None) != (args.collector_completed is None):
        raise ValueError("collector done/completed paths must be supplied together")
    args.output.mkdir(parents=True, exist_ok=False)
    invalid_reasons: list[str] = []
    all_records: list[Any] = []
    predictors: dict[str, KServeV2Predictor] = {}
    stage = "predictor_setup"
    try:
        predictors = {
            arm: KServeV2Predictor(url, arm, timeout_seconds=args.timeout_seconds)
            for arm, url in endpoints.items()
        }
        stage = "calibration"
        calibration, target_rps = calibrate(
            predictors,
            images[0:1],
            warmup_requests=args.warmup,
            measured_requests=args.calibration,
        )
        run_body = {
            "schema_version": 1,
            "benchmark_id": args.output.name,
            "promotion_fingerprint": promotion["fingerprint"],
            "trace_fingerprints": {str(trace["trace_id"]): trace["fingerprint"]},
            "arms": list(endpoints),
            "warmup_requests": args.warmup,
            "calibration_requests": args.calibration,
            "requests_per_repeat": len(images),
            "repeats": args.repeats,
            "target_rps": target_rps,
            "concurrency": 1,
            "seed": args.seed,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        run_manifest = seal_manifest(run_body)
        BenchmarkRunManifestV1.from_dict(run_manifest)
        # Seal execution parameters after calibration and before any steady-state request.
        write_json(args.output / "run.json", run_manifest)
        write_json(
            args.output / "calibration.json",
            {"latency_seconds": calibration, "target_rps": target_rps},
        )
        stage = "steady_state"
        markers = args.output / "markers.jsonl"
        markers.write_text(
            json.dumps({"event": "traffic_start", "timestamp_ns": __import__("time").time_ns()})
            + "\n",
            encoding="utf-8",
        )
        expert_ids = [item["id"] for item in promotion["experts"]]
        for repeat, order in enumerate(latin_square(tuple(endpoints), args.repeats)):
            for arm in order:
                with markers.open("a", encoding="utf-8") as marker_stream:
                    marker_stream.write(
                        json.dumps(
                            {
                                "event": "arm_start",
                                "arm": arm,
                                "repeat": repeat,
                                "timestamp_ns": __import__("time").time_ns(),
                            }
                        )
                        + "\n"
                    )
                expected_identity = {"model_name": arm}
                if arm == "dispatcher-000733":
                    expected_identity.update(
                        {
                            "promotion_id": promotion["promotion_id"],
                            "dispatcher_id": "dispatcher-000733",
                        }
                    )
                elif arm == "static-resnet-router":
                    expected_identity.update(
                        {
                            "promotion_id": promotion["promotion_id"],
                            "dispatcher_id": "static",
                            "expert_id": expert_ids[-1],
                        }
                    )
                records, invalid = run_scheduled_arm(
                    dataset_id=str(trace["trace_id"]),
                    arm=arm,
                    repeat=repeat,
                    images=images,
                    labels=labels,
                    sample_ids=sample_ids,
                    predict=predictors[arm],
                    request_rate=target_rps,
                    expected_identity=expected_identity,
                )
                all_records.extend(records)
                with markers.open("a", encoding="utf-8") as marker_stream:
                    marker_stream.write(
                        json.dumps(
                            {
                                "event": "arm_end",
                                "arm": arm,
                                "repeat": repeat,
                                "timestamp_ns": __import__("time").time_ns(),
                            }
                        )
                        + "\n"
                    )
                if invalid:
                    invalid_reasons.append(f"{arm}:repeat={repeat}:{invalid}")
                    break
            if invalid_reasons:
                break
        _write_jsonl(args.output / "requests.jsonl", all_records)
        write_parquet([record.to_dict() for record in all_records], args.output / "requests.parquet")
        with markers.open("a", encoding="utf-8") as marker_stream:
            marker_stream.write(
                json.dumps({"event": "traffic_end", "timestamp_ns": __import__("time").time_ns()})
                + "\n"
            )
        stage = "collector_handshake"
        if not _signal_collector(args.collector_done, args.collector_completed):
            invalid_reasons.append("collector_handshake_failed")
        write_json(
            args.output / "validity.json",
            {"valid": not invalid_reasons, "invalid_reasons": invalid_reasons},
        )
        write_json(args.output / "loadgen-complete.json", {"complete": True})
        if args.output_uri:
            stage = "gcs_publish"
            from .serving.gcs import publish_create_only

            publish_create_only(args.output, args.output_uri, kind="run")
    except Exception as error:
        if all_records and not (args.output / "requests.jsonl").exists():
            try:
                _write_jsonl(args.output / "requests.jsonl", all_records)
            except Exception as preservation_error:
                error.add_note(
                    f"could not preserve partial request records: {type(preservation_error).__name__}"
                )
        if args.collector_done is not None and not args.collector_done.exists():
            try:
                if not _signal_collector(args.collector_done, args.collector_completed):
                    error.add_note("evidence collector did not acknowledge the failure signal")
            except Exception as collector_error:
                error.add_note(
                    f"could not signal evidence collector: {type(collector_error).__name__}"
                )
        try:
            _write_benchmark_failure(args.output, stage=stage, error=error)
        except Exception as preservation_error:
            error.add_note(
                f"could not seal local INVALID evidence: {type(preservation_error).__name__}"
            )
        if args.output_uri and stage != "gcs_publish":
            try:
                from .serving.gcs import publish_create_only

                publish_create_only(args.output, args.output_uri, kind="run")
            except Exception as publish_error:
                error.add_note(
                    f"could not publish INVALID evidence: {type(publish_error).__name__}"
                )
        raise
    finally:
        for predictor in predictors.values():
            predictor.close()
    return 0 if not invalid_reasons else 2


def run_command(args: argparse.Namespace) -> int:
    if args.serving_action.startswith("promotion-"):
        return _run_promotion(args)
    if args.serving_action.startswith("trace-"):
        return _run_trace(args)
    if args.serving_action.startswith("serve-"):
        return _run_serve(args)
    return _run_benchmark(args)
