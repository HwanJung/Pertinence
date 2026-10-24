"""Run and analyze the independent transport suite: python -m this module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .benchmark import KServeV2Predictor, run_scheduled_arm, warm_up_arm
from .cli.benchmark import _signal_collector, _write_json_exclusive, _write_jsonl
from .contracts import PromotionManifestV1, seal_manifest
from .trace import verify_trace
from .transport_suite import (
    VARIANTS, compare_runs, join_diagnostics, latency_summary, schedule, validate_manifest,
    verify_run_records, summarize_resources,
)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def wait_for_initial_evidence(path: Path, *, timeout: float = 30.0,
                              clock=time.monotonic, sleep=time.sleep) -> None:
    """Gate warm-up on a full collector cycle so even fast arms have a baseline."""
    deadline = clock() + timeout
    while clock() < deadline:
        try:
            rows = read_rows(path)
            nodes = {row["node"] for row in rows}
            serving_metrics = any('container="kserve-container"' in metric
                                  for row in rows for metric in row.get("metrics", []))
            if len(nodes) == 2 and serving_metrics:
                return
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # the collector may still be writing its first complete cycle
        sleep(0.1)
    raise TimeoutError("initial serving/loadgen cAdvisor cycle was not collected")


def run(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    promotion = PromotionManifestV1.from_text(args.promotion.read_text()).to_dict()
    if args.trace_uri:
        from .gcs import download_trace
        download_trace(args.trace_uri, args.trace)
    trace, images, labels, samples = verify_trace(args.trace)
    if (promotion["fingerprint"] != manifest["promotion_fingerprint"]
            or trace["fingerprint"] != manifest["trace_fingerprint"] or len(images) != 2000):
        raise ValueError("promotion or trace differs from sealed run")
    if (args.collector_done is None) != (args.collector_completed is None):
        raise ValueError("collector paths must be supplied together")
    args.output.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(args.output / "run.json", manifest)
    expected = {"promotion_id": promotion["promotion_id"], "dispatcher_id": "dispatcher-000733",
                "serving_mode": VARIANTS[manifest["arm"]].structure,
                "run_fingerprint": manifest["fingerprint"]}
    predictor = KServeV2Predictor(args.endpoint, "dispatcher-000733",
                                  request_id_prefix=manifest["run_id"])
    rows, invalid = [], None
    markers = []

    def mark(event: str) -> None:
        markers.append({"event": event, "timestamp_ns": time.time_ns()})

    try:
        if args.collector_done is not None:
            wait_for_initial_evidence(args.output / "evidence.jsonl")
        mark("warmup_start")
        invalid = warm_up_arm(images=images, predict=predictor, expected_identity=expected)
        mark("warmup_end")
        if invalid is None:
            mark("arm_start")
            rows, invalid = run_scheduled_arm(
                dataset_id=str(trace["trace_id"]), arm=manifest["arm"], repeat=manifest["block"],
                images=images, labels=labels, sample_ids=samples, predict=predictor,
                request_rate=5, expected_identity=expected,
            )
            mark("arm_end")
    except Exception as error:
        invalid = f"{type(error).__name__}:{error}"
    finally:
        predictor.close()
        _write_jsonl(args.output / "requests.jsonl", rows)
        _write_json_exclusive(args.output / "markers.json", {"markers": markers})
        if args.collector_done is not None:
            time.sleep(6)  # retain a cAdvisor sample after the final measured request
        collected = _signal_collector(args.collector_done, args.collector_completed)
        _write_json_exclusive(args.output / "execution.json", {
            "request_error": invalid, "requests": len(rows),
            "collector_complete": bool(collected and args.collector_done is not None),
        })
    if args.output_uri:
        from .gcs import publish_create_only
        publish_create_only(args.output, args.output_uri, kind="transport-run")
    if args.handoff_file is not None:
        _write_json_exclusive(args.output / "ready.json", {"ready": True})
        deadline = time.monotonic() + 600
        while not args.handoff_file.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("local artifact handoff was not acknowledged in 600 seconds")
            time.sleep(1)
    return int(invalid is not None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("schedule")
    plan.add_argument("--seed", type=int, default=20261011)
    plan.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run")
    for name in ("manifest", "promotion", "trace", "output"):
        execute.add_argument(f"--{name}", type=Path, required=True)
    execute.add_argument("--trace-uri")
    execute.add_argument("--output-uri")
    execute.add_argument("--handoff-file", type=Path)
    execute.add_argument("--endpoint", required=True)
    execute.add_argument("--collector-done", type=Path)
    execute.add_argument("--collector-completed", type=Path)
    analyze = commands.add_parser("analyze")
    for name in ("run", "reference", "evidence", "output"):
        analyze.add_argument(f"--{name}", type=Path, required=True)
    resources = commands.add_parser("resources")
    resources.add_argument("--run", type=Path, required=True)
    resources.add_argument("--pod", action="append", required=True)
    resources.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--summary", type=Path, action="append", required=True)
    compare.add_argument("--output", type=Path, required=True)
    diagnose = commands.add_parser("diagnose")
    diagnose.add_argument("--router", type=Path, required=True)
    diagnose.add_argument("--expert", type=Path, action="append", required=True)
    diagnose.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        return run(args)
    if args.command == "schedule":
        result = {"diagnostic_arms": list(VARIANTS), "performance_blocks": schedule(args.seed)}
    elif args.command == "resources":
        markers = json.loads((args.run / "markers.json").read_text())["markers"]
        times = {row["event"]: row["timestamp_ns"] for row in markers}
        rows = read_rows(args.run / "requests.jsonl")
        execution = json.loads((args.run / "execution.json").read_text())
        result = summarize_resources(read_rows(args.run / "evidence.jsonl"),
                                     start_ns=times["arm_start"], end_ns=times["arm_end"],
                                     pod_names=args.pod, request_count=len(rows))
        result.update(errors=int(execution["request_error"] is not None),
                      missing=2000 - len(rows),
                      duplicates=len(rows) - len({row["sample_id"] for row in rows}),
                      schedule_misses=int("missed_schedule" in str(execution["request_error"])))
    elif args.command == "compare":
        result = compare_runs([json.loads(path.read_text()) for path in args.summary])
    elif args.command == "diagnose":
        result = {"calls": join_diagnostics(read_rows(args.router),
                  [row for path in args.expert for row in read_rows(path)])}
    else:
        manifest = json.loads((args.run / "run.json").read_text())
        validate_manifest(manifest)
        rows = read_rows(args.run / "requests.jsonl")
        evidence = json.loads(args.evidence.read_text())
        validity = verify_run_records(rows, read_rows(args.reference), evidence)
        execution = json.loads((args.run / "execution.json").read_text())
        if execution["request_error"] or not execution["collector_complete"]:
            validity["invalid_reasons"].append("execution_or_collection_failed")
            validity["valid"] = False
        latency = latency_summary(rows)
        elapsed = (rows[-1]["completed_ns"] - rows[0]["started_ns"]) / 1e9
        result = seal_manifest({"manifest": manifest, **validity, **latency["overall"],
                                "latency": latency, "evidence": evidence,
                                "pod_cpu_ms_per_request": evidence["pod_cpu_ms_per_request"],
                                "achieved_rps": len(rows) / elapsed})
    _write_json_exclusive(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
