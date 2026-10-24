"""One pass of three direct experts, with durable per-attempt evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np

from pertinence.serving.benchmark import KServeV2Predictor, _validate_prediction
from pertinence.serving.trace import verify_trace

MODELS = ("shufflenetv2_x0_5", "mobilenetv2_x0_5", "resnet56")
WARMUP = 200
MEASURED = 2000
RPS = 5
TRACE_FINGERPRINT = "3185ad8c7d3663eda8e7e5710929d83afdf255f35a6837caba5f644767ce6b7e"


def latency_stats(values: list[float], prefix: str = "") -> dict:
    """Milliseconds, with NumPy's default linear percentile interpolation."""
    return {prefix + name + "_ms": float(function(values)) if values else None
            for name, function in (("mean", np.mean), ("p50", lambda v: np.percentile(v, 50)),
                                   ("p95", lambda v: np.percentile(v, 95)),
                                   ("p99", lambda v: np.percentile(v, 99)), ("max", np.max))}


def write_summary(output: Path, summaries: list[dict]) -> None:
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def run_once(
    *, images: np.ndarray, labels: np.ndarray, sample_ids: list[str], dataset_id: str,
    arms: list[dict], output: Path, guard: Callable[[], None], run_id: str,
    predictor_factory: Callable = KServeV2Predictor,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Serial start-to-start pacing; never retry or catch up after a delayed call."""
    if (tuple(arm["model_name"] for arm in arms) != MODELS or len(images) != MEASURED
            or images.shape != (MEASURED, 3, 32, 32) or images.dtype != np.float32
            or len(labels) != MEASURED or len(sample_ids) != MEASURED or not run_id):
        raise ValueError("expected three ordered experts and the full 2000-sample trace")
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    failure = None
    with (output / "requests.jsonl").open("x", buffering=1) as journal:
        for arm in arms:
            model = arm["model_name"]
            attempts: list[dict] = []
            client = None
            previous_start = None
            try:
                client = predictor_factory(arm["endpoint"], model, timeout_seconds=10.0,
                                           request_id_prefix=f"{run_id}-{model}")
                for phase, count in (("warmup", WARMUP), ("measured", MEASURED)):
                    for index in range(count):
                        image = images[index][np.newaxis, ...]
                        row = {"run_id": run_id, "dataset_id": dataset_id, "model_name": model,
                               "request_id": f"{run_id}-{model}-{len(attempts)}",
                               "phase": phase, "sample_index": index,
                               "sample_id": sample_ids[index], "repeat": 0,
                               "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                               "model_sha256": arm["model_sha256"], "pod": arm["pod"],
                               "runtime_image": arm["runtime_image"], "modelcar": arm["modelcar"],
                               "success": False}
                        if previous_start is not None:
                            remaining = previous_start + round(1e9 / RPS) - clock_ns()
                            if remaining > 0:
                                sleep(remaining / 1e9)
                        guard()
                        started = previous_start = clock_ns()
                        row["started_ns"] = started
                        try:
                            result = client(image)
                            prediction, route, logits = _validate_prediction(result)
                            identity = {str(k): str(v) for k, v in
                                        dict(result.get("identity", {})).items()}
                            expected = {"model_name": model,
                                        "run_fingerprint": arm["model_sha256"]}
                            if any(identity.get(k) != v for k, v in expected.items()):
                                raise ValueError("response model/hash identity mismatch")
                            stages = {str(k): int(v) for k, v in
                                      dict(result.get("stage_latency_ns", {})).items()}
                            completed = clock_ns()
                            row.update(completed_ns=completed, latency_ns=completed - started,
                                       identity=identity, prediction=prediction, route=route,
                                       logits=logits, correct=prediction == int(labels[index]),
                                       stage_latency_ns=stages, success=True)
                        except Exception as error:
                            completed = clock_ns()
                            row.update(completed_ns=completed, latency_ns=completed - started,
                                       error=f"{type(error).__name__}:{error}")
                            raise
                        finally:
                            attempts.append(row)
                            journal.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                        guard()
            except Exception as error:
                failure = f"{type(error).__name__}:{error}"
            finally:
                if client is not None:
                    client.close()
            measured = [r for r in attempts if r["phase"] == "measured"]
            latency = [r["latency_ns"] / 1e6 for r in measured if r["success"]]
            summaries.append({
                "model": model, "requests": len(measured),
                "errors": sum(not r["success"] for r in measured),
                "warmup_requests": sum(r["phase"] == "warmup" for r in attempts),
                "warmup_errors": sum(r["phase"] == "warmup" and not r["success"] for r in attempts),
                "achieved_rps": len(measured) * 1e9 / (
                    measured[-1]["completed_ns"] - measured[0]["started_ns"]) if measured else None,
                **latency_stats(latency), "failure": failure,
            })
            if failure:
                break
    write_summary(output, summaries)
    result = {"valid": False, "client_valid": failure is None and len(summaries) == 3,
              "failure": failure, "summaries": summaries,
              "validation": "pending server span join and controller cleanup"}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def join_model_spans(rows: list[dict], spans: list[dict]) -> list[dict]:
    """Join direct calls without RPC call_index; fail closed on ambiguous evidence."""
    requests = {(r["request_id"], r["model_name"]): r for r in rows}
    if len(requests) != len(rows) or len({r["request_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate client request ID")
    by_request: dict[tuple, list[dict]] = {}
    for span in spans:
        if span.get("stage") != "expert_model":
            continue
        key = (span.get("request_id"), span.get("expert"))
        if key not in requests:
            raise ValueError(f"unmatched request/model span: {key}")
        by_request.setdefault(key, []).append(span)
    joined = []
    for key, row in requests.items():
        matches = by_request.get(key, [])
        if row["phase"] != "measured":
            continue
        if len(matches) != 1:
            raise ValueError(f"expected exactly one model span for {key}; got {len(matches)}")
        span = matches[0]
        values = [span.get(name) for name in ("wall_seconds", "thread_cpu_seconds")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v < 0 for v in values) or values[0] == 0:
            raise ValueError(f"invalid model times: {key}")
        joined.append({**row, "model_wall_ms": values[0] * 1000,
                       "model_thread_cpu_ms": values[1] * 1000})
    return joined


def finalize(output: Path) -> dict:
    """Validate recovered evidence and write the final per-request and aggregate outputs."""
    result = json.loads((output / "raw/results/result.json").read_text())
    rows = [json.loads(line) for line in
            (output / "raw/results/requests.jsonl").read_text().splitlines()]
    spans = [json.loads(line) for model in MODELS for line in
             (output / f"{model}-spans.jsonl").read_text().splitlines()]
    joined = join_model_spans(rows, spans)
    with (output / "joined-requests.jsonl").open("x") as stream:
        for row in joined:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    for summary in result["summaries"]:
        model_rows = [r for r in joined if r["model_name"] == summary["model"]]
        summary.update(model_spans=len(model_rows),
                       **latency_stats([r["model_wall_ms"] for r in model_rows], "model_"),
                       model_thread_cpu_mean_ms=float(np.mean(
                           [r["model_thread_cpu_ms"] for r in model_rows])) if model_rows else None)
    result["valid"] = (result["client_valid"] and len(joined) == MEASURED * len(MODELS)
                       and all(s["requests"] == MEASURED and s["errors"] == 0
                               and s["warmup_requests"] == WARMUP and s["warmup_errors"] == 0
                               and s["model_spans"] == MEASURED for s in result["summaries"]))
    result["validation"] = "server spans joined; environment and cleanup checked by controller"
    write_summary(output, result["summaries"])
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())

    def guard() -> None:
        state = json.loads(args.control.read_text())
        if state.get("abort") or time.time() - state["time"] > 20:
            raise RuntimeError("environment changed or controller heartbeat expired")

    try:
        deadline = time.monotonic() + 120
        while not args.control.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("controller did not initialize; no inference attempted")
            time.sleep(0.2)
        guard()
        from pertinence.serving.gcs import download_trace
        download_trace(manifest["trace_uri"], args.trace)
        trace, images, labels, ids = verify_trace(args.trace)
        if (trace["trace_id"] != "cifar10-final-20260824"
                or trace["fingerprint"] != TRACE_FINGERPRINT):
            raise ValueError("expected existing fixed CIFAR-10 final trace")
        (args.output.parent / "trace.json").write_text(json.dumps(trace, indent=2))
        run_once(images=images, labels=labels, sample_ids=ids,
                 dataset_id=trace["dataset"]["id"], arms=manifest["arms"],
                 output=args.output, guard=guard, run_id=manifest["run_id"])
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "INVALID.json").write_text(json.dumps({"error": str(error)}))
    finally:
        (args.output.parent / "done").touch()
    # Preserve emptyDir artifacts until the controller retrieves and acknowledges them.
    while not (args.output.parent / "handoff").exists():
        time.sleep(1)


if __name__ == "__main__":
    main()
