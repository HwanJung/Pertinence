"""Low-load, fixed-schedule benchmark primitives for KServe V2 endpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import http.client
import json
import math
import time
from typing import Callable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

import numpy as np

from .contracts import APPROVED_BENCHMARK_ARMS


APPROVED_ARMS = APPROVED_BENCHMARK_ARMS


def validate_benchmark_config(value: Mapping[str, object]) -> None:
    """Validate the fixed low-load benchmark protocol used by the runner."""

    if value.get("schema_version") != 1 or value.get("throughput_saturation_test") is not False:
        raise ValueError("benchmark config must be schema v1 low-load mode")
    if value.get("arms") != list(APPROVED_ARMS):
        raise ValueError("benchmark config arms/order is not approved")
    if "target" in value:
        raise ValueError("benchmark config must declare the approved arms, not a target")
    request = value.get("request")
    if not isinstance(request, Mapping) or any(
        request.get(key) != expected
        for key, expected in (
            ("send_label", False),
            ("batch_size", 1),
            ("concurrency", 1),
            ("retry_count", 0),
            ("max_in_flight", 1),
        )
    ):
        raise ValueError("benchmark request policy differs from the approved serialized policy")
    warmup = value.get("warmup")
    steady = value.get("steady_state")
    if (
        not isinstance(warmup, Mapping)
        or warmup.get("requests_per_arm") != 200
        or warmup.get("distinct_samples") is not True
        or not isinstance(steady, Mapping)
        or "repeats" in steady
        or steady.get("order") != "single_pass"
        or steady.get("arm_order") != list(APPROVED_ARMS)
        or steady.get("requests_per_arm") != 2_000
        or steady.get("fixed_rps") != 2
        or steady.get("queue_on_missed_schedule") is not False
    ):
        raise ValueError("benchmark warm-up or single-pass policy differs")


class Predictor(Protocol):
    def __call__(self, image: np.ndarray) -> Mapping[str, object]: ...


class KServeV2Predictor:
    """Persistent, serialized V2 HTTP client with a single attempt per call."""

    def __init__(self, endpoint: str, model_name: str, *, timeout_seconds: float = 10.0) -> None:
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("endpoint must be an http(s) URL")
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        connection = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        self._connection = connection(parsed.hostname, port, timeout=timeout_seconds)
        base = parsed.path.rstrip("/")
        self._path = f"{base}/v2/models/{model_name}/infer"
        self.model_name = model_name

    def __call__(self, image: np.ndarray) -> Mapping[str, object]:
        value = np.asarray(image)
        if value.shape != (1, 3, 32, 32) or value.dtype != np.float32:
            raise ValueError("image must be float32 [1,3,32,32]")
        request = {
            "inputs": [
                {
                    "name": "image",
                    "shape": [1, 3, 32, 32],
                    "datatype": "FP32",
                    "data": value.reshape(-1).tolist(),
                }
            ]
        }
        body = json.dumps(request, separators=(",", ":"), allow_nan=False).encode()
        try:
            self._connection.request(
                "POST", self._path, body=body, headers={"Content-Type": "application/json"}
            )
            response = self._connection.getresponse()
            payload = response.read()
        except (OSError, TimeoutError, http.client.HTTPException):
            self._connection.close()
            raise
        if response.status != 200:
            raise RuntimeError(f"{self.model_name} returned HTTP {response.status}")
        value = json.loads(payload)
        outputs = value.get("outputs")
        if not isinstance(outputs, list):
            raise ValueError("V2 response outputs must be a list")
        by_name = {item.get("name"): item for item in outputs if isinstance(item, dict)}
        if len(by_name) != len(outputs) or "logits" not in by_name:
            raise ValueError("V2 response output names are invalid")
        logits_output = by_name["logits"]
        if logits_output.get("shape") != [1, 10] or logits_output.get("datatype") != "FP32":
            raise ValueError("V2 logits metadata is invalid")
        logits = np.asarray(logits_output.get("data"), dtype=np.float32).reshape(1, 10)
        route = None
        if "route" in by_name:
            route_output = by_name["route"]
            if route_output.get("shape") != [1] or route_output.get("datatype") != "INT64":
                raise ValueError("V2 route metadata is invalid")
            route = int(route_output.get("data")[0])
        parameters = value.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("V2 response parameters are invalid")
        stage_latency_ns = {
            stage.removesuffix("_seconds") + "_ns": round(float(parameters[stage]) * 1e9)
            for stage in ("extractor_seconds", "dispatcher_seconds", "expert_seconds")
            if stage in parameters
        }
        process_cpu_ns = round(float(parameters.get("process_cpu_seconds", 0.0)) * 1e9)
        if process_cpu_ns < 0:
            raise ValueError("V2 response process CPU time is invalid")
        identity = {
            "model_name": str(value.get("model_name", "")),
            "model_version": str(value.get("model_version", "")),
            **{
                key: str(parameters[key])
                for key in ("promotion_id", "dispatcher_id", "expert_id")
                if key in parameters
            },
        }
        return {
            "logits": logits,
            "route": route,
            "identity": identity,
            "stage_latency_ns": stage_latency_ns,
            "process_cpu_ns": process_cpu_ns,
        }

    def close(self) -> None:
        self._connection.close()


@dataclass(frozen=True)
class RequestRecord:
    dataset_id: str
    arm: str
    repeat: int
    sample_index: int
    sample_id: str
    scheduled_ns: int
    started_ns: int
    completed_ns: int
    latency_ns: int
    prediction: int
    logits: list[float]
    correct: bool
    route: int | None
    identity: Mapping[str, str]
    stage_latency_ns: Mapping[str, int]
    process_cpu_ns: int
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_prediction(result: Mapping[str, object]) -> tuple[int, int | None, list[float]]:
    logits = np.asarray(result.get("logits"), dtype=np.float32)
    if logits.shape != (1, 10) or not np.isfinite(logits).all():
        raise ValueError("predictor returned invalid logits")
    raw_route = result.get("route")
    route = None if raw_route is None else int(raw_route)
    if route is not None and route not in (0, 1, 2):
        raise ValueError("predictor returned invalid route")
    return int(np.argmax(logits[0])), route, logits[0].tolist()


def warm_up_arm(
    *,
    images: np.ndarray,
    predict: Predictor,
    expected_identity: Mapping[str, str],
    requests: int = 200,
) -> str | None:
    """Warm an endpoint with distinct samples; no warm-up observation is returned."""

    values = np.asarray(images)
    if values.ndim != 4 or values.shape[1:] != (3, 32, 32) or values.dtype != np.float32:
        raise ValueError("images must be float32 [N,3,32,32]")
    if requests < 1 or requests > len(values):
        raise ValueError("warm-up requests must fit the distinct input samples")
    for index, image in enumerate(values[:requests]):
        try:
            result = predict(image[np.newaxis, ...])
            _validate_prediction(result)
            identity = {
                str(key): str(value)
                for key, value in dict(result.get("identity", {})).items()
            }
            if any(identity.get(key) != value for key, value in expected_identity.items()):
                return f"identity_mismatch:sample={index}"
        except Exception as error:
            return f"request_error:sample={index}:{type(error).__name__}:{error}"
    return None


def run_scheduled_arm(
    *,
    dataset_id: str,
    arm: str,
    repeat: int,
    images: np.ndarray,
    labels: np.ndarray,
    sample_ids: Sequence[str],
    predict: Predictor,
    request_rate: float,
    expected_identity: Mapping[str, str] | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[RequestRecord], str | None]:
    """Run at most one request at a time; invalidate rather than queue a late request."""

    values = np.asarray(images)
    targets = np.asarray(labels)
    if values.ndim != 4 or values.shape[1:] != (3, 32, 32) or values.dtype != np.float32:
        raise ValueError("images must be float32 [N,3,32,32]")
    if targets.shape != (values.shape[0],) or len(sample_ids) != values.shape[0]:
        raise ValueError("labels and sample IDs must align with images")
    if not math.isfinite(request_rate) or request_rate <= 0.0:
        raise ValueError("request_rate must be positive and finite")

    interval_ns = round(1_000_000_000 / request_rate)
    first_schedule = clock_ns()
    records: list[RequestRecord] = []
    for index, (image, label, sample_id) in enumerate(zip(values, targets, sample_ids)):
        scheduled = first_schedule + index * interval_ns
        now = clock_ns()
        if now > scheduled and index > 0:
            return records, f"missed_schedule:sample={index}:late_ns={now - scheduled}"
        if now < scheduled:
            sleep((scheduled - now) / 1_000_000_000.0)
        started = clock_ns()
        try:
            result = predict(image[np.newaxis, ...])
            prediction, route, logits = _validate_prediction(result)
            identity = {str(key): str(value) for key, value in dict(result.get("identity", {})).items()}
            if expected_identity is not None and any(
                identity.get(key) != value for key, value in expected_identity.items()
            ):
                return records, f"identity_mismatch:sample={index}"
            stages = {
                str(key): int(value)
                for key, value in dict(result.get("stage_latency_ns", {})).items()
            }
            completed = clock_ns()
            records.append(
                RequestRecord(
                    dataset_id=dataset_id,
                    arm=arm,
                    repeat=repeat,
                    sample_index=index,
                    sample_id=str(sample_id),
                    scheduled_ns=scheduled,
                    started_ns=started,
                    completed_ns=completed,
                    latency_ns=completed - started,
                    prediction=prediction,
                    logits=logits,
                    correct=prediction == int(label),
                    route=route,
                    identity=identity,
                    stage_latency_ns=stages,
                    process_cpu_ns=int(result.get("process_cpu_ns", 0)),
                )
            )
        except Exception as error:  # A failed request invalidates the run; there is no retry.
            return records, f"request_error:sample={index}:{type(error).__name__}:{error}"
    return records, None


def records_to_jsonl(records: Sequence[RequestRecord]) -> str:
    return "".join(json.dumps(item.to_dict(), sort_keys=True) + "\n" for item in records)
