"""Expert and router execution with explicit no-retry upstream behavior."""

from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
from pathlib import Path
import threading
import time
from typing import Any, Protocol
from urllib.parse import urlsplit

import numpy as np

from ..pipeline.artifacts import verify_artifact
from .v2 import IMAGE_SHAPE, V2ProtocolError, infer_response, parse_infer_response


class ExpertClientError(RuntimeError):
    """An expert request failed; routers propagate it without retrying."""


class ExpertTimeoutError(ExpertClientError, TimeoutError):
    """The single allowed expert request exceeded its deadline."""


@dataclass(frozen=True)
class ExpertResult:
    logits: np.ndarray
    features: np.ndarray | None = None
    process_cpu_seconds: float = 0.0


class ExpertClient(Protocol):
    def predict(self, image: np.ndarray, *, include_features: bool = False) -> ExpertResult: ...


class Dispatcher(Protocol):
    def predict(self, features: np.ndarray) -> np.ndarray: ...


def _validate_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.shape != IMAGE_SHAPE or image.dtype != np.float32 or not np.isfinite(image).all():
        raise ValueError("image must be finite float32 [1, 3, 32, 32]")
    return image


def _validate_result(result: ExpertResult, feature_dim: int | None = None) -> ExpertResult:
    logits = np.asarray(result.logits)
    if logits.shape != (1, 10) or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise ValueError("expert logits must be finite float32 [1, 10]")
    if feature_dim is not None:
        features = np.asarray(result.features) if result.features is not None else None
        if (
            features is None
            or features.shape != (1, feature_dim)
            or features.dtype != np.float32
            or not np.isfinite(features).all()
        ):
            raise ValueError(f"extractor features must be finite float32 [1, {feature_dim}]")
    return result


class OnnxExpertRunner:
    """A single-worker CPU ONNX expert with fixed one-thread session options."""

    def __init__(
        self,
        model_path: Path,
        model_sha256: str,
        *,
        input_name: str = "image",
        logits_name: str = "logits",
        features_name: str | None = None,
        feature_dim: int | None = None,
    ) -> None:
        import onnxruntime as ort

        verify_artifact(Path(model_path), model_sha256)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = input_name
        self.logits_name = logits_name
        self.features_name = features_name
        self.feature_dim = feature_dim
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].name != input_name or inputs[0].type != "tensor(float)":
            raise ValueError("ONNX expert must have exactly one declared FP32 image input")
        shape = inputs[0].shape
        if len(shape) != 4 or shape[1:] != list(IMAGE_SHAPE[1:]) or (
            isinstance(shape[0], int) and shape[0] != 1
        ):
            raise ValueError("ONNX expert input must be [batch, 3, 32, 32] with batch 1 supported")
        output_names = {output.name for output in self.session.get_outputs()}
        required = {logits_name} | ({features_name} if features_name else set())
        if not required.issubset(output_names):
            raise ValueError("ONNX expert is missing a declared output")
        if (features_name is None) != (feature_dim is None):
            raise ValueError("features_name and feature_dim must be configured together")

    def predict(self, image: np.ndarray, *, include_features: bool = False) -> ExpertResult:
        image = _validate_image(image)
        if include_features and self.features_name is None:
            raise ValueError("expert does not expose extractor features")
        outputs = [self.logits_name]
        if include_features:
            outputs.append(str(self.features_name))
        values = self.session.run(outputs, {self.input_name: image})
        result = ExpertResult(
            logits=np.asarray(values[0], dtype=np.float32),
            features=np.asarray(values[1], dtype=np.float32) if include_features else None,
        )
        return _validate_result(result, self.feature_dim if include_features else None)


class LinearDispatcherRunner:
    """Numpy implementation of the approved single linear routing layer."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        self.weight = np.asarray(weight)
        self.bias = np.asarray(bias)
        if self.weight.dtype != np.float32 or self.bias.dtype != np.float32:
            raise ValueError("dispatcher parameters must be FP32")
        if self.weight.ndim != 2 or self.bias.shape != (self.weight.shape[0],):
            raise ValueError("dispatcher weight/bias shapes are incompatible")
        if not np.isfinite(self.weight).all() or not np.isfinite(self.bias).all():
            raise ValueError("dispatcher parameters must be finite")

    @property
    def feature_dim(self) -> int:
        return int(self.weight.shape[1])

    @property
    def expert_count(self) -> int:
        return int(self.weight.shape[0])

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features)
        if (
            features.shape != (1, self.feature_dim)
            or features.dtype != np.float32
            or not np.isfinite(features).all()
        ):
            raise ValueError("dispatcher features have invalid shape, dtype, or values")
        return features @ self.weight.T + self.bias


def load_dispatcher_npz(path: Path) -> LinearDispatcherRunner:
    """Load the approved checkpoint format without pickle or a Torch dependency."""

    checkpoint = Path(path)
    if checkpoint.suffix != ".npz" or not checkpoint.is_file() or checkpoint.is_symlink():
        raise ValueError("dispatcher checkpoint must be a regular .npz file")
    with np.load(checkpoint, allow_pickle=False) as state:
        if len(state.files) != 2 or set(state.files) != {"linear.weight", "linear.bias"}:
            raise ValueError("dispatcher checkpoint must contain exactly linear.weight/linear.bias")
        weight = state["linear.weight"]
        bias = state["linear.bias"]
    if weight.dtype != np.float32 or bias.dtype != np.float32:
        raise ValueError("dispatcher checkpoint arrays must be FP32")
    if weight.shape != (3, 1024) or bias.shape != (3,):
        raise ValueError("dispatcher checkpoint shapes must be (3,1024) and (3,)")
    return LinearDispatcherRunner(weight, bias)


class HttpExpertClient:
    """One reusable HTTP connection, serialized and never retried."""

    def __init__(
        self,
        endpoint: str,
        model_name: str,
        *,
        timeout_seconds: float,
        feature_dim: int | None = None,
    ) -> None:
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("expert endpoint must be an http(s) URL")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.feature_dim = feature_dim
        self._path = parsed.path.rstrip("/") + f"/v2/models/{model_name}/infer"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        connection_type = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        self._connection = connection_type(parsed.hostname, port, timeout=timeout_seconds)
        self._lock = threading.Lock()

    def predict(self, image: np.ndarray, *, include_features: bool = False) -> ExpertResult:
        image = _validate_image(image)
        if include_features and self.feature_dim is None:
            raise ValueError("client has no configured feature output")
        request = {
            "inputs": [
                {
                    "name": "image",
                    "shape": list(IMAGE_SHAPE),
                    "datatype": "FP32",
                    "data": image.reshape(-1).tolist(),
                }
            ]
        }
        if include_features:
            request["outputs"] = [{"name": "logits"}, {"name": "features"}]
        encoded = json.dumps(request, separators=(",", ":"), allow_nan=False).encode("utf-8")
        # A lock is the bounded connection pool for a concurrency-one router.
        # No loop exists here: timeout, disconnect, 5xx, and protocol errors are
        # surfaced from this single attempt.
        try:
            with self._lock:
                self._connection.request(
                    "POST", self._path, encoded, {"Content-Type": "application/json"}
                )
                response = self._connection.getresponse()
                body = response.read()
        except (TimeoutError, http.client.RemoteDisconnected) as error:
            self._connection.close()
            raise ExpertTimeoutError(f"expert {self.model_name} request timed out") from error
        except (OSError, http.client.HTTPException) as error:
            self._connection.close()
            raise ExpertClientError(f"expert {self.model_name} request failed") from error
        if response.status != 200:
            raise ExpertClientError(f"expert {self.model_name} returned HTTP {response.status}")
        try:
            payload = json.loads(body.decode("utf-8"))
            logits, features = parse_infer_response(
                payload, expected_features=self.feature_dim if include_features else None
            )
        except (UnicodeDecodeError, json.JSONDecodeError, V2ProtocolError) as error:
            raise ExpertClientError(f"expert {self.model_name} returned an invalid response") from error
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ExpertClientError(f"expert {self.model_name} returned invalid parameters")
        process_cpu = parameters.get("process_cpu_seconds", 0.0)
        if not isinstance(process_cpu, (int, float)) or not np.isfinite(process_cpu) or process_cpu < 0:
            raise ExpertClientError(f"expert {self.model_name} returned invalid CPU time")
        return ExpertResult(
            logits=logits,
            features=features,
            process_cpu_seconds=float(process_cpu),
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()


@dataclass(frozen=True)
class RouterResult:
    logits: np.ndarray
    route: int
    expert_id: str
    extractor_seconds: float
    dispatcher_seconds: float
    expert_seconds: float
    router_cpu_seconds: float
    upstream_cpu_seconds: float


class DynamicRouter:
    """Run extractor once and reuse its classifier logits when it is selected."""

    def __init__(
        self,
        model_name: str,
        promotion_id: str,
        dispatcher_id: str,
        expert_order: list[str],
        extractor_id: str,
        clients: dict[str, ExpertClient],
        dispatcher: Dispatcher,
        *,
        feature_dim: int,
    ) -> None:
        if len(expert_order) == 0 or len(expert_order) != len(set(expert_order)):
            raise ValueError("expert_order must contain unique expert IDs")
        if extractor_id not in expert_order or set(clients) != set(expert_order):
            raise ValueError("clients and extractor must exactly match expert_order")
        self.model_name = model_name
        self.promotion_id = promotion_id
        self.dispatcher_id = dispatcher_id
        self.expert_order = tuple(expert_order)
        self.extractor_id = extractor_id
        self.clients = clients.copy()
        self.dispatcher = dispatcher
        self.feature_dim = feature_dim

    def predict(self, image: np.ndarray) -> RouterResult:
        image = _validate_image(image)
        cpu_started = time.process_time()
        started = time.perf_counter()
        extracted = _validate_result(
            self.clients[self.extractor_id].predict(image, include_features=True), self.feature_dim
        )
        after_extractor = time.perf_counter()
        route_logits = np.asarray(self.dispatcher.predict(extracted.features))
        if route_logits.shape != (1, len(self.expert_order)) or not np.isfinite(route_logits).all():
            raise ValueError("dispatcher logits have invalid shape or values")
        route = int(route_logits.argmax(axis=1)[0])
        after_dispatcher = time.perf_counter()
        expert_id = self.expert_order[route]
        if expert_id == self.extractor_id:
            logits = extracted.logits
            expert_seconds = 0.0
            upstream_cpu_seconds = extracted.process_cpu_seconds
        else:
            expert_started = time.perf_counter()
            selected = _validate_result(self.clients[expert_id].predict(image))
            logits = selected.logits
            expert_seconds = time.perf_counter() - expert_started
            upstream_cpu_seconds = extracted.process_cpu_seconds + selected.process_cpu_seconds
        return RouterResult(
            logits=logits,
            route=route,
            expert_id=expert_id,
            extractor_seconds=after_extractor - started,
            dispatcher_seconds=after_dispatcher - after_extractor,
            expert_seconds=expert_seconds,
            router_cpu_seconds=time.process_time() - cpu_started,
            upstream_cpu_seconds=upstream_cpu_seconds,
        )

    def infer(self, image: np.ndarray, request_id: str | None = None) -> dict[str, Any]:
        result = self.predict(image)
        return infer_response(
            self.model_name,
            result.logits,
            request_id=request_id,
            route=result.route,
            parameters={
                "promotion_id": self.promotion_id,
                "dispatcher_id": self.dispatcher_id,
                "expert_id": result.expert_id,
                "extractor_seconds": result.extractor_seconds,
                "dispatcher_seconds": result.dispatcher_seconds,
                "expert_seconds": result.expert_seconds,
                "router_cpu_seconds": result.router_cpu_seconds,
                "upstream_cpu_seconds": result.upstream_cpu_seconds,
                "process_cpu_seconds": result.router_cpu_seconds + result.upstream_cpu_seconds,
            },
        )


class StaticRouter:
    """Control arm with the same router-to-expert HTTP hop and a fixed route."""

    def __init__(
        self,
        model_name: str,
        promotion_id: str,
        expert_id: str,
        route: int,
        client: ExpertClient,
    ) -> None:
        if route < 0:
            raise ValueError("route must be non-negative")
        self.model_name = model_name
        self.promotion_id = promotion_id
        self.expert_id = expert_id
        self.route = route
        self.client = client

    def infer(self, image: np.ndarray, request_id: str | None = None) -> dict[str, Any]:
        cpu_started = time.process_time()
        started = time.perf_counter()
        result = _validate_result(self.client.predict(_validate_image(image)))
        duration = time.perf_counter() - started
        router_cpu = time.process_time() - cpu_started
        return infer_response(
            self.model_name,
            result.logits,
            request_id=request_id,
            route=self.route,
            parameters={
                "promotion_id": self.promotion_id,
                "dispatcher_id": "static",
                "expert_id": self.expert_id,
                "extractor_seconds": 0.0,
                "dispatcher_seconds": 0.0,
                "expert_seconds": duration,
                "router_cpu_seconds": router_cpu,
                "upstream_cpu_seconds": result.process_cpu_seconds,
                "process_cpu_seconds": router_cpu + result.process_cpu_seconds,
            },
        )
