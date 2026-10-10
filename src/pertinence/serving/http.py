"""Dependency-light KServe V2 HTTP application and server."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import time
from typing import Any, Protocol

import numpy as np

from .v2 import (
    V2ProtocolError,
    decode_json,
    infer_response,
    model_metadata,
    parse_infer_request,
    requested_output_names,
    tensor_output,
)


class InferenceModel(Protocol):
    model_name: str

    def infer(self, image: np.ndarray, request_id: str | None = None) -> dict[str, Any]: ...


class ExpertModel:
    """Adapt an expert runner to the public logits-only V2 contract."""

    def __init__(
        self,
        model_name: str,
        runner: Any,
        *,
        feature_dim: int | None = None,
        identity: dict[str, str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.runner = runner
        self.feature_dim = feature_dim
        self.identity = dict(identity or {})

    def infer(
        self,
        image: np.ndarray,
        request_id: str | None = None,
        *,
        requested_outputs: tuple[str, ...] = ("logits",),
    ) -> dict[str, Any]:
        allowed = ("logits", "features") if self.feature_dim is not None else ("logits",)
        if requested_outputs not in {("logits",), allowed}:
            raise V2ProtocolError(f"requested outputs must be exactly {list(allowed)}")
        include_features = requested_outputs == ("logits", "features")
        cpu_started = time.process_time()
        result = self.runner.predict(image, include_features=include_features)
        response = infer_response(self.model_name, result.logits, request_id=request_id)
        response["parameters"] = {
            **self.identity,
            "process_cpu_seconds": time.process_time() - cpu_started,
        }
        if include_features:
            features = np.asarray(result.features)
            if (
                features.shape != (1, self.feature_dim)
                or features.dtype != np.float32
                or not np.isfinite(features).all()
            ):
                raise ValueError("extractor features have invalid shape, dtype, or values")
            response["outputs"].append(tensor_output("features", "FP32", features))
        return response


class V2Application:
    """Pure request handler used both by tests and the stdlib HTTP server."""

    def __init__(self, model: InferenceModel, *, router: bool = False) -> None:
        self.model = model
        self.router = router
        self.ready = True

    def handle(self, method: str, path: str, body: bytes = b"") -> tuple[int, dict[str, str], bytes]:
        try:
            status, response = self._handle(method, path, body)
        except V2ProtocolError as error:
            status, response = 400, {"error": str(error)}
        except TimeoutError:
            status, response = 504, {"error": "upstream expert timed out"}
        except Exception as error:
            # Runtime and upstream failures are observable but never retried.
            status, response = 500, {"error": str(error)}
        encoded = json.dumps(response, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return status, {"Content-Type": "application/json", "Content-Length": str(len(encoded))}, encoded

    def _handle(self, method: str, path: str, body: bytes) -> tuple[int, dict[str, Any]]:
        if method == "GET" and path == "/v2/health/live":
            return 200, {}
        if method == "GET" and path == "/v2/health/ready":
            return (200, {}) if self.ready else (503, {"error": "model is not ready"})
        metadata_path = f"/v2/models/{self.model.model_name}"
        if method == "GET" and path == metadata_path:
            feature_dim = self.model.feature_dim if isinstance(self.model, ExpertModel) else None
            return 200, model_metadata(
                self.model.model_name, router=self.router, feature_dim=feature_dim
            )
        if method == "POST" and path == metadata_path + "/infer":
            if not self.ready:
                return 503, {"error": "model is not ready"}
            request = decode_json(body)
            request_id, image = parse_infer_request(request)
            outputs = requested_output_names(request)
            if isinstance(self.model, ExpertModel):
                return 200, self.model.infer(
                    image, request_id, requested_outputs=outputs
                )
            expected = ("logits", "route")
            if "outputs" in request and outputs != expected:
                raise V2ProtocolError(f"requested outputs must be exactly {list(expected)}")
            return 200, self.model.infer(image, request_id)
        return 404, {"error": "not found"}


def serve(application: V2Application, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Serve a model with one Python request worker."""

    class Handler(BaseHTTPRequestHandler):
        def _dispatch(self) -> None:
            length = self.headers.get("Content-Length", "0")
            try:
                body_length = int(length)
            except ValueError:
                body_length = -1
            if body_length < 0 or body_length > 1_000_000:
                status, headers, body = application.handle("POST", self.path, b"{" * 1_000_001)
            else:
                body = self.rfile.read(body_length)
                status, headers, body = application.handle(self.command, self.path, body)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        do_GET = _dispatch
        do_POST = _dispatch

        def log_message(self, format: str, *args: object) -> None:
            return

    server = HTTPServer((host, port), Handler)
    server.serve_forever()
