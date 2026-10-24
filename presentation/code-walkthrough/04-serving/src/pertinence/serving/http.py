"""Dependency-light KServe V2 HTTP application and server."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time
import threading
import socket
from typing import Any, Protocol

import numpy as np

from . import binary
from .diagnostics import Diagnostics, OFF, identity
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
        binary_output: bool = False,
        diagnostics: Diagnostics = OFF,
    ) -> dict[str, Any]:
        allowed = ("logits", "features") if self.feature_dim is not None else ("logits",)
        if requested_outputs not in {("logits",), allowed}:
            raise V2ProtocolError(f"requested outputs must be exactly {list(allowed)}")
        include_features = requested_outputs == ("logits", "features")
        cpu_started = time.process_time()
        with diagnostics.span("expert_model"):
            result = self.runner.predict(image, include_features=include_features)
        from .runtime import _validate_result

        with diagnostics.span("expert_output"):
            _validate_result(result, self.feature_dim if include_features else None)
            if binary_output:
                response = {"model_name": self.model_name, "model_version": "1",
                            "outputs": [{"name": "logits", "data": result.logits}]}
                if request_id is not None:
                    response["id"] = request_id
            else:
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
                response["outputs"].append(
                    {"name": "features", "data": features} if binary_output
                    else tensor_output("features", "FP32", features)
                )
        return response


class V2Application:
    """Pure request handler used both by tests and the stdlib HTTP server."""

    def __init__(self, model: InferenceModel, *, router: bool = False,
                 diagnostics: Diagnostics = OFF, run_fingerprint: str | None = None) -> None:
        self.run_fingerprint = run_fingerprint
        self.model = model
        self.router = router
        self.ready = True
        self.diagnostics = diagnostics
        self._model_lock = threading.Semaphore(1)

    def handle(self, method: str, path: str, body: bytes = b"",
               headers: Any = None) -> tuple[int, dict[str, str], bytes]:
        headers = headers or {}
        token = identity.set({"request_id": headers.get("X-Request-Id"),
                              "expert": self.model.model_name,
                              "call_index": headers.get("X-Call-Index")})
        try:
            try:
                binary_length = headers.get(binary.HEADER)
                if binary_length is not None and self.router:
                    raise V2ProtocolError("router accepts JSON only")
                status, response = self._handle(method, path, body, binary_length)
                if status == 200 and "outputs" in response and self.run_fingerprint is not None:
                    response.setdefault("parameters", {})["run_fingerprint"] = self.run_fingerprint
                if status == 200 and binary_length is not None:
                    with self.diagnostics.span("expert_encode") as record:
                        encoded, result_headers = binary.encode(response, "outputs")
                        record["body_bytes"] = len(encoded)
                    return status, result_headers, encoded
            except V2ProtocolError as error:
                status, response = 400, {"error": str(error)}
            except TimeoutError:
                status, response = 504, {"error": "upstream expert timed out"}
            except Exception as error:
                status, response = 500, {"error": str(error)}
            with self.diagnostics.span("response_encode") as record:
                encoded = json.dumps(response, separators=(",", ":"), allow_nan=False).encode()
                record["body_bytes"] = len(encoded)
            return status, {"Content-Type": "application/json",
                            "Content-Length": str(len(encoded))}, encoded
        finally:
            identity.reset(token)

    def _handle(self, method: str, path: str, body: bytes, binary_length: str | None = None) -> tuple[int, dict[str, Any]]:
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
            with self.diagnostics.span("request_decode", body_bytes=len(body)):
                request = (decode_json(body) if binary_length is None
                           else binary.decode(body, binary_length, "inputs"))
                request_id, image = parse_infer_request(request)
                outputs = requested_output_names(request)
            identity.set({**identity.get(), "request_id": request_id})
            if isinstance(self.model, ExpertModel):
                with self._model_lock:
                    return 200, self.model.infer(
                        image, request_id, requested_outputs=outputs,
                        binary_output=binary_length is not None, diagnostics=self.diagnostics,
                    )
            expected = ("logits", "route")
            if "outputs" in request and outputs != expected:
                raise V2ProtocolError(f"requested outputs must be exactly {list(expected)}")
            with self._model_lock:
                return 200, self.model.infer(image, request_id)
        return 404, {"error": "not found"}


def make_server(application: V2Application, host: str = "0.0.0.0", port: int = 8080,
                *, keep_alive: bool = False, idle_timeout: float = 10.0) -> ThreadingHTTPServer:
    """Serve a model with one Python request worker."""

    if idle_timeout <= 0:
        raise ValueError("idle_timeout must be positive")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1" if keep_alive else "HTTP/1.0"

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(idle_timeout)
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def _dispatch(self) -> None:
            with application.diagnostics.span(
                "request_total", scope="total", request_id=self.headers.get("X-Request-Id"),
                expert=application.model.model_name, call_index=self.headers.get("X-Call-Index"),
            ):
                self._dispatch_request()

        def _dispatch_request(self) -> None:
            length = self.headers.get("Content-Length", "0")
            try:
                body_length = int(length)
            except ValueError:
                body_length = -1
            if (body_length < 0 or body_length > 1_000_000
                    or self.headers.get("Transfer-Encoding") is not None):
                self.close_connection = True
                status, headers, body = application.handle("POST", self.path, b"{" * 1_000_001)
            else:
                body = self.rfile.read(body_length)
                status, headers, body = application.handle(self.command, self.path, body, self.headers)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            if keep_alive and not self.close_connection:
                self.send_header("Keep-Alive", f"timeout={idle_timeout:g}")
            self.end_headers()
            with application.diagnostics.span(
                "response_write", request_id=self.headers.get("X-Request-Id"),
                expert=application.model.model_name,
                call_index=self.headers.get("X-Call-Index"), body_bytes=len(body),
            ):
                self.wfile.write(body)
                self.wfile.flush()

        do_GET = _dispatch
        do_POST = _dispatch

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def serve(application: V2Application, host: str = "0.0.0.0", port: int = 8080,
          *, keep_alive: bool = False, idle_timeout: float = 10.0) -> None:
    with make_server(application, host, port, keep_alive=keep_alive,
                     idle_timeout=idle_timeout) as server:
        server.serve_forever()
