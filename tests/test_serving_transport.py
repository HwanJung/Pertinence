"""Binary transport and real HTTP connection regression tests."""

from contextlib import contextmanager
import http.client
import json
import threading
import time

import numpy as np
import pytest

from pertinence.serving import binary
from pertinence.serving.diagnostics import Diagnostics
from pertinence.serving.http import ExpertModel, V2Application, make_server
from pertinence.serving.runtime import (
    DynamicRouter, ExpertClientError, ExpertResult, HttpExpertClient,
    HybridDynamicRouter, OnnxExpertRunner,
)
from pertinence.serving.v2 import V2ProtocolError, parse_infer_request


class Runner:
    def __init__(self):
        self.calls = 0

    def predict(self, image, *, include_features=False):
        self.calls += 1
        return ExpertResult(np.arange(10, dtype=np.float32)[None],
                            np.ones((1, 1024), dtype=np.float32) if include_features else None)


@contextmanager
def endpoint(*, keep_alive=False, idle_timeout=10, diagnostics=None):
    app = V2Application(ExpertModel("expert", Runner(), feature_dim=1024),
                        **({"diagnostics": diagnostics} if diagnostics else {}))
    server = make_server(app, "127.0.0.1", 0, keep_alive=keep_alive, idle_timeout=idle_timeout)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", server, app
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def request():
    image = np.random.default_rng(1).normal(size=(1, 3, 32, 32)).astype(np.float32)
    return {"id": "request-1", "inputs": [{"name": "image", "data": image}]}


def test_binary_exact_fp32():
    value = request()
    body, headers = binary.encode(value, "inputs")
    result = binary.decode(body, headers[binary.HEADER], "inputs")
    request_id, image = parse_infer_request(result)
    assert request_id == "request-1"
    assert image.tobytes() == value["inputs"][0]["data"].tobytes()
    assert len(body) - int(headers[binary.HEADER]) == 12288


@pytest.mark.parametrize("mutation", ["short", "long", "shape", "dtype", "nan", "inf", "size"])
def test_binary_rejects_invalid(mutation):
    body, headers = binary.encode(request(), "inputs")
    length = int(headers[binary.HEADER])
    header, data = json.loads(body[:length]), body[length:]
    if mutation == "short":
        data = data[:-1]
    elif mutation == "long":
        data += b"x"
    elif mutation == "shape":
        header["inputs"][0]["shape"] = [0]
    elif mutation == "dtype":
        header["inputs"][0]["datatype"] = "FP64"
    elif mutation == "size":
        header["inputs"][0]["parameters"]["binary_data_size"] = True
    else:
        data = np.array([float(mutation)], dtype="<f4").tobytes() + data[4:]
    prefix = json.dumps(header).encode()
    with pytest.raises(V2ProtocolError):
        binary.decode(prefix + data, str(len(prefix)), "inputs")


@pytest.mark.parametrize("transport", ["json", "binary"])
@pytest.mark.parametrize("keep_alive", [False, True])
def test_connection_reuse_and_probe(transport, keep_alive, tmp_path):
    diagnostics = Diagnostics(tmp_path / "spans.jsonl")
    with endpoint(keep_alive=keep_alive, diagnostics=diagnostics) as (url, server, _):
        client = HttpExpertClient(url, "expert", timeout_seconds=1, feature_dim=1024,
                                  transport=transport, diagnostics=diagnostics)
        try:
            for index in range(3):
                result = client.predict(request()["inputs"][0]["data"], include_features=True,
                                        request_id=f"r-{index}")
                assert result.features.shape == (1, 1024)
                assert result.logits.tobytes() == np.arange(10, dtype=np.float32).tobytes()
                probe = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=1)
                probe.request("GET", "/v2/health/ready")
                response = probe.getresponse()
                assert response.status == 200
                response.read()
                probe.close()
            assert client.connections_created == (1 if keep_alive else 3)
            assert client.connections_reused == (2 if keep_alive else 0)
        finally:
            client.close()
    records = [json.loads(line) for line in diagnostics.path.read_text().splitlines()]
    assert any(row.get("request_id") == "r-0" and row["stage"] == "expert_model"
               for row in records)
    assert all(row["thread_cpu_seconds"] >= 0 and row["wall_seconds"] >= 0 for row in records)


def test_idle_disconnect_is_not_retried():
    with endpoint(keep_alive=True, idle_timeout=0.05) as (url, _, app):
        client = HttpExpertClient(url, "expert", timeout_seconds=1, transport="binary")
        image = request()["inputs"][0]["data"]
        client.predict(image)
        time.sleep(0.1)
        with pytest.raises(ExpertClientError):
            client.predict(image)
        assert app.model.runner.calls == 1
        client.predict(image)
        assert app.model.runner.calls == 2
        assert client.connections_created == 2
        client.close()


def test_binary_bypasses_json_response(monkeypatch):
    import pertinence.serving.http as module

    def forbidden(*args, **kwargs):
        raise AssertionError("JSON output used")

    monkeypatch.setattr(module, "infer_response", forbidden)
    app = V2Application(ExpertModel("expert", Runner(), feature_dim=1024))
    value = request()
    value["outputs"] = [{"name": "logits"}, {"name": "features"}]
    body, headers = binary.encode(value, "inputs")
    status, response_headers, response = app.handle("POST", "/v2/models/expert/infer", body, headers)
    assert status == 200
    assert binary.decode(response, response_headers[binary.HEADER], "outputs")["outputs"][1][
        "data"].shape == (1, 1024)
    value["outputs"] = [{"name": "features"}]
    body, headers = binary.encode(value, "inputs")
    assert app.handle("POST", "/v2/models/expert/infer", body, headers)[0] == 400


@pytest.mark.parametrize("route", [0, 1, 2])
def test_hybrid_routes(route):
    class Local(OnnxExpertRunner, Runner):
        def __init__(self):
            Runner.__init__(self)

        predict = Runner.predict

    class Remote(HttpExpertClient, Runner):
        def __init__(self):
            Runner.__init__(self)
            self.transport = "binary"

        def predict(self, image, *, include_features=False, **kwargs):
            return Runner.predict(self, image, include_features=include_features)

    class Dispatcher:
        def predict(self, features):
            return np.eye(3, dtype=np.float32)[route:route + 1]

    clients = {"s": Local(), "m": Remote(), "r": Remote()}
    router = HybridDynamicRouter("router", "promotion", "dispatcher", list(clients), "s",
                                 clients, Dispatcher(), feature_dim=1024)
    result = router.predict(request()["inputs"][0]["data"])
    assert result.route == route
    assert [client.calls for client in clients.values()] == [1, int(route == 1), int(route == 2)]
    assert isinstance(router, DynamicRouter)


def test_model_is_serialized_while_probe_remains_responsive():
    from concurrent.futures import ThreadPoolExecutor

    started, release = threading.Event(), threading.Event()

    class BlockingRunner(Runner):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.maximum = 0

        def predict(self, image, *, include_features=False):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            started.set()
            assert release.wait(2)
            result = super().predict(image, include_features=include_features)
            self.active -= 1
            return result

    with endpoint(keep_alive=True) as (url, server, app):
        runner = BlockingRunner()
        app.model.runner = runner
        clients = [HttpExpertClient(url, "expert", timeout_seconds=3) for _ in range(2)]
        try:
            with ThreadPoolExecutor(max_workers=2) as workers:
                futures = [workers.submit(client.predict, request()["inputs"][0]["data"])
                           for client in clients]
                assert started.wait(1)
                probe = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=1)
                try:
                    probe.request("GET", "/v2/health/ready")
                    response = probe.getresponse()
                    assert response.status == 200
                    response.read()
                finally:
                    probe.close()
                    release.set()
                for future in futures:
                    future.result()
            assert runner.calls == 2
            assert runner.maximum == 1
        finally:
            release.set()
            for client in clients:
                client.close()


def test_distributed_shares_one_image_byte_buffer():
    buffers = []

    class Remote(HttpExpertClient):
        def __init__(self):
            self.transport = "binary"

        def predict(self, image, *, include_features=False, image_bytes=None, **kwargs):
            buffers.append(image_bytes)
            return Runner().predict(image, include_features=include_features)

    class Dispatcher:
        def predict(self, features):
            return np.array([[0, 1, 0]], dtype=np.float32)

    clients = {name: Remote() for name in ("s", "m", "r")}
    router = DynamicRouter("router", "promotion", "dispatcher", list(clients), "s", clients,
                           Dispatcher(), feature_dim=1024)
    router.predict(request()["inputs"][0]["data"])
    assert len(buffers) == 2
    assert buffers[0] is buffers[1]
    assert len(buffers[0]) == 12288


@pytest.mark.parametrize("route", [0, 1, 2])
def test_hybrid_uses_one_binary_http10_call_only_for_selected_remote(route):
    class Local(OnnxExpertRunner, Runner):
        def __init__(self):
            Runner.__init__(self)

        predict = Runner.predict

    class Dispatcher:
        def predict(self, features):
            return np.eye(3, dtype=np.float32)[route:route + 1]

    with endpoint() as (url1, server1, app1), endpoint() as (url2, server2, app2):
        assert server1.RequestHandlerClass.protocol_version == "HTTP/1.0"
        assert server2.RequestHandlerClass.protocol_version == "HTTP/1.0"
        local = Local()
        first = HttpExpertClient(url1, "expert", timeout_seconds=1, transport="binary")
        second = HttpExpertClient(url2, "expert", timeout_seconds=1, transport="binary")
        router = HybridDynamicRouter("router", "promotion", "dispatcher", ["s", "m", "r"],
                                     "s", {"s": local, "m": first, "r": second},
                                     Dispatcher(), feature_dim=1024)
        try:
            result = router.predict(request()["inputs"][0]["data"])
            np.testing.assert_array_equal(result.logits, np.arange(10, dtype=np.float32)[None])
            assert result.route == route and router.serving_mode == "hybrid"
            assert [local.calls, app1.model.runner.calls, app2.model.runner.calls] == [
                1, int(route == 1), int(route == 2),
            ]
            assert first.connections_created + second.connections_created == int(route != 0)
            assert first.connections_reused == second.connections_reused == 0
            if route == 0:
                assert result.expert_seconds == 0
        finally:
            first.close()
            second.close()
