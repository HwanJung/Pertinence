import hashlib

import numpy as np
import pytest

from pertinence.serving import (
    DynamicRouter,
    ExpertClientError,
    ExpertModel,
    ExpertResult,
    FusedDynamicRouter,
    LinearDispatcherRunner,
    StaticRouter,
    load_dispatcher_npz,
    OnnxExpertRunner,
)
from pertinence.serving.cli.common import _verified_model_path, _wait_for_modelcar


class Client:
    def __init__(self, expert: int, *, feature_dim: int = 4, error=None):
        self.expert = expert
        self.feature_dim = feature_dim
        self.error = error
        self.calls = []

    def predict(self, image, *, include_features=False):
        self.calls.append(include_features)
        if self.error:
            raise self.error
        logits = np.full((1, 10), self.expert, dtype=np.float32)
        features = np.arange(self.feature_dim, dtype=np.float32)[None, :] if include_features else None
        return ExpertResult(logits, features)


class LocalRunner(OnnxExpertRunner):
    """Test double that preserves the fused router's concrete runner boundary."""

    def __init__(self, expert: int, *, feature_dim: int = 4):
        self.expert = expert
        self.feature_dim = feature_dim
        self.calls = []

    def predict(self, image, *, include_features=False):
        self.calls.append(include_features)
        logits = np.full((1, 10), self.expert, dtype=np.float32)
        features = np.arange(self.feature_dim, dtype=np.float32)[None, :] if include_features else None
        return ExpertResult(logits, features)


def image():
    return np.zeros((1, 3, 32, 32), dtype=np.float32)


def router(route: int):
    clients = {name: Client(index) for index, name in enumerate(["shuffle", "mobile", "resnet"])}
    weight = np.zeros((3, 4), dtype=np.float32)
    bias = np.zeros(3, dtype=np.float32)
    bias[route] = 1
    dynamic = DynamicRouter(
        "dynamic",
        "promotion-1",
        "dispatcher-000733",
        list(clients),
        "shuffle",
        clients,
        LinearDispatcherRunner(weight, bias),
        feature_dim=4,
    )
    return dynamic, clients


def test_dynamic_router_reuses_extractor_logits_when_route_zero() -> None:
    dynamic, clients = router(0)
    response = dynamic.infer(image(), "r1")

    assert clients["shuffle"].calls == [True]
    assert clients["mobile"].calls == []
    assert clients["resnet"].calls == []
    assert response["outputs"][0]["data"] == [0.0] * 10
    assert response["outputs"][1]["data"] == [0]
    assert response["parameters"]["expert_id"] == "shuffle"
    assert response["parameters"]["promotion_id"] == "promotion-1"


def test_dynamic_router_runs_only_selected_remote_expert() -> None:
    dynamic, clients = router(2)
    result = dynamic.predict(image())

    assert clients["shuffle"].calls == [True]
    assert clients["mobile"].calls == []
    assert clients["resnet"].calls == [False]
    assert result.route == 2
    assert result.logits.tolist() == [[2.0] * 10]


@pytest.mark.parametrize("route", [0, 1, 2])
def test_fused_router_reuses_extractor_and_runs_only_selected_local_expert(route: int) -> None:
    names = ["shuffle", "mobile", "resnet"]
    runners = {name: LocalRunner(index) for index, name in enumerate(names)}
    weight = np.zeros((3, 4), dtype=np.float32)
    bias = np.zeros(3, dtype=np.float32)
    bias[route] = 1
    fused = FusedDynamicRouter(
        "dispatcher-000733",
        "promotion-1",
        "dispatcher-000733",
        names,
        names[0],
        runners,
        LinearDispatcherRunner(weight, bias),
        feature_dim=4,
    )

    response = fused.infer(image())

    assert runners["shuffle"].calls == [True]
    assert runners["mobile"].calls == ([False] if route == 1 else [])
    assert runners["resnet"].calls == ([False] if route == 2 else [])
    assert response["outputs"][0]["data"] == [float(route)] * 10
    assert response["outputs"][1]["data"] == [route]
    assert response["parameters"]["serving_mode"] == "fused"


def test_dynamic_router_propagates_upstream_error_without_fallback_or_retry() -> None:
    dynamic, clients = router(1)
    clients["mobile"].error = ExpertClientError("timed out")

    with pytest.raises(ExpertClientError, match="timed out"):
        dynamic.predict(image())

    assert clients["shuffle"].calls == [True]
    assert clients["mobile"].calls == [False]
    assert clients["resnet"].calls == []


def test_static_router_has_same_single_expert_hop_and_identity() -> None:
    client = Client(2)
    static = StaticRouter("static-resnet", "promotion-1", "resnet", 2, client)
    response = static.infer(image())

    assert client.calls == [False]
    assert response["outputs"][1]["data"] == [2]
    assert response["parameters"]["dispatcher_id"] == "static"
    assert response["parameters"]["expert_id"] == "resnet"


def test_direct_expert_includes_serving_mode_identity() -> None:
    direct = ExpertModel("resnet56", Client(2), identity={"serving_mode": "direct"})

    response = direct.infer(image())

    assert response["model_name"] == "resnet56"
    assert response["parameters"]["serving_mode"] == "direct"


def test_dispatcher_loader_accepts_only_exact_fp32_npz(tmp_path) -> None:
    path = tmp_path / "dispatcher.npz"
    np.savez(
        path,
        **{
            "linear.weight": np.zeros((3, 1024), dtype=np.float32),
            "linear.bias": np.zeros(3, dtype=np.float32),
        },
    )
    dispatcher = load_dispatcher_npz(path)
    assert dispatcher.predict(np.zeros((1, 1024), dtype=np.float32)).shape == (1, 3)

    wrong = tmp_path / "wrong.npz"
    np.savez(
        wrong,
        **{
            "linear.weight": np.zeros((3, 1024), dtype=np.float64),
            "linear.bias": np.zeros(3, dtype=np.float32),
        },
    )
    with pytest.raises(ValueError, match="FP32"):
        load_dispatcher_npz(wrong)
    with pytest.raises(ValueError, match=r"\.npz"):
        load_dispatcher_npz(tmp_path / "dispatcher.pt")


def test_router_cpu_includes_router_and_upstream_cpu() -> None:
    dynamic, clients = router(2)
    clients["shuffle"].predict = lambda image, include_features=False: ExpertResult(
        np.zeros((1, 10), dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32) if include_features else None,
        0.25,
    )
    clients["resnet"].predict = lambda image, include_features=False: ExpertResult(
        np.zeros((1, 10), dtype=np.float32), None, 0.5
    )
    response = dynamic.infer(image())
    assert response["parameters"]["upstream_cpu_seconds"] == pytest.approx(0.75)
    assert response["parameters"]["process_cpu_seconds"] >= 0.75


def test_wait_for_modelcar_retries_until_loader_is_ready() -> None:
    attempts = 0

    def load() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("copy in progress")
        return "ready"

    assert _wait_for_modelcar(load, timeout_seconds=1, poll_seconds=0.001) == "ready"
    assert attempts == 3


def test_wait_for_modelcar_times_out_with_original_failure() -> None:
    def missing() -> str:
        raise FileNotFoundError("not copied")

    with pytest.raises(TimeoutError, match="startup timeout") as caught:
        _wait_for_modelcar(missing, timeout_seconds=0, poll_seconds=0.001)

    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_verified_model_path_requires_promoted_bytes(tmp_path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"complete")

    assert _verified_model_path(model, hashlib.sha256(b"complete").hexdigest()) == model
    with pytest.raises(ValueError, match="SHA-256"):
        _verified_model_path(model, "0" * 64)
