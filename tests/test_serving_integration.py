from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pertinence.serving import integration
from pertinence.serving.integration import (
    EXPERT_IDS,
    IntegrationExpectations,
    _ExpertSession,
    _infer,
    array_sha256,
    integrate_serving_artifacts,
)
from pertinence.serving.v2 import parse_infer_request


class _BatchSession:
    def __init__(self, logits: np.ndarray, features: np.ndarray | None = None) -> None:
        self.logits = logits
        self.features = features
        self.offset = 0

    def predict(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        stop = self.offset + len(images)
        result = self.logits[self.offset : stop]
        features = None if self.features is None else self.features[self.offset : stop]
        self.offset = stop
        return result, features


def _logits(predictions: list[int]) -> np.ndarray:
    values = np.full((len(predictions), 10), -1.0, dtype=np.float32)
    values[np.arange(len(predictions)), predictions] = 1.0
    return values


def test_infer_routes_and_reuses_extractor_logits() -> None:
    routes = np.asarray([0, 1, 2, 2])
    features = np.zeros((4, 1024), dtype=np.float32)
    features[np.arange(4), routes] = 1.0
    dispatcher = SimpleNamespace(
        weight=np.eye(3, 1024, dtype=np.float32), bias=np.zeros(3, dtype=np.float32)
    )
    sessions = [
        _BatchSession(_logits([0, 0, 0, 0]), features),
        _BatchSession(_logits([1, 1, 1, 1])),
        _BatchSession(_logits([2, 2, 2, 3])),
    ]

    result = _infer(np.zeros((4, 3, 32, 32), dtype=np.float32), sessions, dispatcher)

    assert result["routes"].tolist() == routes.tolist()
    assert result["dispatcher_predictions"].tolist() == [0, 1, 2, 3]
    assert np.array_equal(result["selected_logits"][0], result["expert_logits"][0, 0])


def test_expert_session_rejects_wrong_feature_interface() -> None:
    def tensor(name, shape):
        return SimpleNamespace(name=name, shape=shape, type="tensor(float)")

    runtime = SimpleNamespace(
        session=SimpleNamespace(
            get_inputs=lambda: [tensor("image", ["batch", 3, 32, 32])],
            get_outputs=lambda: [
                tensor("logits", ["batch", 10]),
                tensor("features", ["batch", 1000]),
            ],
        )
    )

    with pytest.raises(ValueError, match="features has an unexpected static shape"):
        _ExpertSession(runtime, EXPERT_IDS[0])


def test_integration_writes_consumable_create_only_outputs(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    experts = tmp_path / "experts"
    evaluated = tmp_path / "evaluated"
    for path in (dataset, experts, evaluated):
        path.mkdir()
        (path / "marker").write_text(path.name, encoding="utf-8")
    for name in ("contract.json", "search-result.json", "final-report.json", "manifest.json"):
        (evaluated / name).write_text(name, encoding="utf-8")
    checkpoint = evaluated / "dispatcher-000733.npz"
    checkpoint.write_bytes(b"dispatcher")
    for expert_id in EXPERT_IDS:
        model = experts / f"{expert_id}.onnx"
        model.write_bytes(expert_id.encode())
    split = tmp_path / "split.json"
    split.write_text("{}", encoding="utf-8")
    final_ids = [f"sample-{index}" for index in range(4)]
    raw = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(4, dtype=np.int64)
    routes = np.asarray([0, 1, 2, 2])
    features = np.zeros((4, 1024), dtype=np.float32)
    features[np.arange(4), routes] = 1.0
    by_expert = {
        EXPERT_IDS[0]: _BatchSession(_logits([0, 0, 0, 0]), features),
        EXPERT_IDS[1]: _BatchSession(_logits([1, 1, 1, 1])),
        EXPERT_IDS[2]: _BatchSession(_logits([2, 2, 2, 3])),
    }
    dispatcher = SimpleNamespace(
        weight=np.eye(3, 1024, dtype=np.float32), bias=np.zeros(3, dtype=np.float32)
    )
    monkeypatch.setattr(
        integration,
        "_validate_evaluated_bundle",
        lambda *_: ({}, checkpoint, {"contract_fingerprint": "a" * 64}),
    )
    monkeypatch.setattr(
        integration,
        "_validate_inputs",
        lambda *_: (
            object(),
            final_ids,
            [
                {
                    "id": item,
                    "model_path": f"{item}.onnx",
                    "model_sha256": "c" * 64,
                }
                for item in EXPERT_IDS
            ],
        ),
    )
    monkeypatch.setattr(integration, "_load_final_rows", lambda *_: (raw, labels))
    monkeypatch.setattr(integration, "ExpertRuntime", lambda *args, **kwargs: object())
    monkeypatch.setattr(integration, "load_dispatcher_npz", lambda _: dispatcher)
    expectations = IntegrationExpectations(
        samples=4,
        sample_order_sha256="b" * 64,
        correct={
            EXPERT_IDS[0]: 1,
            EXPERT_IDS[1]: 1,
            EXPERT_IDS[2]: 2,
            "dispatcher-000733": 4,
        },
        route_counts=(1, 1, 2),
        expert_model_sha256={item: "c" * 64 for item in EXPERT_IDS},
    )
    output = tmp_path / "output"

    receipt = integrate_serving_artifacts(
        dataset_bundle=dataset,
        split_manifest=split,
        expert_bundle=experts,
        evaluated_bundle=evaluated,
        dispatcher_id="dispatcher-000733",
        approved_dispatcher_sha256="d" * 64,
        approved_label_sha256=array_sha256(labels),
        output=output,
        expectations=expectations,
        session_factory=lambda _runtime, expert_id: by_expert[expert_id],
    )

    assert receipt["valid"] is True
    assert set(path.name for path in output.iterdir()) == {
        "evaluation-final.npz",
        "local-predictions.npz",
        "request.json",
        "receipt.json",
        "build-bundle",
    }
    for expert_id in EXPERT_IDS:
        model_root = output / "build-bundle" / "experts" / expert_id
        assert (model_root / "model.onnx").read_bytes() == expert_id.encode()
        assert (model_root / "model.sha256").read_text().startswith("c" * 64)
    with np.load(output / "local-predictions.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            *(f"{item}_predictions" for item in EXPERT_IDS),
            *(f"{item}_logits" for item in EXPERT_IDS),
            "static-resnet-router_predictions",
            "static-resnet-router_logits",
            "dispatcher-000733_predictions",
            "dispatcher-000733_routes",
            "dispatcher-000733_logits",
        }
        assert archive["dispatcher-000733_logits"].dtype == np.float32
        assert np.array_equal(
            archive["static-resnet-router_logits"], archive["resnet56_logits"]
        )
    with np.load(output / "evaluation-final.npz", allow_pickle=False) as archive:
        assert archive["images"].dtype == np.uint8
        assert archive["images"].shape == (4, 32, 32, 3)
    request = __import__("json").loads((output / "request.json").read_text(encoding="utf-8"))
    _, image = parse_infer_request(request)
    assert image.shape == (1, 3, 32, 32)

    with pytest.raises(ValueError, match="output already exists"):
        integrate_serving_artifacts(
            dataset_bundle=dataset,
            split_manifest=split,
            expert_bundle=experts,
            evaluated_bundle=evaluated,
            dispatcher_id="dispatcher-000733",
            approved_dispatcher_sha256="d" * 64,
            approved_label_sha256=array_sha256(labels),
            output=output,
            expectations=expectations,
        )
