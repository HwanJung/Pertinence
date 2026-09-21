import json

import numpy as np
import pytest

from pertinence.serving import ExpertModel, ExpertResult, V2Application, V2ProtocolError
from pertinence.serving.v2 import IMAGE_SHAPE, parse_infer_request


def request(data=None, **tensor_updates):
    tensor = {
        "name": "image",
        "shape": list(IMAGE_SHAPE),
        "datatype": "FP32",
        "data": data if data is not None else [0.0] * int(np.prod(IMAGE_SHAPE)),
    }
    tensor.update(tensor_updates)
    return {"id": "request-1", "inputs": [tensor]}


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"name": "pixels"}, "name"),
        ({"shape": [1, 3, 32, 31]}, "shape"),
        ({"datatype": "FP64"}, "FP32"),
        ({"data": [0.0]}, "element count"),
        ({"data": [False] * int(np.prod(IMAGE_SHAPE))}, "numeric"),
        ({"data": [float("nan")] * int(np.prod(IMAGE_SHAPE))}, "finite"),
    ],
)
def test_request_rejects_wrong_shape_dtype_and_values(updates, message) -> None:
    with pytest.raises(V2ProtocolError, match=message):
        parse_infer_request(request(**updates))


def test_request_returns_exact_float32_batch() -> None:
    request_id, image = parse_infer_request(request())
    assert request_id == "request-1"
    assert image.shape == IMAGE_SHAPE
    assert image.dtype == np.float32


class Runner:
    def predict(self, image, *, include_features=False):
        assert image.dtype == np.float32
        features = np.arange(4, dtype=np.float32)[None, :] if include_features else None
        return ExpertResult(np.arange(10, dtype=np.float32)[None, :], features)


def test_application_serves_health_metadata_and_inference() -> None:
    app = V2Application(ExpertModel("resnet", Runner()))
    for path in ("/v2/health/live", "/v2/health/ready"):
        status, _, body = app.handle("GET", path)
        assert status == 200
        assert json.loads(body) == {}

    status, _, body = app.handle("GET", "/v2/models/resnet")
    assert status == 200
    assert json.loads(body)["inputs"][0]["shape"] == [1, 3, 32, 32]

    status, _, body = app.handle(
        "POST", "/v2/models/resnet/infer", json.dumps(request()).encode()
    )
    response = json.loads(body)
    assert status == 200
    assert response["id"] == "request-1"
    assert response["outputs"][0] == {
        "name": "logits",
        "shape": [1, 10],
        "datatype": "FP32",
        "data": list(map(float, range(10))),
    }


def test_application_returns_client_error_for_invalid_request() -> None:
    app = V2Application(ExpertModel("resnet", Runner()))
    status, _, body = app.handle("POST", "/v2/models/resnet/infer", b"not-json")
    assert status == 400
    assert "error" in json.loads(body)


def test_extractor_exposes_features_only_when_configured_and_requested() -> None:
    app = V2Application(ExpertModel("shuffle", Runner(), feature_dim=4))
    status, _, body = app.handle("GET", "/v2/models/shuffle")
    assert status == 200
    assert [item["name"] for item in json.loads(body)["outputs"]] == ["logits", "features"]

    payload = request()
    payload["outputs"] = [{"name": "logits"}, {"name": "features"}]
    status, _, body = app.handle(
        "POST", "/v2/models/shuffle/infer", json.dumps(payload).encode()
    )
    assert status == 200
    outputs = json.loads(body)["outputs"]
    assert outputs[1] == {
        "name": "features",
        "shape": [1, 4],
        "datatype": "FP32",
        "data": [0.0, 1.0, 2.0, 3.0],
    }

    plain = V2Application(ExpertModel("resnet", Runner()))
    status, _, body = plain.handle(
        "POST", "/v2/models/resnet/infer", json.dumps(payload).encode()
    )
    assert status == 400
    assert "requested outputs" in json.loads(body)["error"]
