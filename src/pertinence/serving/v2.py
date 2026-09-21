"""Strict KServe V2 inference request and response helpers."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


IMAGE_SHAPE = (1, 3, 32, 32)


class V2ProtocolError(ValueError):
    """A request or upstream response violates the supported V2 subset."""


def _fields(value: Any, required: set[str], optional: set[str] | None = None) -> None:
    if not isinstance(value, dict):
        raise V2ProtocolError("expected a JSON object")
    missing = required - value.keys()
    unknown = value.keys() - required - (optional or set())
    if missing or unknown:
        raise V2ProtocolError(f"invalid fields: missing={sorted(missing)}, unknown={sorted(unknown)}")


def decode_json(body: bytes, *, max_bytes: int = 1_000_000) -> dict[str, Any]:
    if len(body) > max_bytes:
        raise V2ProtocolError("request body is too large")
    try:
        value = json.loads(body.decode("utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(
            V2ProtocolError(f"non-finite JSON number: {item}")
        ))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V2ProtocolError("request body must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise V2ProtocolError("request body must be a JSON object")
    return value


def parse_infer_request(value: dict[str, Any]) -> tuple[str | None, np.ndarray]:
    """Parse the supported V2 request: one FP32 image with exact batch-one shape."""
    _fields(value, {"inputs"}, {"id", "parameters", "outputs"})
    request_id = value.get("id")
    if request_id is not None and (not isinstance(request_id, str) or not request_id):
        raise V2ProtocolError("request id must be a non-empty string")
    if "parameters" in value and not isinstance(value["parameters"], dict):
        raise V2ProtocolError("request parameters must be an object")
    if "outputs" in value:
        outputs = value["outputs"]
        if not isinstance(outputs, list) or not outputs:
            raise V2ProtocolError("requested outputs must be a list of objects")
        names: list[str] = []
        for output in outputs:
            _fields(output, {"name"}, {"parameters"})
            if not isinstance(output["name"], str) or not output["name"]:
                raise V2ProtocolError("requested output names must be non-empty strings")
            names.append(output["name"])
        if len(names) != len(set(names)):
            raise V2ProtocolError("requested output names must be unique")
    inputs = value["inputs"]
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise V2ProtocolError("exactly one image input is required")
    tensor = inputs[0]
    _fields(tensor, {"name", "shape", "datatype", "data"}, {"parameters"})
    if tensor["name"] != "image":
        raise V2ProtocolError("input name must be image")
    if tensor["datatype"] != "FP32":
        raise V2ProtocolError("image datatype must be FP32")
    if tensor["shape"] != list(IMAGE_SHAPE):
        raise V2ProtocolError(f"image shape must be {list(IMAGE_SHAPE)}")
    data = tensor["data"]
    if not isinstance(data, list):
        raise V2ProtocolError("image data must be a JSON array")
    try:
        array = np.asarray(data)
    except (TypeError, ValueError) as error:
        raise V2ProtocolError("image data must contain numeric FP32 values") from error
    if array.dtype.kind not in "iuf" or array.dtype.kind == "b":
        raise V2ProtocolError("image data must contain numeric FP32 values")
    if array.size != math.prod(IMAGE_SHAPE):
        raise V2ProtocolError("image data element count does not match shape")
    try:
        array = np.asarray(data, dtype=np.float32).reshape(IMAGE_SHAPE)
    except (TypeError, ValueError, OverflowError) as error:
        raise V2ProtocolError("image data cannot be represented as FP32") from error
    if not np.isfinite(array).all():
        raise V2ProtocolError("image data must contain only finite values")
    return request_id, array


def requested_output_names(value: dict[str, Any]) -> tuple[str, ...]:
    """Return requested output names after ``parse_infer_request`` validates the request."""
    if "outputs" not in value:
        return ("logits",)
    return tuple(output["name"] for output in value["outputs"])


def tensor_output(name: str, datatype: str, array: np.ndarray) -> dict[str, Any]:
    if datatype not in {"FP32", "INT64"}:
        raise ValueError("unsupported V2 output datatype")
    expected = np.float32 if datatype == "FP32" else np.int64
    array = np.asarray(array, dtype=expected)
    if datatype == "FP32" and not np.isfinite(array).all():
        raise ValueError("output contains non-finite logits")
    return {
        "name": name,
        "shape": list(array.shape),
        "datatype": datatype,
        "data": array.reshape(-1).tolist(),
    }


def infer_response(
    model_name: str,
    logits: np.ndarray,
    *,
    request_id: str | None = None,
    route: int | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    logits = np.asarray(logits)
    if logits.shape != (1, 10):
        raise ValueError("logits must have shape [1, 10]")
    response: dict[str, Any] = {
        "model_name": model_name,
        "model_version": "1",
        "outputs": [tensor_output("logits", "FP32", logits)],
    }
    if request_id is not None:
        response["id"] = request_id
    if route is not None:
        response["outputs"].append(tensor_output("route", "INT64", np.asarray([route])))
    if parameters is not None:
        response["parameters"] = parameters
    return response


def parse_infer_response(
    value: dict[str, Any], *, expected_features: int | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Strictly validate an expert response received by a router."""
    _fields(value, {"model_name", "outputs"}, {"id", "model_version", "parameters"})
    outputs = value["outputs"]
    if not isinstance(outputs, list):
        raise V2ProtocolError("upstream outputs must be a list")
    by_name: dict[str, Any] = {}
    for output in outputs:
        _fields(output, {"name", "shape", "datatype", "data"}, {"parameters"})
        if output["name"] in by_name:
            raise V2ProtocolError("upstream output names must be unique")
        by_name[output["name"]] = output
    expected_names = {"logits"} | ({"features"} if expected_features is not None else set())
    if by_name.keys() != expected_names:
        raise V2ProtocolError(f"upstream outputs must be exactly {sorted(expected_names)}")

    def parse(name: str, shape: list[int], datatype: str, dtype: Any) -> np.ndarray:
        output = by_name[name]
        if output["shape"] != shape or output["datatype"] != datatype:
            raise V2ProtocolError(f"invalid upstream {name} shape or datatype")
        data = output["data"]
        if not isinstance(data, list) or any(type(item) not in (int, float) for item in data):
            raise V2ProtocolError(f"invalid upstream {name} data")
        array = np.asarray(data, dtype=dtype)
        if array.size != math.prod(shape) or not np.isfinite(array).all():
            raise V2ProtocolError(f"invalid upstream {name} values")
        return array.reshape(shape)

    logits = parse("logits", [1, 10], "FP32", np.float32)
    features = None
    if expected_features is not None:
        features = parse("features", [1, expected_features], "FP32", np.float32)
    return logits, features


def model_metadata(
    model_name: str, *, router: bool = False, feature_dim: int | None = None
) -> dict[str, Any]:
    outputs = [{"name": "logits", "datatype": "FP32", "shape": [1, 10]}]
    if router:
        outputs.append({"name": "route", "datatype": "INT64", "shape": [1]})
    if feature_dim is not None:
        outputs.append({"name": "features", "datatype": "FP32", "shape": [1, feature_dim]})
    return {
        "name": model_name,
        "versions": ["1"],
        "platform": "onnxruntime_onnx",
        "inputs": [{"name": "image", "datatype": "FP32", "shape": list(IMAGE_SHAPE)}],
        "outputs": outputs,
    }
