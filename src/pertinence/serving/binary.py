"""Strict little-endian FP32 V2 binary tensor extension."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from .v2 import V2ProtocolError, decode_json

HEADER = "Inference-Header-Content-Length"
CONTENT_TYPE = "application/octet-stream"


def encode(value: dict[str, Any], key: str, *, image_bytes: bytes | None = None
           ) -> tuple[bytes, dict[str, str]]:
    """Encode ndarray tensors directly, without materializing Python float lists."""
    header = dict(value)
    tensors, chunks = [], []
    for tensor in value[key]:
        array = np.asarray(tensor["data"])
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise V2ProtocolError("binary tensors must contain finite FP32 values")
        raw = image_bytes if image_bytes is not None and tensor["name"] == "image" else (
            np.ascontiguousarray(array, dtype="<f4").tobytes()
        )
        if len(raw) != array.size * 4:
            raise V2ProtocolError("binary tensor byte count differs")
        tensors.append({"name": tensor["name"], "shape": list(array.shape),
                        "datatype": "FP32", "parameters": {"binary_data_size": len(raw)}})
        chunks.append(raw)
    header[key] = tensors
    prefix = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    body = prefix + b"".join(chunks)
    return body, {"Content-Type": CONTENT_TYPE, HEADER: str(len(prefix)),
                  "Content-Length": str(len(body))}


def decode(body: bytes, header_length: str, key: str) -> dict[str, Any]:
    try:
        size = int(header_length)
    except (ValueError, TypeError) as error:
        raise V2ProtocolError("invalid binary header length") from error
    if not 0 < size <= len(body) <= 1_000_000:
        raise V2ProtocolError("invalid binary body length")
    value = decode_json(body[:size])
    tensors = value.get(key)
    if not isinstance(tensors, list) or not tensors:
        raise V2ProtocolError("binary tensor list is required")
    offset = size
    decoded = []
    for tensor in tensors:
        if not isinstance(tensor, dict) or set(tensor) != {
            "name", "shape", "datatype", "parameters"
        }:
            raise V2ProtocolError("invalid binary tensor fields")
        if not isinstance(tensor["name"], str) or not tensor["name"]:
            raise V2ProtocolError("invalid binary tensor name")
        shape = tensor["shape"]
        if (tensor["datatype"] != "FP32" or not isinstance(shape, list) or not shape
                or any(type(n) is not int or n <= 0 for n in shape)):
            raise V2ProtocolError("invalid binary shape or datatype")
        count = math.prod(shape) * 4
        parameters = tensor["parameters"]
        if (not isinstance(parameters, dict) or set(parameters) != {"binary_data_size"}
                or type(parameters["binary_data_size"]) is not int
                or parameters["binary_data_size"] != count or offset + count > len(body)):
            raise V2ProtocolError("invalid binary tensor length")
        array = np.frombuffer(body, dtype="<f4", count=count // 4, offset=offset).reshape(shape)
        if not np.isfinite(array).all():
            raise V2ProtocolError("binary tensor contains non-finite values")
        decoded.append({"name": tensor["name"], "shape": shape, "datatype": "FP32",
                        "data": array.astype(np.float32, copy=False)})
        offset += count
    if offset != len(body):
        raise V2ProtocolError("trailing binary tensor bytes")
    value[key] = decoded
    return value
