#!/usr/bin/env python3
"""Validate the exact model identity contract in one KServe V2 response."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EXPERTS = ("shufflenetv2_x0_5", "mobilenetv2_x0_5", "resnet56")


def _validated_outputs(value: dict[str, object], *, router: bool) -> dict[str, dict[str, object]]:
    outputs = value.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("response outputs must be a non-empty list")
    by_name: dict[str, dict[str, object]] = {}
    for output in outputs:
        if not isinstance(output, dict) or not isinstance(output.get("name"), str):
            raise ValueError("response output entries must be named objects")
        name = output["name"]
        if name in by_name:
            raise ValueError("response output names must be unique")
        by_name[name] = output
    expected_names = {"logits", "route"} if router else {"logits"}
    if set(by_name) != expected_names:
        raise ValueError(f"response outputs must be exactly {sorted(expected_names)}")
    logits = by_name["logits"]
    logits_data = logits.get("data")
    if (
        logits.get("datatype") != "FP32"
        or logits.get("shape") != [1, 10]
        or not isinstance(logits_data, list)
        or len(logits_data) != 10
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in logits_data
        )
    ):
        raise ValueError("response logits tensor is invalid")
    return by_name


def validate_identity(value: object, expected: str) -> None:
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    if value.get("model_version") != "1":
        raise ValueError("response model_version must be exactly 1")
    parameters = value.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("response parameters must be an object")

    if expected in EXPERTS:
        if value.get("model_name") != expected:
            raise ValueError("expert response model_name mismatch")
        _validated_outputs(value, router=False)
        unexpected = {"promotion_id", "dispatcher_id", "expert_id"} & set(parameters)
        if unexpected:
            raise ValueError("direct expert response contains router identity fields")
        return

    outputs = _validated_outputs(value, router=True)
    route_output = outputs["route"]
    if route_output.get("datatype") != "INT64":
        raise ValueError("router response must contain one INT64 route")
    route_data = route_output.get("data")
    if (
        route_output.get("shape") != [1]
        or not isinstance(route_data, list)
        or len(route_data) != 1
        or type(route_data[0]) is not int
        or route_data[0] not in range(len(EXPERTS))
    ):
        raise ValueError("router route tensor is invalid")
    promotion_id = parameters.get("promotion_id")
    if not isinstance(promotion_id, str) or not promotion_id:
        raise ValueError("router promotion_id is missing")
    route = route_data[0]

    if expected == "static":
        identity = (
            value.get("model_name"),
            parameters.get("dispatcher_id"),
            parameters.get("expert_id"),
            route,
        )
        if identity != ("static-resnet-router", "static", "resnet56", 2):
            raise ValueError("static router identity mismatch")
        return
    if expected == "dispatcher-000733":
        if value.get("model_name") != expected or parameters.get("dispatcher_id") != expected:
            raise ValueError("dispatcher router identity mismatch")
        if parameters.get("expert_id") != EXPERTS[route]:
            raise ValueError("dispatcher expert identity disagrees with route")
        return
    raise ValueError(f"unsupported expected identity: {expected}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("response", type=Path)
    parser.add_argument("expected")
    args = parser.parse_args()
    try:
        value = json.loads(args.response.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    try:
        validate_identity(value, args.expected)
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
