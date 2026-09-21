from __future__ import annotations

import pytest

from cloud.kserve.scripts.assert_identity import validate_identity


def _router(model: str, dispatcher: str, expert: str, route: int) -> dict[str, object]:
    return {
        "model_name": model,
        "model_version": "1",
        "parameters": {
            "promotion_id": "promotion-1",
            "dispatcher_id": dispatcher,
            "expert_id": expert,
        },
        "outputs": [
            {"name": "logits", "datatype": "FP32", "shape": [1, 10], "data": [0.0] * 10},
            {"name": "route", "datatype": "INT64", "shape": [1], "data": [route]},
        ],
    }


def _expert(model: str) -> dict[str, object]:
    return {
        "model_name": model,
        "model_version": "1",
        "parameters": {"process_cpu_seconds": 0.1},
        "outputs": [
            {"name": "logits", "datatype": "FP32", "shape": [1, 10], "data": [0.0] * 10}
        ],
    }


def test_identity_validator_accepts_exact_direct_and_router_contracts() -> None:
    validate_identity(_expert("resnet56"), "resnet56")
    validate_identity(
        _router("static-resnet-router", "static", "resnet56", 2),
        "static",
    )
    validate_identity(
        _router("dispatcher-000733", "dispatcher-000733", "mobilenetv2_x0_5", 1),
        "dispatcher-000733",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_name", "static-resnet-router"),
        ("dispatcher_id", "static"),
        ("expert_id", "resnet56"),
        ("route", 2),
    ),
)
def test_dispatcher_identity_rejects_mixed_response(field: str, value: object) -> None:
    response = _router(
        "dispatcher-000733", "dispatcher-000733", "mobilenetv2_x0_5", 1
    )
    if field == "model_name":
        response[field] = value
    elif field == "route":
        response["outputs"][1]["data"] = [value]  # type: ignore[index]
    else:
        response["parameters"][field] = value  # type: ignore[index]

    with pytest.raises(ValueError):
        validate_identity(response, "dispatcher-000733")


@pytest.mark.parametrize(
    "mutation",
    (
        lambda response: response.pop("outputs"),
        lambda response: response["outputs"].append(response["outputs"][0]),
        lambda response: response["outputs"][0].update({"shape": [10]}),
        lambda response: response["outputs"][0]["data"].__setitem__(0, float("nan")),
    ),
)
def test_identity_validator_rejects_malformed_logits(mutation) -> None:
    response = _router("static-resnet-router", "static", "resnet56", 2)
    mutation(response)
    with pytest.raises(ValueError):
        validate_identity(response, "static")


def test_identity_validator_rejects_router_fields_on_direct_expert() -> None:
    response = _expert("resnet56")
    response["parameters"]["dispatcher_id"] = "static"  # type: ignore[index]
    with pytest.raises(ValueError, match="router identity"):
        validate_identity(response, "resnet56")
