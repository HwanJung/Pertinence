"""Network-independent regression coverage for the browser inference adapter."""

import asyncio
from io import BytesIO
import json

import httpx
import numpy as np
from PIL import Image, ImageOps
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from pertinence.demo.app import (
    EXPERTS, MAX_BODY_BYTES, MAX_FILE_BYTES, Settings, create_app, preprocess_image,
)
from pertinence.serving.trace import normalize_cifar_images
from pertinence.serving.v2 import infer_response, parse_infer_request


def picture(mode="RGB", size=(32, 32), *, format="PNG", exif=None):
    image = Image.new(mode, size, 100)
    output = BytesIO()
    image.save(output, format=format, **({"exif": exif} if exif else {}))
    return output.getvalue()


def router_response(request, route=1):
    request_id, _ = parse_infer_request(json.loads(request.content))
    return infer_response("dispatcher-000733", np.arange(10, dtype=np.float32)[None],
                          request_id=request_id, route=route, parameters={
                              "expert_id": EXPERTS[route], "dispatcher_id": "dispatcher-000733",
                              "serving_mode": "hybrid", "extractor_seconds": .01,
                              "dispatcher_seconds": .001, "expert_seconds": 0 if route == 0 else .02,
                          })


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_preprocessing_matches_canonical_cifar(mode):
    content = picture(mode)
    raw = np.asarray(Image.open(BytesIO(content)).convert("RGB"))[None]
    actual = preprocess_image(content, "image/png")
    np.testing.assert_array_equal(actual, normalize_cifar_images(raw))
    assert actual.shape == (1, 3, 32, 32) and actual.dtype == np.float32


def test_preprocessing_applies_exif_before_resize():
    image = Image.new("RGB", (64, 32), "red")
    image.paste("blue", (32, 0, 64, 32))
    exif = Image.Exif()
    exif[274] = 6
    encoded = BytesIO()
    image.save(encoded, format="JPEG", exif=exif)
    content = encoded.getvalue()
    expected = ImageOps.exif_transpose(Image.open(BytesIO(content))).convert("RGB")
    expected = expected.resize((32, 32), Image.Resampling.BILINEAR)
    np.testing.assert_array_equal(preprocess_image(content, "image/jpeg"),
                                  normalize_cifar_images(np.asarray(expected)[None]))


@pytest.mark.parametrize("content,mime,status", [
    (b"broken", "image/png", 400), (picture(), "text/plain", 415),
    (picture(format="GIF"), "image/png", 415),
    (b"x" * (MAX_FILE_BYTES + 1), "image/png", 413),
    (picture(size=(5000, 4001)), "image/png", 413),
], ids=["corrupt", "mime", "format", "file-size", "pixels"])
def test_preprocessing_rejects_invalid_uploads(content, mime, status):
    with pytest.raises(HTTPException) as error:
        preprocess_image(content, mime)
    assert error.value.status_code == status


@pytest.mark.parametrize("route", [0, 1, 2])
def test_upload_maps_real_v2_contract(route):
    def upstream(request):
        if request.method == "GET":
            return httpx.Response(200, json={})
        assert request.url.path == "/v2/models/dispatcher-000733/infer"
        _, tensor = parse_infer_request(json.loads(request.content))
        np.testing.assert_array_equal(tensor, preprocess_image(picture(), "image/png"))
        return httpx.Response(200, json=router_response(request, route))

    with TestClient(create_app(transport=httpx.MockTransport(upstream))) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/health/ready").status_code == 200
        response = client.post("/api/predict", files={"file": ("a.png", picture(), "image/png")})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["serving_mode"] == "hybrid"
        assert result["prediction"]["class_name"] == "truck"
        assert result["expert_id"] == EXPERTS[route]
        assert result["route"] == route and result["extractor_reused"] == (route == 0)
        assert result["timings"]["extractor_ms"] == 10
        assert result["timings"]["total_ms"] > 0
        assert len(result["top_k"]) == 3
        assert result["logits"] == list(range(10))
        assert 0 < result["top_k"][0]["score"] < 1
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("failure,status", [("timeout", 504), ("http", 502),
                                            ("upstream_timeout", 504), ("identity", 502),
                                            ("route", 502), ("nan", 502), ("json", 502),
                                            ("mode", 502), ("distributed", 502)])
def test_upstream_errors_are_reported_and_retry_is_possible(failure, status):
    calls = []

    def upstream(request):
        calls.append(request)
        if len(calls) > 1:
            return httpx.Response(200, json=router_response(request))
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if failure in {"http", "upstream_timeout"}:
            return httpx.Response(504 if failure == "upstream_timeout" else 500)
        if failure == "json":
            return httpx.Response(200, text="bad json")
        value = router_response(request)
        if failure == "identity":
            value["id"] = "wrong"
        elif failure == "route":
            value["outputs"][1]["data"] = [99]
        elif failure == "nan":
            value["outputs"][0]["data"] = ["NaN"] * 10
        elif failure in {"mode", "distributed"}:
            value["parameters"]["serving_mode"] = "distributed" if failure == "distributed" else "fused"
        return httpx.Response(200, json=value)

    with TestClient(create_app(transport=httpx.MockTransport(upstream))) as client:
        files = {"file": ("a.png", picture(), "image/png")}
        assert client.post("/api/predict", files=files).status_code == status
        assert len(calls) == 1  # No automatic retry.
        assert client.post("/api/predict", files=files).status_code == 200


def test_upload_limits_and_multipart_validation():
    def unexpected(request):
        pytest.fail("invalid uploads must not reach inference")

    with TestClient(create_app(transport=httpx.MockTransport(unexpected))) as client:
        assert client.post("/api/predict", content=b"hello").status_code == 415
        assert client.post("/api/predict", files={"other": ("a.png", picture())}).status_code == 400
        assert client.post("/api/predict", files=[
            ("file", ("a.png", picture())), ("file", ("b.png", picture())),
        ]).status_code == 400
        headers = {"content-type": "multipart/form-data; boundary=x"}
        assert client.post("/api/predict", content=b"x" * (MAX_BODY_BYTES + 1),
                           headers=headers).status_code == 413
        assert client.post("/api/predict", content=iter([b"x" * (MAX_BODY_BYTES + 1)]),
                           headers=headers).status_code == 413
        assert client.post("/api/predict", content=b"hello",
                           headers={"content-type": "multipart/form-data;"}).status_code == 400
        assert client.post("/api/predict", content=b"malformed",
                           headers=headers).status_code == 400


@pytest.mark.parametrize("failed", ["mobilenet", "resnet56", "dispatcher"])
def test_readiness_checks_all_experts_but_liveness_is_independent(failed):
    visited = []

    def upstream(request):
        visited.append(str(request.url))
        return httpx.Response(503 if failed in str(request.url) else 200)

    with TestClient(create_app(transport=httpx.MockTransport(upstream))) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert len(visited) == 3
        assert not any("shufflenet" in url for url in visited)


def test_concurrent_prediction_is_rejected_and_deadline_releases_lock():
    async def scenario():
        entered = asyncio.Event()

        async def upstream(request):
            entered.set()
            await asyncio.sleep(1)
            return httpx.Response(200, json=router_response(request))

        app = create_app(Settings(timeout_seconds=.1), transport=httpx.MockTransport(upstream))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://demo") as client:
                files = {"file": ("a.png", picture(), "image/png")}
                pending = asyncio.create_task(client.post("/api/predict", files=files))
                await asyncio.wait_for(entered.wait(), timeout=2)
                assert (await client.post("/api/predict", files=files)).status_code == 429
                assert (await pending).status_code == 504
                assert (await client.post("/api/predict", files=files)).status_code == 504

    asyncio.run(scenario())
