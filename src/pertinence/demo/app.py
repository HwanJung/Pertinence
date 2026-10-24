"""Memory-only image upload adapter for the existing KServe V2 router."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
import logging
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4
import warnings

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from python_multipart.exceptions import MultipartParseError
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from pertinence.serving.trace import normalize_cifar_images
from pertinence.serving.v2 import V2ProtocolError, parse_infer_response

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_BODY_BYTES = MAX_FILE_BYTES + 64 * 1024
MAX_PIXELS = 20_000_000
EXPERTS = ("shufflenetv2_x0_5", "mobilenetv2_x0_5", "resnet56")
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship",
           "truck")
LABELS = ("비행기", "자동차", "새", "고양이", "사슴", "개", "개구리", "말", "배", "트럭")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    router_url: str = (
        "http://demo-dispatcher-000733-predictor.pertinence-serving.svc.cluster.local"
    )
    model_name: str = "dispatcher-000733"
    expert_urls: tuple[str, ...] = tuple(
        f"http://{name}-predictor.pertinence-serving.svc.cluster.local"
        for name in ("demo-expert-mobilenet", "demo-expert-resnet56")
    )
    timeout_seconds: float = 30.0


class MemoryMultipartParser(MultiPartParser):
    # The entire request is bounded before parsing; never spill an upload to disk.
    spool_max_size = MAX_BODY_BYTES


def preprocess_image(content: bytes, content_type: str) -> np.ndarray:
    """Decode one JPEG/PNG and reproduce the canonical CIFAR normalization."""
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(413, "이미지는 5MiB 이하여야 합니다.")
    if content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(415, "JPEG 또는 PNG 이미지를 선택하세요.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise HTTPException(415, "JPEG 또는 PNG 이미지를 선택하세요.")
                if image.width * image.height > MAX_PIXELS:
                    raise HTTPException(413, "이미지는 2천만 픽셀 이하여야 합니다.")
                image = ImageOps.exif_transpose(image).convert("RGB")
                image = image.resize((32, 32), Image.Resampling.BILINEAR)
                pixels = np.asarray(image, dtype=np.uint8)
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise HTTPException(413, "이미지 해상도가 너무 큽니다.") from error
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as error:
        raise HTTPException(400, "이미지를 읽을 수 없습니다. 다른 파일을 선택하세요.") from error
    return normalize_cifar_images(pixels[None, ...])


def prediction_result(value: dict[str, Any], request_id: str, model_name: str) -> dict[str, Any]:
    """Validate router identity and map its existing outputs into display fields."""
    if value.get("id") != request_id or value.get("model_name") != model_name:
        raise ValueError("router response identity mismatch")
    outputs = value["outputs"]
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise ValueError("router must return logits and route")
    routes = [output for output in outputs if output.get("name") == "route"]
    if len(routes) != 1:
        raise ValueError("missing route")
    route = routes[0]
    if (route.get("datatype") != "INT64" or route.get("shape") != [1]
            or not isinstance(route.get("data"), list) or len(route["data"]) != 1
            or type(route["data"][0]) is not int or not 0 <= route["data"][0] < 3):
        raise ValueError("invalid route")
    index = route["data"][0]
    logits, _ = parse_infer_response({
        **value, "outputs": [item for item in outputs if item.get("name") != "route"],
    })
    params = value["parameters"]
    if params["expert_id"] != EXPERTS[index] or params["dispatcher_id"] != model_name:
        raise ValueError("expert or dispatcher identity mismatch")
    if params["serving_mode"] != "hybrid":
        raise ValueError("demo requires hybrid routing")
    timings = {}
    for field in ("extractor_seconds", "dispatcher_seconds", "expert_seconds"):
        seconds = params[field]
        if type(seconds) not in (int, float) or not np.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid timing")
        timings[field.replace("_seconds", "_ms")] = seconds * 1000
    scores = np.exp(logits[0].astype(np.float64) - float(logits.max()))
    scores /= scores.sum()
    order = np.argsort(-scores, kind="stable")[:3]
    top = [{"class_id": int(i), "class_name": CLASSES[i], "label": LABELS[i],
            "score": float(scores[i])} for i in order]
    return {"request_id": request_id, "model_id": model_name, "prediction": top[0],
            "logits": logits[0].tolist(),
            "top_k": top, "route": index, "expert_id": EXPERTS[index],
            "serving_mode": "hybrid", "extractor_reused": index == 0, "timings": timings}


def create_app(settings: Settings | None = None, *,
               transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """Create a demo app; transport injection keeps unit tests network independent."""
    settings = settings or Settings(router_url=os.environ.get("ROUTER_URL", Settings.router_url))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(
            timeout=settings.timeout_seconds, transport=transport, trust_env=False,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        ) as client:
            app.state.client = client
            yield

    app = FastAPI(title="PERTINENCE 이미지 시연", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    busy = asyncio.Lock()
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    async def index():
        return FileResponse(static / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/health/live")
    async def live():
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready():
        async def check(url):
            try:
                response = await app.state.client.get(url + "/v2/health/ready", timeout=1.0)
                return response.status_code == 200
            except httpx.HTTPError:
                return False

        checks = await asyncio.gather(*(check(url) for url in (
            settings.router_url, *settings.expert_urls,
        )))
        return JSONResponse({"ready": all(checks)}, status_code=200 if all(checks) else 503)

    @app.post("/api/predict")
    async def predict(request: Request):
        if busy.locked():
            raise HTTPException(429, "다른 이미지를 처리하고 있습니다. 잠시 후 다시 시도하세요.")
        async with busy:
            started = time.perf_counter()
            request_id = str(uuid4())
            try:
                async with asyncio.timeout(settings.timeout_seconds):
                    content_type = request.headers.get("content-type", "")
                    if not content_type.lower().startswith("multipart/form-data;"):
                        raise HTTPException(415, "multipart 이미지 업로드가 필요합니다.")
                    length = request.headers.get("content-length")
                    if length is not None:
                        if not length.isdecimal():
                            raise HTTPException(400, "잘못된 요청 크기입니다.")
                        if len(length) > 10 or int(length) > MAX_BODY_BYTES:
                            raise HTTPException(413, "이미지는 5MiB 이하여야 합니다.")
                    body = bytearray()
                    async for chunk in request.stream():
                        if len(body) + len(chunk) > MAX_BODY_BYTES:
                            raise HTTPException(413, "이미지는 5MiB 이하여야 합니다.")
                        body.extend(chunk)

                    async def stream():
                        yield bytes(body)

                    parser = MemoryMultipartParser(request.headers, stream(),
                                                   max_files=1, max_fields=0)
                    form = await parser.parse()
                    try:
                        upload = form.get("file")
                        if not isinstance(upload, UploadFile) or len(form.multi_items()) != 1:
                            raise HTTPException(400, "file 필드에 이미지 한 장을 업로드하세요.")
                        image = await asyncio.to_thread(
                            preprocess_image, await upload.read(), upload.content_type or "",
                        )
                    finally:
                        await form.close()
                    payload = {"id": request_id, "inputs": [{
                        "name": "image", "shape": [1, 3, 32, 32], "datatype": "FP32",
                        "data": image.reshape(-1).tolist(),
                    }], "outputs": [{"name": "logits"}, {"name": "route"}]}
                    response = await app.state.client.post(
                        settings.router_url + f"/v2/models/{settings.model_name}/infer",
                        json=payload, headers={"X-Request-Id": request_id},
                    )
                    if response.status_code == 504:
                        raise HTTPException(504, "모델 응답 시간이 초과됐습니다. 다시 시도하세요.")
                    response.raise_for_status()
                    try:
                        result = prediction_result(response.json(), request_id, settings.model_name)
                    except (ValueError, KeyError, TypeError, AttributeError, V2ProtocolError) as error:
                        raise HTTPException(502, "모델 응답을 해석할 수 없습니다.") from error
                    result["timings"]["total_ms"] = (time.perf_counter() - started) * 1000
                    LOGGER.info("prediction request_id=%s expert=%s total_ms=%.3f", request_id,
                                result["expert_id"], result["timings"]["total_ms"])
                    return JSONResponse(result, headers={"Cache-Control": "no-store"})
            except (MultiPartException, MultipartParseError) as error:
                raise HTTPException(400, "이미지 한 장을 multipart 형식으로 업로드하세요.") from error
            except (TimeoutError, httpx.TimeoutException) as error:
                raise HTTPException(504, "모델 응답 시간이 초과됐습니다. 다시 시도하세요.") from error
            except httpx.HTTPError as error:
                raise HTTPException(502, "추론 서비스에 연결할 수 없습니다.") from error

    return app
