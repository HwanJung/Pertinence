#!/usr/bin/env python3
"""Verify uploaded PNGs and direct V2 requests yield identical logits and routes."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import time
from uuid import uuid4

import httpx
import numpy as np
from PIL import Image

from pertinence.demo.app import EXPERTS, preprocess_image
from pertinence.serving.trace import NORMALIZE_MEAN, NORMALIZE_STD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo-url", default="http://127.0.0.1:8088")
    parser.add_argument("--router-url", default="http://127.0.0.1:18080")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--idle-seconds", type=float, default=12.0)
    parser.add_argument("--expected-mode", choices=("hybrid", "distributed"), default="hybrid")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    if baseline is not None and (not baseline["valid"] or len(baseline["records"]) != args.samples):
        parser.error("baseline must contain the same number of verified samples")
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if not output.is_relative_to(root / "artifacts" / "cifar10"):
        parser.error("output must be under artifacts/cifar10/")
    if args.samples < 1:
        parser.error("samples must be positive")
    if not 0 <= args.idle_seconds <= 60:
        parser.error("idle-seconds must be between 0 and 60")
    output.mkdir(parents=True, exist_ok=False)
    with np.load(args.trace, allow_pickle=False) as trace:
        images = np.array(trace["images"][:args.samples])
    if len(images) == 0:
        parser.error("trace must contain at least one image")
    records = []
    with httpx.Client(timeout=35, trust_env=False) as client:
        client.get(args.demo_url + "/health/ready").raise_for_status()
        for index, normalized in enumerate(images):
            pixels = np.rint((normalized * NORMALIZE_STD + NORMALIZE_MEAN) * 255)
            pixels = np.clip(pixels, 0, 255).astype(np.uint8).transpose(1, 2, 0)
            encoded = BytesIO()
            Image.fromarray(pixels).save(encoded, format="PNG")
            content = encoded.getvalue()
            (output / f"sample-{index}.png").write_bytes(content)
            tensor = preprocess_image(content, "image/png")
            np.testing.assert_array_equal(tensor[0], normalized)
            response = client.post(args.demo_url + "/api/predict",
                                   files={"file": ("sample.png", content, "image/png")})
            response.raise_for_status()
            uploaded = response.json()
            direct = client.post(args.router_url + "/v2/models/dispatcher-000733/infer", json={
                "id": str(uuid4()), "inputs": [{"name": "image", "shape": [1, 3, 32, 32],
                    "datatype": "FP32", "data": tensor.reshape(-1).tolist()}],
                "outputs": [{"name": "logits"}, {"name": "route"}],
            })
            direct.raise_for_status()
            value = direct.json()
            outputs = {item["name"]: item["data"] for item in value["outputs"]}
            assert value["parameters"]["serving_mode"] == args.expected_mode
            if args.expected_mode == "hybrid":
                assert uploaded["serving_mode"] == "hybrid"
            np.testing.assert_array_equal(uploaded["logits"], outputs["logits"])
            if baseline is not None:
                previous = baseline["records"][index]
                np.testing.assert_array_equal(
                    preprocess_image((args.baseline.parent / f"sample-{index}.png").read_bytes(),
                                     "image/png"), tensor)
                np.testing.assert_allclose(uploaded["logits"], previous["upload"]["logits"],
                                           rtol=1e-5, atol=1e-6)
                assert uploaded["route"] == previous["upload"]["route"]
            assert uploaded["route"] == outputs["route"][0]
            assert uploaded["expert_id"] == EXPERTS[uploaded["route"]]
            records.append({"sample": index, "upload": uploaded, "direct": value})
        invalid = client.post(args.demo_url + "/api/predict",
                              files={"file": ("broken.png", b"broken", "image/png")})
        assert invalid.status_code == 400
        recovered = client.post(args.demo_url + "/api/predict",
                                files={"file": ("sample.png", content, "image/png")})
        recovered.raise_for_status()
        time.sleep(args.idle_seconds)
        after_idle = client.post(args.demo_url + "/api/predict",
                                 files={"file": ("sample.png", content, "image/png")})
        after_idle.raise_for_status()
        np.testing.assert_array_equal(after_idle.json()["logits"], records[-1]["upload"]["logits"])
        assert after_idle.json()["route"] == records[-1]["upload"]["route"]
    result = {"valid": True, "samples": len(records), "logits_exact": True,
              "route_exact": True, "serving_mode": args.expected_mode,
              "baseline_parity": baseline is not None, "rtol": 1e-5, "atol": 1e-6,
              "invalid_upload_recovery": True,
              "idle_seconds": args.idle_seconds, "idle_recovery": True, "records": records}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {len(records)} uploads: exact logits/route parity, error and idle recovery.")


if __name__ == "__main__":
    main()
