#!/usr/bin/env python3
"""Verify fixed-artifact parity across all transports locally; no performance claims."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import threading

import numpy as np

from pertinence.serving.cli.benchmark import _write_json_exclusive
from pertinence.serving.contracts import PromotionManifestV1
from pertinence.serving.http import ExpertModel, V2Application, make_server
from pertinence.serving.promotion import validate_promotion_bundle
from pertinence.serving.runtime import (
    DynamicRouter, FusedDynamicRouter, HttpExpertClient, HybridDynamicRouter,
    OnnxExpertRunner, load_dispatcher_npz,
)
from pertinence.serving.trace import verify_trace
from pertinence.serving.transport_suite import VARIANTS


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("bundle", "trace", "promotion", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    promotion = PromotionManifestV1.from_text(args.promotion.read_text()).to_dict()
    verified = validate_promotion_bundle(promotion, args.bundle / "evaluated")
    dispatcher = load_dispatcher_npz(args.bundle / "evaluated" / verified["checkpoint"])
    trace, images, _, _ = verify_trace(args.trace)
    names = [item["id"] for item in promotion["experts"]]
    runners = {item["id"]: OnnxExpertRunner(
        args.bundle / "experts" / item["id"] / "model.onnx", item["model_sha256"],
        features_name="features" if index == 0 else None,
        feature_dim=1024 if index == 0 else None,
    ) for index, item in enumerate(promotion["experts"])}
    positional = ("dispatcher-000733", promotion["promotion_id"], "dispatcher-000733", names, names[0])
    fused = FusedDynamicRouter(*positional, runners, dispatcher, feature_dim=1024)
    reference = [fused.predict(image[None]) for image in images]
    summary = {}
    for arm, variant in VARIANTS.items():
        with ExitStack() as stack:
            clients = {}
            for index, name in enumerate(names):
                if variant.structure == "fused" or (variant.structure == "hybrid" and index == 0):
                    clients[name] = runners[name]
                    continue
                server = make_server(V2Application(ExpertModel(
                    name, runners[name], feature_dim=1024 if index == 0 else None,
                )), "127.0.0.1", 0, keep_alive=variant.connection == "keep-alive")
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                stack.callback(thread.join)
                stack.callback(server.server_close)
                stack.callback(server.shutdown)
                client = HttpExpertClient(f"http://127.0.0.1:{server.server_port}", name,
                                          timeout_seconds=10, transport=variant.transport,
                                          feature_dim=1024 if index == 0 else None)
                stack.callback(client.close)
                clients[name] = client
            router_type = {"fused": FusedDynamicRouter, "hybrid": HybridDynamicRouter,
                           "distributed": DynamicRouter}[variant.structure]
            router = router_type(*positional, clients, dispatcher, feature_dim=1024)
            mismatches, error, exact = 0, 0.0, True
            for index, (image, expected) in enumerate(zip(images, reference)):
                result = router.predict(image[None], f"{arm}-{index}")
                mismatches += int(result.route != expected.route or
                                  result.logits.argmax() != expected.logits.argmax())
                error = max(error, float(np.max(np.abs(result.logits - expected.logits))))
                exact &= result.logits.tobytes() == expected.logits.tobytes()
            summary[arm] = {"mismatches": mismatches, "max_absolute_error": error,
                            "byte_exact": exact, "samples": len(images)}
            print(json.dumps({arm: summary[arm]}), flush=True)
    _write_json_exclusive(args.output, {"trace_fingerprint": trace["fingerprint"],
                          "promotion_fingerprint": promotion["fingerprint"], "arms": summary,
                          "scope": "local parity only; not cluster performance"})
    if any(row["mismatches"] or row["max_absolute_error"] > 1e-6 for row in summary.values()):
        raise SystemExit("parity failed")


if __name__ == "__main__":
    main()
