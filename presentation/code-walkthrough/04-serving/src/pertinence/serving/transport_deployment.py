"""Render isolated suite workloads from the established distributed KServe template."""

from __future__ import annotations

import copy
from typing import Any

from .contracts import PromotionManifestV1, _pinned_image
from .transport_suite import VARIANTS, validate_manifest


def render_services(base: list[dict], manifest: dict, promotion: dict, *,
                    fused_modelcar: str, hybrid_modelcar: str) -> list[dict[str, Any]]:
    validate_manifest(manifest)
    PromotionManifestV1.from_dict(promotion)
    if promotion["fingerprint"] != manifest["promotion_fingerprint"]:
        raise ValueError("deployment promotion differs from run identity")
    for image in (fused_modelcar, hybrid_modelcar):
        _pinned_image(image, "modelcar")
    variant = VARIANTS[manifest["arm"]]
    services = {item["metadata"]["name"]: copy.deepcopy(item) for item in base
                if item.get("kind") == "InferenceService"}
    expected = {"router-dispatcher-000733", "expert-shufflenet", "expert-mobilenet", "expert-resnet56"}
    if services.keys() != expected:
        raise ValueError("base must contain exactly the four distributed InferenceServices")
    router = services["router-dispatcher-000733"]
    predictor = router["spec"]["predictor"]
    container = predictor["containers"][0]
    args = container["args"]
    if variant.structure != "distributed":
        urls = []
        if variant.structure == "hybrid":
            for index, arg in enumerate(args):
                if arg == "--expert-url" and not args[index + 1].startswith("shufflenet"):
                    urls.extend([arg, args[index + 1]])
        container["args"] = ["serve", "router", "--promotion", "/etc/pertinence/promotion.json",
                             "--bundle-path", "/mnt/models", "--router-mode",
                             f"{variant.structure}-dynamic", *urls]
        cpu = "4" if variant.structure == "fused" else "2"
        container["resources"] = {key: {"cpu": cpu, "memory": f"{cpu}Gi"}
                                  for key in ("requests", "limits")}
        predictor["storageUris"] = [{"uri": "oci://" + (
            fused_modelcar if variant.structure == "fused" else hybrid_modelcar
        ), "mountPath": "/mnt/models"}]
        services.pop("expert-shufflenet")
        if variant.structure == "fused":
            services = {"router-dispatcher-000733": router}
    if variant.transport != "local":
        container["args"].extend(["--expert-transport", variant.transport])
    for name, service in services.items():
        service["metadata"]["annotations"].update({
            "pertinence.dev/arm-id": manifest["arm"],
            "pertinence.dev/run-fingerprint": manifest["fingerprint"],
            "pertinence.dev/promotion-fingerprint": promotion["fingerprint"],
            "pertinence.dev/serving-mode": variant.structure,
            "pertinence.dev/runtime-image": manifest["runtime_image"],
            "pertinence.dev/expert-transport": variant.transport,
            "pertinence.dev/connection-mode": variant.connection,
            "pertinence.dev/diagnostics": str(manifest["diagnostics"]).lower(),
        })
        predictor = service["spec"]["predictor"]
        predictor["minReplicas"] = predictor["maxReplicas"] = 1
        container = predictor["containers"][0]
        container["image"] = manifest["runtime_image"]
        container["args"].extend(["--run-fingerprint", manifest["fingerprint"]])
        if name.startswith("expert-") and variant.connection == "keep-alive":
            container["args"].extend(["--keep-alive", "--idle-timeout", "10"])
        if manifest["diagnostics"]:
            container["args"].extend(["--diagnostics-path", "/diagnostics/spans.jsonl"])
            container.setdefault("volumeMounts", []).append(
                {"name": "diagnostics", "mountPath": "/diagnostics"})
            predictor.setdefault("volumes", []).append({"name": "diagnostics", "emptyDir": {}})
        for volume in predictor.get("volumes", []):
            if volume.get("name") == "promotion":
                volume["configMap"]["name"] = "transport-promotion-" + manifest["fingerprint"][:16]
    return list(services.values())
