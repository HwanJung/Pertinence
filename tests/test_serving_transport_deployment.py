import copy

import pytest

from pertinence.serving.transport_deployment import render_services
from pertinence.serving.transport_suite import run_manifest
from test_serving_contracts import promotion


def base():
    services = []
    for name in ("router-dispatcher-000733", "expert-shufflenet", "expert-mobilenet", "expert-resnet56"):
        services.append({"kind": "InferenceService", "metadata": {"name": name, "annotations": {}},
                         "spec": {"predictor": {"containers": [{"args": [
                             "serve", "router" if name.startswith("router") else "expert",
                             "--expert-url", "shufflenet=one", "--expert-url", "mobile=two",
                             "--expert-url", "resnet=three"], "resources": {
                                 key: {"cpu": "1", "memory": "1Gi"}
                                 for key in ("requests", "limits")}}], "volumes": [
                                     {"name": "promotion", "configMap": {"name": "old"}}]}}})
    return services


@pytest.mark.parametrize("arm,count", [("D0", 4), ("D1", 4), ("D2", 4), ("D3", 4), ("H1", 3), ("F0", 1)])
def test_deployment_resource_budget_and_flags(arm, count):
    promoted = promotion()
    manifest = run_manifest(run_id="run", arm=arm, block=0, diagnostics=True,
                            promotion_fingerprint=promoted["fingerprint"], trace_fingerprint="b" * 64,
                            runtime_image="runtime@sha256:" + "c" * 64,
                            serving_node_uid="s", loadgen_node_uid="l")
    original = base()
    unchanged = copy.deepcopy(original)
    services = render_services(original, manifest, promoted,
                               fused_modelcar="fused@sha256:" + "d" * 64,
                               hybrid_modelcar="hybrid@sha256:" + "e" * 64)
    assert original == unchanged
    assert len(services) == count
    assert sum(int(service["spec"]["predictor"]["containers"][0]["resources"]["limits"]["cpu"])
               for service in services) == 4
    for service in services:
        predictor = service["spec"]["predictor"]
        container = predictor["containers"][0]
        assert manifest["fingerprint"] in container["args"]
        assert "--diagnostics-path" in container["args"]
        if service["metadata"]["name"].startswith("expert"):
            assert ("--keep-alive" in container["args"]) == (arm in {"D2", "D3", "H1"})
    if arm == "H1":
        router = services[0]["spec"]["predictor"]
        assert "hybrid-dynamic" in router["containers"][0]["args"]
        assert "shufflenet=one" not in router["containers"][0]["args"]
        assert router["storageUris"][0]["uri"].startswith("oci://hybrid@sha256:")
