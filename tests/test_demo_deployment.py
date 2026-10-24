"""Demo rendering and ownership-safe restoration tests without cloud access."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from pertinence.serving.contracts import APPROVED_DISPATCHER_SHA256, seal_manifest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("demo_manage", ROOT / "cloud/demo/manage.py")
manage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manage)
IMAGE = "registry/demo@sha256:" + "a" * 64


@pytest.fixture
def inventory():
    promoted = [{"id": name, "model_sha256": str(i) * 64,
                 "modelcar_image": f"registry/{name}@sha256:" + str(i) * 64}
                for i, name in enumerate(manage.EXPERT_IDS, 1)]
    promotion = seal_manifest({
        "schema_version": 1, "promotion_id": "demo-existing-promotion",
        "evaluated_bundle": {"uri": "gs://runs/manifest.json", "generation": 123,
                             "fingerprint": "a" * 64},
        "lineage": {"contract_fingerprint": "a" * 64, "search_fingerprint": "b" * 64,
                    "final_fingerprint": "c" * 64},
        "dispatcher": {"id": "dispatcher-000733", "checkpoint_sha256": APPROVED_DISPATCHER_SHA256,
                       "feature_dim": 1024, "expert_count": 3, "modelcar_image": IMAGE},
        "experts": promoted, "runtime_image": IMAGE, "approved_by": "test@example.com",
        "approved_at": "2026-10-18T00:00:00Z",
    })
    experts = [{"metadata": {"name": name, "uid": name, "annotations": {manage.STOP: "true"}},
                "spec": {"predictor": {"storageUris": [{"mountPath": "/mnt/models",
                    "uri": "oci://" + entry["modelcar_image"]}], "containers": [{"image": IMAGE,
                    "args": ["serve", "expert", "--model-name", entry["id"],
                             "--model-sha256", entry["model_sha256"]]}]}}}
               for name, entry in zip(manage.EXPERT_NAMES, promoted, strict=True)]
    router = {"spec": {"predictor": {"containers": [{"image": IMAGE}], "volumes": []}}}
    return promotion, experts, router


def test_render_preserves_sources_and_builds_hybrid_router(inventory):
    promotion, experts, router = inventory
    before = deepcopy(inventory)
    resources = manage.demo_resources(IMAGE, router, experts, promotion, "testsession")
    assert inventory == before
    assert {item["kind"] for item in resources} == {
        "ServiceAccount", "Deployment", "Service", "ConfigMap", "InferenceService",
    }
    assert all(item["metadata"]["labels"][manage.OWNER] == "testsession" for item in resources)
    service = next(item for item in resources if item["kind"] == "Service")
    assert service["spec"]["type"] == "ClusterIP"
    deployment = next(item for item in resources if item["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    assert not pod["automountServiceAccountToken"]
    assert pod["containers"][0]["securityContext"]["readOnlyRootFilesystem"]
    predictor = resources[-1]["spec"]["predictor"]
    args = predictor["containers"][0]["args"]
    assert args[args.index("--router-mode") + 1] == "hybrid-dynamic"
    assert args.count("--expert-url") == 2
    assert not any("demo-expert-shufflenet-predictor" in arg for arg in args)
    assert args[args.index("--expert-transport") + 1] == "binary"
    assert predictor["containers"][0]["resources"]["requests"] == {"cpu": "2", "memory": "2Gi"}
    assert predictor["storageUris"][0]["uri"] == "oci://" + manage.HYBRID_MODELCAR
    services = [item for item in resources if item["kind"] == "InferenceService"]
    assert len(services) == 3
    for service in services:
        assert service["metadata"]["name"].startswith("demo-")
        assert "--keep-alive" not in service["spec"]["predictor"]["containers"][0]["args"]
        if service["metadata"]["name"] != "demo-dispatcher-000733":
            assert service["spec"]["predictor"]["containers"][0]["resources"] == {
                "requests": {"cpu": "1", "memory": "1Gi"},
                "limits": {"cpu": "1", "memory": "1Gi"},
            }


@pytest.mark.parametrize("image", ["registry/demo:latest", "registry/demo@sha256:" + "0" * 64])
def test_render_rejects_unpinned_demo_image(inventory, image):
    promotion, experts, router = inventory
    with pytest.raises(ValueError, match="digest"):
        manage.demo_resources(image, router, experts, promotion, "testsession")


def test_render_rejects_expert_lineage_mismatch(inventory):
    promotion, experts, router = inventory
    experts[0]["spec"]["predictor"]["containers"][0]["args"][-1] = "d" * 64
    with pytest.raises(ValueError, match="sealed promotion"):
        manage.demo_resources(IMAGE, router, experts, promotion, "testsession")


@pytest.mark.parametrize("annotation", ["true", "false", None])
def test_stop_restores_original_annotations_and_is_idempotent(tmp_path, monkeypatch,
                                                            inventory, annotation):
    _, experts, _ = inventory
    experts = deepcopy(experts[:1])
    if annotation is None:
        experts[0]["metadata"]["annotations"] = {}
    else:
        experts[0]["metadata"]["annotations"][manage.STOP] = annotation
    current = deepcopy(experts[0])
    current["metadata"]["annotations"][manage.STOP] = "false"
    state = {"context": manage.CONTEXT, "session": "s1", "resources": [],
             "experts": experts, "pool": {"size": 1}}
    manage.save(tmp_path / "state.json", state)
    calls = []
    monkeypatch.setattr(manage, "get", lambda *args: current)
    monkeypatch.setattr(manage, "kube", lambda *args, **kwargs: calls.append(args) or "{}")
    monkeypatch.setattr(manage, "pool_state", lambda: {"size": 1})
    manage.stop(tmp_path)
    patches = [call for call in calls if call[0] == "annotate"]
    if annotation == "false":
        assert not patches
    else:
        expected = f"{manage.STOP}={annotation}" if annotation else f"{manage.STOP}-"
        assert patches[0][3] == expected
    assert json.loads((tmp_path / "state.json").read_text())["stopped"]
    count = len(calls)
    manage.stop(tmp_path)
    assert len(calls) == count


def test_stop_refuses_unowned_resources(tmp_path, monkeypatch):
    manage.save(tmp_path / "state.json", {"context": manage.CONTEXT, "session": "mine",
        "resources": [{"kind": "Service", "metadata": {"name": "pertinence-demo"}}]})
    monkeypatch.setattr(manage, "get", lambda *args: {"metadata": {"labels": {manage.OWNER: "other"}}})
    with pytest.raises(ValueError, match="unowned"):
        manage.stop(tmp_path)


def test_stop_refuses_expert_changed_by_another_session(tmp_path, monkeypatch, inventory):
    _, experts, _ = inventory
    manage.save(tmp_path / "state.json", {"context": manage.CONTEXT, "session": "mine",
                                         "resources": [], "experts": experts})
    current = deepcopy(experts[0])
    current["spec"]["changed"] = True
    monkeypatch.setattr(manage, "get", lambda *args: current)
    with pytest.raises(ValueError, match="changed outside"):
        manage.stop(tmp_path)


@pytest.mark.parametrize("system_agent", [True, False])
def test_stop_scales_down_only_when_node_has_no_user_workload(tmp_path, monkeypatch, system_agent):
    manage.save(tmp_path / "state.json", {"context": manage.CONTEXT, "session": "mine",
        "resources": [], "experts": [], "pool": {"size": 0}, "resize_requested": True})
    pod = {"metadata": {"namespace": "kube-system" if system_agent else "workload",
                        "ownerReferences": [{"kind": "Node" if system_agent else "ReplicaSet"}],
                        "annotations": {"kubernetes.io/config.mirror": "hash"} if system_agent else {}},
           "spec": {"nodeName": "serving"}, "status": {"phase": "Running"}}
    monkeypatch.setattr(manage, "kube", lambda *args: json.dumps({
        "items": [{"metadata": {"name": "serving"}}],
    }))
    monkeypatch.setattr(manage, "run", lambda *args: json.dumps({"items": [pod]}))
    monkeypatch.setattr(manage, "pool_state", lambda: {"size": 0})
    sizes = []
    monkeypatch.setattr(manage, "resize", lambda size: sizes.append(size))
    if system_agent:
        manage.stop(tmp_path)
        assert sizes == [0]
    else:
        with pytest.raises(ValueError, match="workload Pods"):
            manage.stop(tmp_path)
        assert not sizes


def test_start_creates_only_demo_objects_and_never_reconfigures_existing_experts(
    tmp_path, monkeypatch, inventory,
):
    promotion, experts, router = inventory
    router["spec"]["predictor"]["volumes"] = [{"name": "promotion",
                                               "configMap": {"name": "existing-promotion"}}]
    sources = {("inferenceservice", item["metadata"]["name"]): item for item in experts}
    sources[("inferenceservice", "router-dispatcher-000733")] = router
    sources[("configmap", "existing-promotion")] = {
        "data": {"promotion.json": json.dumps(promotion)},
    }
    before = deepcopy(sources)
    calls = []
    monkeypatch.setattr(manage, "ROOT", tmp_path)
    monkeypatch.setattr(manage, "get", lambda kind, name: sources.get((kind, name)))
    monkeypatch.setattr(manage, "pool_state", lambda: {"size": 1})

    def kube(*args, **kwargs):
        calls.append((args, kwargs))
        return json.dumps({"items": []})

    monkeypatch.setattr(manage, "kube", kube)
    directory = tmp_path / "artifacts/cifar10/demo/session"
    manage.start(directory, IMAGE)
    assert sources == before
    assert not any(args[0] in {"annotate", "patch", "apply", "delete"} for args, _ in calls)
    created = [json.loads(kwargs["input_text"]) for args, kwargs in calls if args[0] == "create"]
    assert len(created) == 7
    assert all(item["metadata"]["name"].startswith(("demo-", "pertinence-demo")) for item in created)
    state = json.loads((directory / "state.json").read_text())
    assert state["ready"] and state["activated_experts"] == []


@pytest.mark.parametrize("fail_verification", [False, True])
def test_upgrade_verifies_before_cleanup_and_restores_on_failure(
    tmp_path, monkeypatch, inventory, fail_verification,
):
    promotion, experts, router = inventory
    session = "session"
    old = manage.demo_resources(IMAGE, router, experts, promotion, session)
    shuffle = deepcopy(next(item for item in old if item["metadata"]["name"] == "demo-expert-mobilenet"))
    shuffle["metadata"]["name"] = "demo-expert-shufflenet"
    old.append(shuffle)
    old_router = next(item for item in old if item["metadata"]["name"] == "demo-dispatcher-000733")
    old_router["spec"]["predictor"]["containers"][0]["args"] = ["serve", "router", "--router-mode", "dynamic"]
    manage.save(tmp_path / "state.json", {
        "context": manage.CONTEXT, "session": session, "ready": True,
        "resources": old, "experts": experts, "router_before": router,
        "pool": {"size": 0}, "resize_requested": True, "activated_experts": [],
    })
    live = {(item["kind"], item["metadata"]["name"]): deepcopy(item) for item in old}
    calls = []
    monkeypatch.setattr(manage, "get", lambda kind, name: live.get((kind, name)))
    monkeypatch.setattr(manage, "pool_state", lambda: {"size": 1})
    monkeypatch.setattr(manage, "wait_ready", lambda: None)

    def replace(item, owner):
        assert (tmp_path / "upgrade-hybrid/state-before.json").exists()
        assert owner == session
        calls.append(("replace", item["metadata"]["name"]))
        live[(item["kind"], item["metadata"]["name"])] = deepcopy(item)

    def verify(directory, trace, mode, baseline=None):
        calls.append(("verify", mode))
        assert ("InferenceService", "demo-expert-shufflenet") in live
        if mode == "hybrid":
            assert baseline == tmp_path / "upgrade-hybrid/distributed/verification.json"
            if fail_verification:
                raise RuntimeError("parity failed")

    def kube(*args, **kwargs):
        calls.append(args)
        assert ("verify", "hybrid") in calls
        live.pop((args[1], args[2]))
        return ""

    monkeypatch.setattr(manage, "replace_workload", replace)
    monkeypatch.setattr(manage, "verify_session", verify)
    monkeypatch.setattr(manage, "kube", kube)
    if fail_verification:
        with pytest.raises(RuntimeError, match="parity failed"):
            manage.upgrade_hybrid(tmp_path, "registry/new@sha256:" + "b" * 64, tmp_path / "trace.npz")
        assert live == {(item["kind"], item["metadata"]["name"]): item for item in old}
    else:
        manage.upgrade_hybrid(tmp_path, "registry/new@sha256:" + "b" * 64, tmp_path / "trace.npz")
        assert ("InferenceService", "demo-expert-shufflenet") not in live
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["upgrade"]["status"] == ("rolled-back" if fail_verification else "complete")
    assert state["pool"] == {"size": 0} and state["resize_requested"]
    assert len(state["resources"]) == (8 if fail_verification else 7)


def test_upgrade_refuses_objects_with_changed_owner(tmp_path, monkeypatch, inventory):
    promotion, experts, router = inventory
    manage.save(tmp_path / "state.json", {
        "context": manage.CONTEXT, "session": "mine", "ready": True,
        "resources": manage.demo_resources(IMAGE, router, experts, promotion, "mine"),
    })
    monkeypatch.setattr(manage, "get", lambda *args: {"metadata": {"labels": {manage.OWNER: "other"}}})
    with pytest.raises(ValueError, match="unowned"):
        manage.upgrade_hybrid(tmp_path, IMAGE, tmp_path / "trace.npz")


@pytest.mark.parametrize("status", ["applying", "rollback-required", "verified-cleanup-pending"])
def test_stop_cleans_owned_resources_during_interrupted_upgrade(tmp_path, monkeypatch, status):
    resources = [{"kind": kind, "metadata": {"name": name, "labels": {manage.OWNER: "s1"}}}
                 for kind, name in [("InferenceService", "demo-expert-shufflenet"),
                                    ("InferenceService", "demo-dispatcher-000733"),
                                    ("Deployment", "pertinence-demo")]]
    manage.save(tmp_path / "state.json", {
        "context": manage.CONTEXT, "session": "s1", "resources": resources,
        "experts": [], "activated_experts": [], "pool": {"size": 1},
        "upgrade": {"status": status},
    })
    live = {(item["kind"], item["metadata"]["name"]): item for item in resources}
    monkeypatch.setattr(manage, "get", lambda kind, name: live.get((kind, name)))
    monkeypatch.setattr(manage, "pool_state", lambda: {"size": 1})

    def kube(*args):
        assert args[0] == "delete"
        live.pop((args[1], args[2]))
        return ""

    monkeypatch.setattr(manage, "kube", kube)
    manage.stop(tmp_path)
    assert not live
    assert json.loads((tmp_path / "state.json").read_text())["stopped"]
    manage.stop(tmp_path)
