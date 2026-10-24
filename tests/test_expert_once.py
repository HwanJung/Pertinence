"""Offline regression checks for the fixed, non-resumable expert comparison."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pertinence.serving.expert_once import MEASURED, MODELS, WARMUP, join_model_spans, run_once


class Clock:
    now = 0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)


@pytest.mark.parametrize("failure", [None, "request", "identity", "schedule", "environment",
                                     "warmup_request", "warmup_identity", "warmup_schedule"])
def test_one_pass_and_fail_closed(tmp_path, failure):
    clock = Clock()
    calls = []
    closed = []
    arms = [{"model_name": model, "endpoint": "http://unused", "model_sha256": "a" * 64,
             "runtime_image": "runtime@sha256:" + "b" * 64,
             "modelcar": "model@sha256:" + "c" * 64, "pod": {"uid": model}}
            for model in MODELS]

    class Client:
        def __init__(self, endpoint, model, *, timeout_seconds, request_id_prefix):
            assert request_id_prefix == f"test-{model}"
            assert timeout_seconds == 10
            self.model = model

        def __call__(self, image):
            calls.append(self.model)
            if (len(calls) == 202 and failure == "request") or (
                    len(calls) == 3 and failure == "warmup_request"):
                raise TimeoutError("no retry")
            late = (len(calls) == 202 and failure == "schedule") or (
                len(calls) == 3 and failure == "warmup_schedule")
            clock.now += 300_000_000 if late else 1_000_000
            return {"logits": np.zeros((1, 10), np.float32), "identity": {
                "model_name": self.model,
                "run_fingerprint": "wrong" if (len(calls) == 202 and failure == "identity") or (
                    len(calls) == 3 and failure == "warmup_identity")
                else "a" * 64}}

        def close(self):
            closed.append(self.model)

    def guard():
        if failure == "environment" and len(calls) == 202:
            raise RuntimeError("Pod restarted")

    output = tmp_path / "run"
    kwargs = dict(images=np.zeros((2000, 3, 32, 32), np.float32),
                  labels=np.zeros(2000, np.int64), sample_ids=[str(i) for i in range(2000)],
                  dataset_id="cifar10", run_id="test", arms=arms, output=output, guard=guard,
                  predictor_factory=Client, clock_ns=clock.clock, sleep=clock.sleep)
    result = run_once(**kwargs)
    rows = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    if failure in (None, "schedule", "warmup_schedule"):
        assert calls == [m for m in MODELS for _ in range(2200)]
        assert closed == list(MODELS)
        assert result["client_valid"]
        assert not result["valid"]
        for model in MODELS:
            warmup = [r for r in rows if r["model_name"] == model and r["phase"] == "warmup"]
            measured = [r for r in rows if r["model_name"] == model and r["phase"] == "measured"]
            assert len(warmup) == WARMUP and len(measured) == MEASURED
            assert [r["sample_id"] for r in measured] == [str(i) for i in range(2000)]
            assert {r["repeat"] for r in measured} == {0}
        assert all(s["requests"] == MEASURED for s in result["summaries"])
        assert len({r["request_id"] for r in rows}) == 6600
        for model in MODELS:
            selected = [r for r in rows if r["model_name"] == model]
            assert all(b["started_ns"] - a["started_ns"] >= 200_000_000
                       for a, b in zip(selected, selected[1:]))
            assert all(b["started_ns"] >= a["completed_ns"]
                       for a, b in zip(selected, selected[1:]))
            measured = [r for r in selected if r["phase"] == "measured"]
            summary = next(s for s in result["summaries"] if s["model"] == model)
            assert summary["achieved_rps"] == MEASURED * 1e9 / (
                measured[-1]["completed_ns"] - measured[0]["started_ns"])
            assert [r["sample_id"] for r in selected[:WARMUP]] == list(map(str, range(WARMUP)))
    else:
        assert not result["client_valid"]
        warmup_failure = failure.startswith("warmup_")
        assert len(calls) == (3 if warmup_failure else 202)
        assert closed == [MODELS[0]]
        assert result["summaries"][0]["requests"] == (0 if warmup_failure else 2)
        if warmup_failure:
            assert result["summaries"][0]["warmup_errors"] == int(failure != "warmup_schedule")
            assert result["summaries"][0]["mean_ms"] is None
        if failure in ("identity", "request"):
            assert result["summaries"][0]["errors"] == 1
            assert not rows[-1]["success"]
    with pytest.raises(FileExistsError):
        run_once(**kwargs)
    assert len(calls) == (6600 if failure in (None, "schedule", "warmup_schedule") else 3 if failure.startswith("warmup_") else 202)


def test_service_render_preserves_digests_and_equal_settings():
    path = Path("experiments/cifar10/scripts/run-expert-once.py")
    spec = importlib.util.spec_from_file_location("expert_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = {"apiVersion": "serving.kserve.io/v1beta1", "kind": "InferenceService",
                "metadata": {"name": "expert", "annotations": {}}, "spec": {"predictor": {
                    "storageUris": [{"uri": "oci://model@sha256:" + "a" * 64}],
                    "containers": [{"image": "runtime@sha256:" + "b" * 64, "args": [
                        "serve", "expert", "--model-name", MODELS[0], "--model-sha256", "c" * 64,
                        "--features-name", "features", "--feature-dim", "1024"]}]}}}
    rendered = module.render_service(original)
    c = rendered["spec"]["predictor"]["containers"][0]
    assert c["image"] == original["spec"]["predictor"]["containers"][0]["image"]
    assert "--features-name" not in c["args"]
    assert "--diagnostics-path" in c["args"]
    assert {"name": "diagnostics", "emptyDir": {}} in rendered["spec"]["predictor"]["volumes"]
    assert module.MEASUREMENT_TIMEOUT_SECONDS == 1800
    assert module.JOB_DEADLINE_SECONDS == 2400
    assert "--keep-alive" in c["args"]
    assert c["args"][c["args"].index("--run-fingerprint") + 1] == "c" * 64
    assert c["resources"] == {k: {"cpu": "1", "memory": "1Gi"} for k in ("requests", "limits")}
    assert rendered["spec"]["predictor"]["minReplicas"] == 1
    assert rendered["spec"]["predictor"]["maxReplicas"] == 1


@pytest.mark.parametrize("error", [None, "missing", "duplicate", "model", "negative", "nan",
                                  "infinity", "absent", "zero", "duplicate_request"])
def test_direct_span_join(error):
    rows = [{"request_id": "r-0", "model_name": MODELS[0], "phase": "warmup"},
            {"request_id": "r-1", "model_name": MODELS[0], "phase": "measured"}]
    span = {"stage": "expert_model", "request_id": "r-1", "expert": MODELS[0],
            "wall_seconds": 0.002, "thread_cpu_seconds": 0.001, "call_index": None}
    spans = [span, {"stage": "response_write"}]
    if error == "missing":
        spans = []
    elif error == "duplicate":
        spans.append(dict(span))
    elif error == "model":
        span["expert"] = MODELS[1]
    elif error in ("negative", "nan", "infinity", "zero"):
        span["wall_seconds"] = {"negative": -1, "nan": float("nan"),
                                "infinity": float("inf"), "zero": 0}[error]
    elif error == "absent":
        del span["thread_cpu_seconds"]
    elif error == "duplicate_request":
        rows.append(dict(rows[-1]))
    if error:
        with pytest.raises(ValueError):
            join_model_spans(rows, spans)
    else:
        joined = join_model_spans(rows, spans)
        assert len(joined) == 1
        assert joined[0]["model_wall_ms"] == 2
        assert joined[0]["model_thread_cpu_ms"] == 1


def test_final_summary_excludes_warmup_and_requires_all_spans(tmp_path):
    from pertinence.serving.expert_once import finalize

    raw = tmp_path / "raw/results"
    raw.mkdir(parents=True)
    rows, summaries = [], []
    for model in MODELS:
        spans = []
        for index in range(WARMUP + MEASURED):
            measured = index >= WARMUP
            request_id = f"run-{model}-{index}"
            rows.append({"request_id": request_id, "model_name": model,
                         "phase": "measured" if measured else "warmup", "success": True})
            spans.append({"request_id": request_id, "expert": model, "stage": "expert_model",
                          "wall_seconds": 0.002 if measured else 99,
                          "thread_cpu_seconds": 0.001 if measured else 99})
        (tmp_path / f"{model}-spans.jsonl").write_text(
            "\n".join(json.dumps(s) for s in spans) + "\n")
        summaries.append({"model": model, "requests": MEASURED, "warmup_requests": WARMUP,
                          "errors": 0, "warmup_errors": 0})
    (raw / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (raw / "result.json").write_text(json.dumps({"client_valid": True, "summaries": summaries}))
    result = finalize(tmp_path)
    assert result["valid"]
    assert all(s["model_spans"] == MEASURED and s["model_mean_ms"] == 2
               and s["model_p99_ms"] == 2 and s["model_thread_cpu_mean_ms"] == 1
               for s in result["summaries"])
    assert len((tmp_path / "joined-requests.jsonl").read_text().splitlines()) == 6000


def test_latency_boundary_includes_validation_excludes_hash_and_journal(tmp_path, monkeypatch):
    import pertinence.serving.expert_once as module

    monkeypatch.setattr(module, "WARMUP", 1)
    monkeypatch.setattr(module, "MEASURED", 2)
    clock = Clock()
    original_hash = module.hashlib.sha256
    original_dumps = module.json.dumps

    def hash_image(value):
        clock.now += 40_000_000
        return original_hash(value)

    def encode(value, **kwargs):
        clock.now += 50_000_000
        return original_dumps(value, **kwargs)

    monkeypatch.setattr(module.hashlib, "sha256", hash_image)
    monkeypatch.setattr(module.json, "dumps", encode)

    from collections import UserDict

    class TimedItems(UserDict):
        def keys(self):
            clock.now += 2_000_000
            return super().keys()

    class Client:
        def __init__(self, endpoint, model, **kwargs):
            self.model = model

        def __call__(self, image):
            clock.now += 1_000_000
            return {"logits": np.zeros((1, 10), np.float32),
                    "identity": TimedItems(model_name=self.model, run_fingerprint="hash"),
                    "stage_latency_ns": TimedItems(expert_ns=123)}

        def close(self):
            pass

    run_once(images=np.zeros((2, 3, 32, 32), np.float32), labels=np.zeros(2),
             sample_ids=["a", "b"], dataset_id="cifar10", run_id="boundary",
             arms=[{"model_name": m, "endpoint": "unused", "model_sha256": "hash",
                    "pod": {}, "modelcar": "car", "runtime_image": "runtime"} for m in MODELS],
             output=tmp_path / "run", guard=lambda: None, predictor_factory=Client,
             clock_ns=clock.clock, sleep=clock.sleep)
    rows = [json.loads(line) for line in
            (tmp_path / "run/requests.jsonl").read_text().splitlines()]
    assert all(row["latency_ns"] == 5_000_000 for row in rows)
    assert all(row["stage_latency_ns"] == {"expert_ns": 123} for row in rows)


@pytest.mark.parametrize("change", [None, "threads", "image", "resources", "args", "mount",
                                    "duplicate_env"])
def test_actual_pod_validation_allows_kserve_metadata_env(change):
    import copy

    path = Path("experiments/cifar10/scripts/run-expert-once.py")
    spec = importlib.util.spec_from_file_location("expert_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {"resources": {"limits": {"cpu": "1", "memory": "1Gi"}},
                "image": "runtime@sha256:fixed", "args": ["serve", "expert"],
                "env": [{"name": "ORT_INTRA_OP_NUM_THREADS", "value": "1"}],
                "volumeMounts": [{"name": "diagnostics", "mountPath": "/diagnostics"}]}
    actual = copy.deepcopy(expected)
    actual["env"].extend([{"name": "INFERENCE_SERVICE_NAME", "value": "expert-shufflenet"},
                          {"name": "MODEL_INIT_MODE", "value": "async"}])
    if change == "threads":
        actual["env"][0]["value"] = "2"
    elif change in ("image", "resources", "args"):
        actual[change] = None
    elif change == "mount":
        actual["volumeMounts"][0]["readOnly"] = True
    elif change == "duplicate_env":
        actual["env"].append(actual["env"][0])
    if change:
        with pytest.raises(RuntimeError):
            module.validate_expert_container(actual, expected)
    else:
        module.validate_expert_container(actual, expected)


def test_job_render_preserves_security_and_disables_retries_and_collectors():
    path = Path("experiments/cifar10/scripts/run-expert-once.py")
    spec = importlib.util.spec_from_file_location("expert_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = {"kind": "Job", "spec": {"template": {"spec": {
        "serviceAccountName": "benchmark-runner", "securityContext": {"runAsNonRoot": True},
        "initContainers": [{"name": "collector"}],
        "containers": [{"name": "loadgen", "securityContext": {"readOnlyRootFilesystem": True}},
                       {"name": "extra-collector"}],
        "tolerations": [{"key": "workload", "value": "pertinence-loadgen"}],
    }}}}
    job = module.render_job(base, "fresh-run", "image@sha256:fixed")
    assert job["spec"]["activeDeadlineSeconds"] == 2400
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["completions"] == job["spec"]["parallelism"] == 1
    pod = job["spec"]["template"]["spec"]
    assert pod["restartPolicy"] == "Never"
    assert len(pod["containers"]) == 1 and "initContainers" not in pod
    assert pod["serviceAccountName"] == "benchmark-runner"
    assert pod["securityContext"] == {"runAsNonRoot": True}
    assert pod["containers"][0]["securityContext"] == {"readOnlyRootFilesystem": True}
    assert pod["containers"][0]["image"] == "image@sha256:fixed"
    assert pod["containers"][0]["command"] == ["python", "/code/worker.py"]
    assert pod["nodeSelector"] == {"cloud.google.com/gke-nodepool": "loadgen-pool"}
    assert len(base["spec"]["template"]["spec"]["containers"]) == 2


def test_real_http_worker_span_join_and_report(tmp_path, monkeypatch):
    from contextlib import ExitStack
    import threading

    import pertinence.serving.expert_once as module
    from pertinence.serving.diagnostics import Diagnostics
    from pertinence.serving.http import ExpertModel, V2Application, make_server
    from pertinence.serving.runtime import ExpertResult

    monkeypatch.setattr(module, "WARMUP", 2)
    monkeypatch.setattr(module, "MEASURED", 3)
    clock = Clock()
    arms, runners = [], []

    class Runner:
        def __init__(self):
            self.samples = []

        def predict(self, image, *, include_features=False):
            assert not include_features
            self.samples.append(int(image[0, 0, 0, 0]))
            return ExpertResult(np.arange(10, dtype=np.float32)[None])

    with ExitStack() as stack:
        for model in MODELS:
            runner = Runner()
            runners.append(runner)
            app = V2Application(ExpertModel(model, runner), run_fingerprint="hash",
                                diagnostics=Diagnostics(tmp_path / f"{model}-spans.jsonl"))
            server = make_server(app, "127.0.0.1", 0, keep_alive=True, idle_timeout=10)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            stack.callback(thread.join)
            stack.callback(server.server_close)
            stack.callback(server.shutdown)
            arms.append({"model_name": model, "endpoint": f"http://127.0.0.1:{server.server_port}",
                         "model_sha256": "hash", "pod": {}, "runtime_image": "fixed",
                         "modelcar": "fixed"})
        images = np.zeros((3, 3, 32, 32), np.float32)
        images[:, 0, 0, 0] = [0, 1, 2]
        run_once(images=images, labels=np.zeros(3), sample_ids=["a", "b", "c"],
                 dataset_id="cifar10", run_id="http-test", arms=arms,
                 output=tmp_path / "raw/results", guard=lambda: None,
                 clock_ns=clock.clock, sleep=clock.sleep)
        result = module.finalize(tmp_path)
    assert result["valid"]
    assert all(runner.samples == [0, 1, 0, 1, 2] for runner in runners)
    assert all(summary["requests"] == summary["model_spans"] == 3
               and summary["errors"] == 0 and summary["model_mean_ms"] > 0
               for summary in result["summaries"])
    joined = [json.loads(line) for line in
              (tmp_path / "joined-requests.jsonl").read_text().splitlines()]
    assert {r["request_id"] for r in joined} == {
        f"http-test-{model}-{index}" for model in MODELS for index in range(2, 5)}
    path = Path("experiments/cifar10/scripts/run-expert-once.py")
    spec = importlib.util.spec_from_file_location("expert_driver", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.write_report(tmp_path, "http-test", result, [])
    assert "model_p99_ms" in (tmp_path / "results.ko.md").read_text()
