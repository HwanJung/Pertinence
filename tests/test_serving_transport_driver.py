"""Artifact retrieval failures must not cause inference retries or silently lose evidence."""

import importlib.util
import io
from pathlib import Path
import subprocess
import zipfile

import pytest


def driver():
    path = Path(__file__).parents[1] / "experiments/cifar10/scripts/run-transport-suite.py"
    spec = importlib.util.spec_from_file_location("transport_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_retries_only_artifact_read_and_preserves_error(tmp_path, monkeypatch):
    module = driver()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zipped:
        zipped.writestr("spans.jsonl", '{"stage":"model"}\n')
    calls = []

    def kubectl(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess([], int(len(calls) == 1),
                                           stdout=payload.getvalue(), stderr=b"stream closed")

    monkeypatch.setattr(module, "kubectl", kubectl)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    module.archive("pod", "kserve-container", "/diagnostics", tmp_path / "result")
    assert len(calls) == 2
    assert all(call[0][0] == "exec" and call[1]["timeout_seconds"] == 120 for call in calls)
    assert (tmp_path / "result/spans.jsonl").read_text() == '{"stage":"model"}\n'
    assert (tmp_path / "result.attempt-0.error.txt").read_text() == "stream closed"


def test_archive_stops_after_three_failures(tmp_path, monkeypatch):
    module = driver()
    calls = []

    def kubectl(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"denied")

    monkeypatch.setattr(module, "kubectl", kubectl)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="denied"):
        module.archive("pod", "kserve-container", "/diagnostics", tmp_path / "result")
    assert len(calls) == 3
    assert not (tmp_path / "result").exists()


def test_status_reads_retry_without_reissuing_inference(monkeypatch):
    module = driver()
    calls = []

    def kubectl(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess([], int(len(calls) < 3), stdout=b'{"items": []}',
                                           stderr=b"deadline exceeded")

    monkeypatch.setattr(module, "kubectl", kubectl)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    assert module.get("pods") == {"items": []}
    assert len(calls) == 3
    assert all(call[0][0] == "get" for call in calls)


@pytest.mark.parametrize("changed", [False, True])
def test_completed_arm_resume_requires_identical_contract(tmp_path, monkeypatch, changed):
    import json
    from types import SimpleNamespace
    from pertinence.serving.transport_suite import run_manifest

    module = driver()
    promotion = tmp_path / "promotion.json"
    promotion.write_text(json.dumps({"fingerprint": "a" * 64}))
    args = SimpleNamespace(session="session", output=tmp_path, promotion=promotion,
                           trace_fingerprint="b" * 64, runtime="runtime@sha256:" + "c" * 64)
    path = tmp_path / "session-diag-d0"
    path.mkdir()
    manifest = run_manifest(run_id=path.name, arm="D0", block=0, diagnostics=True,
                            promotion_fingerprint="a" * 64, trace_fingerprint="b" * 64,
                            runtime_image=args.runtime, serving_node_uid="s", loadgen_node_uid="l")
    (path / "manifest.json").write_text(json.dumps(manifest))
    (path / "resources.json").write_text("{}")
    monkeypatch.setattr(module, "node_identity", lambda: ("s", "l"))
    monkeypatch.setattr(module, "stop_services", lambda: None)
    if changed:
        args.trace_fingerprint = "d" * 64
        with pytest.raises(RuntimeError, match="incompatible"):
            module.execute(args, "D0", 0, True, ("s", "l"), "diag")
    else:
        assert module.execute(args, "D0", 0, True, ("s", "l"), "diag") == path
