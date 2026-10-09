from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "cloud" / "kserve" / "scripts"


def _executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o700)


def _fake_kubectl(directory: Path) -> Path:
    path = directory / "kubectl"
    _executable(
        path,
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >>"$KUBE_LOG"
if [[ "$*" == "config current-context" ]]; then
  printf '%s\\n' "$EXPECTED_CONTEXT"
  exit 0
fi
if [[ "$*" == *" get service pertinence-active "* ]]; then
  printf '%s' 'router-static-resnet'
  exit 0
fi
if [[ "$*" == *" get inferenceservice "* || "$*" == *" get job "* ]]; then
  printf '%s' 'true'
  exit 0
fi
if [[ "$*" == *" port-forward "* ]]; then
  exec /bin/sleep 60
fi
if [[ "${FAIL_APPLY:-}" == true && "$*" == *" apply --server-side "* ]]; then
  exit 1
fi
if [[ "${FAIL_FIRST_STOP:-}" == true && "$*" == *"annotate inferenceservice expert-shufflenet "* ]]; then
  exit 1
fi
if [[ "$*" == *" get inferenceservices,pods "* ]]; then
  printf '%s\\n' '{"items":[]}'
fi
""",
    )
    return path


def _environment(bin_dir: Path, log: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "KUBE_LOG": str(log),
        "EXPECTED_CONTEXT": "expected-context",
    }


def test_stop_attempts_every_target_after_one_failure(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_kubectl(bin_dir)
    log = tmp_path / "kubectl.log"
    environment = _environment(bin_dir, log)
    environment["FAIL_FIRST_STOP"] = "true"

    result = subprocess.run(
        [str(SCRIPTS / "stop.sh"), "expected-context"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    calls = log.read_text(encoding="utf-8").splitlines()
    assert sum(" annotate inferenceservice " in call for call in calls) == 4
    assert sum(" patch job " in call for call in calls) == 1
    assert all("--context expected-context" in call for call in calls[1:])
    assert "no object was deleted" in result.stderr


def test_switch_uses_pinned_context_and_fresh_identity_checks(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_kubectl(bin_dir)
    _executable(
        bin_dir / "curl",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == *"/v2/health/ready"* ]]; then
  exit 0
fi
printf '%s\\n' '{"model_name":"static-resnet-router","model_version":"1","parameters":{"promotion_id":"promotion-1","dispatcher_id":"static","expert_id":"resnet56"},"outputs":[{"name":"logits","datatype":"FP32","shape":[1,10],"data":[0,0,0,0,0,0,0,0,0,0]},{"name":"route","datatype":"INT64","shape":[1],"data":[2]}]}'
""",
    )
    log = tmp_path / "kubectl.log"
    payload = tmp_path / "request.json"
    payload.write_text("{}\n", encoding="utf-8")
    environment = _environment(bin_dir, log)
    environment["DRAIN_SECONDS"] = "0"

    result = subprocess.run(
        [str(SCRIPTS / "switch-active.sh"), "expected-context", "static", str(payload)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert all("--context expected-context" in call for call in calls[1:])
    assert any("patch service pertinence-active" in call for call in calls)
    assert "20 fresh responses matched static" in result.stdout


def test_cold_start_failure_preserves_evidence_and_reapplies_stop(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_kubectl(bin_dir)
    log = tmp_path / "kubectl.log"
    payload = tmp_path / "request.json"
    payload.write_text("{}\n", encoding="utf-8")
    workload = tmp_path / "workloads.yaml"
    workload.write_text("---\nkind: List\nitems: []\n", encoding="utf-8")
    output = tmp_path / "cold-start"
    environment = _environment(bin_dir, log)
    environment["FAIL_APPLY"] = "true"

    result = subprocess.run(
        [
            str(SCRIPTS / "measure-cold-start.sh"),
            "expected-context",
            str(workload),
            str(payload),
            str(output),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert (output / "failure-stop.txt").is_file()
    calls = log.read_text(encoding="utf-8").splitlines()
    assert any("apply --server-side" in call for call in calls)
    assert sum(" annotate inferenceservice " in call for call in calls) == 4
    assert sum(" patch job " in call for call in calls) == 1
    assert all(
        call == "config current-context" or "--context expected-context" in call
        for call in calls
    )


def test_external_collector_can_follow_job_to_terminal_condition(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(
        bin_dir / "kubectl",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$KUBE_LOG"
if [[ "$*" == "config current-context" ]]; then
  printf '%s\n' "$EXPECTED_CONTEXT"
elif [[ "$*" == *" get nodes -l "*"serving-pool"* ]]; then
  printf '%s\n' 'serving-node'
elif [[ "$*" == *" get nodes -l "*"loadgen-pool"* ]]; then
  printf '%s\n' 'loadgen-node'
elif [[ "$*" == *" get job pertinence-loadgen -o json"* ]]; then
  count=0
  [[ ! -f "$JOB_STATE_FILE" ]] || count=$(<"$JOB_STATE_FILE")
  count=$((count + 1))
  printf '%s\n' "$count" >"$JOB_STATE_FILE"
  if ((count >= 2)); then
    printf '%s\n' '{"status":{"conditions":[{"type":"Complete","status":"True"}]}}'
  else
    printf '%s\n' '{"status":{"conditions":[]}}'
  fi
elif [[ "$*" == *" get nodes -o json"* ]]; then
  printf '%s\n' '{"items":[{"metadata":{"name":"serving-node","labels":{"cloud.google.com/gke-nodepool":"serving-pool","node.kubernetes.io/instance-type":"n2-standard-8"}},"spec":{"providerID":"provider-serving"}},{"metadata":{"name":"loadgen-node","labels":{"cloud.google.com/gke-nodepool":"loadgen-pool","node.kubernetes.io/instance-type":"n2-standard-4"}},"spec":{"providerID":"provider-loadgen"}}]}'
elif [[ "$*" == *" get pods -o json"* || "$*" == *" get workflows.argoproj.io -o json"* ]]; then
  printf '%s\n' '{"items":[]}'
elif [[ "$*" == *" -o json"* ]]; then
  printf '%s\n' '{"items":[]}'
elif [[ "$*" == *" get --raw "* ]]; then
  printf '%s\n' '{}'
else
  printf '%s\n' 'ok'
fi
""",
    )
    _executable(bin_dir / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    log = tmp_path / "kubectl.log"
    output = tmp_path / "external-window"
    environment = _environment(bin_dir, log)
    environment["JOB_STATE_FILE"] = str(tmp_path / "job-state")

    result = subprocess.run(
        [
            str(SCRIPTS / "collect-evidence.sh"),
            "expected-context",
            str(output),
            "until-job:pertinence-loadgen",
            "5",
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert (output / "sample-0001.job.json").is_file()
    assert (output / "sample-0002.job.json").is_file()
    assert json.loads((output / "validation.json").read_text())["snapshot_invalid_reasons"] == []
    assert not (output / "INVALID.json").exists()


def _load_preflight_module():
    path = SCRIPTS / "preflight-gke.py"
    spec = importlib.util.spec_from_file_location("preflight_gke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _preflight_output(
    command: list[str],
    *,
    configured_project: str = "project-1",
    all_regions_cpu_available: int = 20,
) -> str:
    joined = " ".join(command)
    if command == ["kubectl", "config", "current-context"]:
        return "gke_project-1_region-1-a_cluster-1\n"
    if "auth list" in joined:
        return "operator@example.com\n"
    if "config get-value project" in joined:
        return configured_project + "\n"
    if "container clusters describe" in joined:
        return json.dumps({"name": "cluster-1", "location": "region-1-a", "status": "RUNNING"})
    if "container node-pools list" in joined:
        return "[]"
    if "compute regions describe" in joined:
        return json.dumps(
            {"quotas": [{"metric": metric, "limit": 20, "usage": 0} for metric in ("CPUS", "N2_CPUS")]}
        )
    if "compute project-info describe" in joined:
        return json.dumps(
            {
                "quotas": [
                    {
                        "metric": "CPUS_ALL_REGIONS",
                        "limit": all_regions_cpu_available + 4,
                        "usage": 4,
                    }
                ]
            }
        )
    if " version -o json" in joined:
        return json.dumps({"serverVersion": {"major": "1", "minor": "35"}})
    if " get nodes -o json" in joined and "pipeline-pool" not in joined:
        return json.dumps({"items": [{"status": {"nodeInfo": {"kubeletVersion": "v1.35.8"}}}]})
    if " get namespace kubeflow " in joined:
        return json.dumps({"status": {"phase": "Active"}})
    if " get pvc " in joined:
        return json.dumps(
            {"items": [{"metadata": {"namespace": "kubeflow"}, "spec": {"volumeName": "pv-1"}, "status": {"phase": "Bound"}}]}
        )
    if " get pv " in joined:
        return json.dumps({"items": [{"metadata": {"name": "pv-1"}, "status": {"phase": "Bound"}}]})
    if "workflows.argoproj.io" in joined or "pipeline-pool" in joined:
        return '{"items":[]}'
    if "get serviceaccount benchmark-runner" in joined:
        return json.dumps(
            {"kind": "ServiceAccount", "metadata": {"name": "benchmark-runner", "namespace": "pertinence-serving"}}
        )
    if "artifacts repositories get-iam-policy" in joined:
        return '{"bindings":[]}'
    if "projects get-iam-policy" in joined:
        return json.dumps({
            "bindings": [{
                "role": "roles/artifactregistry.reader",
                "members": ["serviceAccount:node@project-1.iam.gserviceaccount.com"],
            }]
        })
    if " auth can-i " in joined:
        return "yes\n"
    if "api-resources" in joined:
        return ""
    if "metrics.k8s.io" in joined:
        return "{}"
    if " get service " in joined or " get ingress " in joined:
        return '{"items":[]}'
    if "forwarding-rules list" in joined:
        return "[]"
    raise AssertionError(f"unexpected command: {command}")


@pytest.mark.parametrize("configured_project", ("project-1", "wrong-project"))
def test_preflight_seals_cluster_and_project_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured_project: str
) -> None:
    module = _load_preflight_module()
    monkeypatch.setattr(
        module,
        "run",
        lambda command: _preflight_output(command, configured_project=configured_project),
    )
    output = tmp_path / "preflight"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight-gke.py",
            "--context", "gke_project-1_region-1-a_cluster-1",
            "--project", "project-1",
            "--region", "region-1",
            "--zone", "region-1-a",
            "--cluster", "cluster-1",
            "--repository", "repository-1",
            "--principal", "operator@example.com",
            "--node-service-account", "node@project-1.iam.gserviceaccount.com",
            "--output", str(output),
        ],
    )

    if configured_project == "project-1":
        assert module.main() == 0
        assert json.loads((output / "receipt.json").read_text())["valid"] is True
        assert (output / "cluster.json").is_file()
        assert (output / "pvs.json").is_file()
    else:
        with pytest.raises(SystemExit):
            module.main()
        receipt = json.loads((output / "receipt.json").read_text())
        assert receipt["valid"] is False
        assert receipt["invalid_reasons"] == ["active gcloud project differs"]


def test_preflight_rejects_insufficient_all_regions_cpu_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_preflight_module()
    monkeypatch.setattr(
        module,
        "run",
        lambda command: _preflight_output(command, all_regions_cpu_available=8),
    )
    output = tmp_path / "preflight"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight-gke.py",
            "--context", "gke_project-1_region-1-a_cluster-1",
            "--project", "project-1",
            "--region", "region-1",
            "--zone", "region-1-a",
            "--cluster", "cluster-1",
            "--repository", "repository-1",
            "--principal", "operator@example.com",
            "--node-service-account", "node@project-1.iam.gserviceaccount.com",
            "--output", str(output),
            "--quota-only",
        ],
    )

    with pytest.raises(SystemExit):
        module.main()

    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["valid"] is False
    assert receipt["invalid_reasons"] == [
        "all-regions CPU quota has less than 12 vCPUs available"
    ]
    assert (output / "global-quota.json").is_file()
    assert not (output / "cluster.json").exists()
