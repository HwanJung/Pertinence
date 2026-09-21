"""Regression checks for create-only CIFAR-10.1 Job derivation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from copy import deepcopy
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "cloud" / "kserve" / "scripts" / "derive-cifar101-job.py"
VALIDATOR = ROOT / "cloud" / "kserve" / "scripts" / "validate_rendered.py"
DIGEST_IMAGE = "example.invalid/runtime@sha256:" + "a" * 64


def _job(name: str = "pertinence-loadgen", image: str = DIGEST_IMAGE) -> dict[str, object]:
    template = (ROOT / "cloud/kserve/manifests/10-workloads.yaml.tmpl").read_text(
        encoding="utf-8"
    )
    rendered = (
        template.replace("${RUNTIME_IMAGE}", image)
        .replace("${BENCHMARK_ID}", "benchmark-1")
        .replace(
            "${BENCHMARK_TRACE_URI}",
            "gs://pertinence-509116-pertinence-runs/pertinence/benchmarks/traces/trace-1",
        )
        .replace(
            "${BENCHMARK_OUTPUT_URI}",
            "gs://pertinence-509116-pertinence-runs/pertinence/benchmarks/runs/benchmark-1",
        )
    )
    job = next(
        document
        for document in yaml.safe_load_all(rendered)
        if isinstance(document, dict) and document.get("kind") == "Job"
    )
    job = deepcopy(job)
    job["metadata"]["name"] = name
    return job


def _write_source(path: Path, *jobs: dict[str, object]) -> bytes:
    documents: list[dict[str, object]] = [
        {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "inputs"}},
        *jobs,
    ]
    data = yaml.safe_dump_all(documents, explicit_start=True, sort_keys=False).encode()
    path.write_bytes(data)
    return data


def _run(source: Path, output: Path, receipt: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(source),
            "--output",
            str(output),
            "--receipt",
            str(receipt),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _run_validator(tmp_path: Path, job: dict[str, object]) -> subprocess.CompletedProcess[str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "job.yaml"
    manifest.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(manifest)],
        check=False,
        capture_output=True,
        text=True,
    )


def _set_path(document: dict[str, object], path: tuple[str | int, ...], value: Any) -> None:
    current: Any = document
    for component in path[:-1]:
        current = current[component]
    current[path[-1]] = value


def _remove_loadgen_argument(job: dict[str, object], flag: str) -> None:
    args = job["spec"]["template"]["spec"]["containers"][0]["args"]
    index = args.index(flag)
    del args[index : index + 2]


def _canonical(document: object) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def test_derives_only_renamed_job_and_seals_receipt(tmp_path: Path) -> None:
    source = tmp_path / "10-workloads.yaml"
    source_bytes = _write_source(source, _job())
    output = tmp_path / "cifar101-job.yaml"
    receipt_path = tmp_path / "cifar101-job.receipt.json"

    result = _run(source, output, receipt_path)

    assert result.returncode == 0, result.stderr
    derived = list(yaml.safe_load_all(output.read_text()))
    assert len(derived) == 1
    assert derived[0]["metadata"]["name"] == "pertinence-loadgen-cifar101-v6"
    restored = derived[0]
    restored["metadata"]["name"] = "pertinence-loadgen"
    assert _canonical(restored) == _canonical(_job())

    receipt = json.loads(receipt_path.read_text())
    assert receipt["valid"] is True
    assert receipt["source"]["file_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert receipt["destination"]["file_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt["destination"]["job_name"] == "pertinence-loadgen-cifar101-v6"
    assert receipt["equivalence"] == {
        "allowed_change": "/metadata/name",
        "canonical_bytes_equal_after_name_restore": True,
        "restored_destination_canonical_sha256": receipt["source"][
            "job_canonical_sha256"
        ],
    }
    assert receipt["validation"]["destination_valid"] is True


def test_refuses_to_overwrite_output_or_receipt(tmp_path: Path) -> None:
    source = tmp_path / "10-workloads.yaml"
    _write_source(source, _job())
    output = tmp_path / "cifar101-job.yaml"
    receipt = tmp_path / "receipt.json"
    output.write_text("owned by another run\n")

    result = _run(source, output, receipt)

    assert result.returncode != 0
    assert "File exists" in result.stderr
    assert output.read_text() == "owned by another run\n"
    assert not receipt.exists()


def test_rejects_ambiguous_or_wrong_source_job(tmp_path: Path) -> None:
    for case, jobs, message in (
        ("ambiguous", (_job(), _job("another-job")), "exactly one Job"),
        ("wrong-name", (_job("another-job"),), "expected Job metadata.name"),
    ):
        case_dir = tmp_path / case
        case_dir.mkdir()
        source = case_dir / "10-workloads.yaml"
        _write_source(source, *jobs)

        result = _run(source, case_dir / "output.yaml", case_dir / "receipt.json")

        assert result.returncode != 0
        assert message in result.stderr
        assert not (case_dir / "output.yaml").exists()
        assert not (case_dir / "receipt.json").exists()


def test_revalidates_and_rejects_mutable_image(tmp_path: Path) -> None:
    source = tmp_path / "10-workloads.yaml"
    _write_source(source, _job(image="example.invalid/runtime:latest"))
    output = tmp_path / "output.yaml"
    receipt = tmp_path / "receipt.json"

    result = _run(source, output, receipt)

    assert result.returncode != 0
    assert "not digest pinned" in result.stderr
    assert not output.exists()
    assert not receipt.exists()


def test_rejects_symlink_source(tmp_path: Path) -> None:
    target = tmp_path / "actual.yaml"
    _write_source(target, _job())
    source = tmp_path / "10-workloads.yaml"
    source.symlink_to(target)

    result = _run(source, tmp_path / "output.yaml", tmp_path / "receipt.json")

    assert result.returncode != 0
    assert "non-symlink" in result.stderr


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("metadata", "namespace"), "default", "namespace must be pertinence-serving"),
        (("spec", "suspend"), False, "suspend must be True"),
        (("spec", "parallelism"), 2, "parallelism must be 1"),
        (("spec", "completions"), 2, "completions must be 1"),
        (("spec", "backoffLimit"), 1, "backoffLimit must be 0"),
        (
            ("spec", "template", "spec", "restartPolicy"),
            "OnFailure",
            "restartPolicy must be Never",
        ),
        (
            ("spec", "template", "spec", "serviceAccountName"),
            "default",
            "must use benchmark-runner",
        ),
        (
            ("spec", "template", "spec", "nodeSelector"),
            {"cloud.google.com/gke-nodepool": "system-pool"},
            "must select loadgen-pool",
        ),
        (
            ("spec", "template", "spec", "tolerations"),
            [],
            "toleration differs",
        ),
        (
            ("spec", "template", "spec", "initContainers", 0, "restartPolicy"),
            "Never",
            "evidence collector restartPolicy must be Always",
        ),
        (
            ("spec", "template", "spec", "containers", 0, "securityContext", "runAsUser"),
            0,
            "loadgen security context differs",
        ),
        (
            ("spec", "template", "spec", "initContainers", 0, "resources", "limits", "cpu"),
            "1",
            "evidence collector resources differ",
        ),
        (
            ("spec", "template", "spec", "containers", 0, "image"),
            "example.invalid/runtime:latest",
            "not digest pinned",
        ),
        (
            ("spec", "template", "spec", "initContainers", 0, "args", 3),
            "/work/wrong/evidence.jsonl",
            "collector paths must match",
        ),
    ],
)
def test_job_validator_rejects_contract_mutations(
    tmp_path: Path, path: tuple[str | int, ...], value: Any, message: str
) -> None:
    job = _job()
    _set_path(job, path, value)

    result = _run_validator(tmp_path, job)

    assert result.returncode != 0
    assert message in result.stderr


def test_job_validator_rejects_unknown_or_extra_jobs(tmp_path: Path) -> None:
    unknown = _run_validator(tmp_path / "unknown", _job("unreviewed-loadgen"))
    assert unknown.returncode != 0
    assert "Job name must be one of" in unknown.stderr

    job = _job()
    pod = job["spec"]["template"]["spec"]
    pod["containers"].append(deepcopy(pod["containers"][0]))
    extra = _run_validator(tmp_path / "extra", job)
    assert extra.returncode != 0
    assert "exactly one loadgen container" in extra.stderr


def test_job_validator_rejects_pvc_and_missing_required_argument(tmp_path: Path) -> None:
    pvc_job = _job()
    volumes = pvc_job["spec"]["template"]["spec"]["volumes"]
    volumes[-1] = {"name": "work", "persistentVolumeClaim": {"claimName": "benchmark"}}
    pvc = _run_validator(tmp_path / "pvc", pvc_job)
    assert pvc.returncode != 0
    assert "direct GCS and emptyDir, not PVC" in pvc.stderr

    args_job = _job()
    _remove_loadgen_argument(args_job, "--trace-uri")
    missing_arg = _run_validator(tmp_path / "args", args_job)
    assert missing_arg.returncode != 0
    assert "benchmark arguments must be exactly" in missing_arg.stderr


def test_job_validator_rejects_inconsistent_output_ids(tmp_path: Path) -> None:
    job = _job()
    args = job["spec"]["template"]["spec"]["containers"][0]["args"]
    args[args.index("--output-uri") + 1] = (
        "gs://pertinence-509116-pertinence-runs/pertinence/benchmarks/runs/another-run"
    )

    result = _run_validator(tmp_path, job)

    assert result.returncode != 0
    assert "local and GCS output benchmark IDs must match" in result.stderr


@pytest.mark.parametrize("name", ["pertinence-loadgen", "pertinence-loadgen-cifar101-v6"])
def test_job_validator_accepts_only_approved_job_variants(tmp_path: Path, name: str) -> None:
    result = _run_validator(tmp_path, _job(name))

    assert result.returncode == 0, result.stderr


def test_job_validator_preserves_generic_addon_job_validation(tmp_path: Path) -> None:
    addon_job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "cert-manager-startupapicheck", "namespace": "cert-manager"},
        "spec": {
            "template": {
                "spec": {
                    "restartPolicy": "OnFailure",
                    "containers": [{"name": "check", "image": DIGEST_IMAGE}],
                }
            }
        },
    }

    valid = _run_validator(tmp_path / "valid", addon_job)
    assert valid.returncode == 0, valid.stderr

    addon_job["spec"]["template"]["spec"]["volumes"] = [
        {"name": "state", "persistentVolumeClaim": {"claimName": "unexpected"}}
    ]
    pvc = _run_validator(tmp_path / "pvc", addon_job)
    assert pvc.returncode != 0
    assert "Job must not use a PVC" in pvc.stderr
