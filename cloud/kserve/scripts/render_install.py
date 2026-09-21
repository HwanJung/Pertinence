#!/usr/bin/env python3
"""Verify supplied official Helm archives and render pinned manifests."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

import yaml

from render_config import VARIABLE_RE, read_env


VERSIONS = {
    "CERT_MANAGER_CHART_VERSION": "v1.20.4",
    "KSERVE_CRD_CHART_VERSION": "v0.20.0",
    "KSERVE_RESOURCES_CHART_VERSION": "v0.20.0",
}
CHARTS = (
    ("CERT_MANAGER", "cert-manager", "cert-manager"),
    ("KSERVE_CRD", "kserve-crd", "kserve"),
    ("KSERVE_RESOURCES", "kserve-resources", "kserve"),
)


def run(command: list[str], *, stdout: Path | None = None) -> None:
    if stdout is None:
        subprocess.run(command, check=True)
        return
    with stdout.open("xb") as stream:
        subprocess.run(command, check=True, stdout=stream)


def render_values(template: Path, values: dict[str, str], output: Path) -> None:
    text = template.read_text(encoding="utf-8")
    required = set(VARIABLE_RE.findall(text))
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"missing values inputs: {', '.join(missing)}")
    rendered = VARIABLE_RE.sub(lambda match: values[match.group(1)], text)
    list(yaml.safe_load_all(rendered))
    output.write_text(rendered, encoding="utf-8")


def validate_lock(values: dict[str, str]) -> None:
    expected = {
        *(f"{prefix}_CHART_PATH" for prefix, _, _ in CHARTS),
        *(f"{prefix}_CHART_VERSION" for prefix, _, _ in CHARTS),
        *(f"{prefix}_CHART_SHA256" for prefix, _, _ in CHARTS),
        "CERT_MANAGER_CONTROLLER_DIGEST",
        "CERT_MANAGER_WEBHOOK_DIGEST",
        "CERT_MANAGER_CAINJECTOR_DIGEST",
        "CERT_MANAGER_STARTUPAPICHECK_DIGEST",
        "CERT_MANAGER_ACMESOLVER_DIGEST",
        "KSERVE_CONTROLLER_DIGEST",
        "KSERVE_RBAC_PROXY_IMAGE",
        "KSERVE_STORAGE_INITIALIZER_DIGEST",
        "KSERVE_AGENT_DIGEST",
        "KSERVE_ROUTER_DIGEST",
        "KSERVE_ART_EXPLAINER_DIGEST",
    }
    if values.keys() != expected:
        missing = sorted(expected - values.keys())
        extra = sorted(values.keys() - expected)
        raise ValueError(f"lock keys differ; missing={missing}, extra={extra}")
    for key, expected_version in VERSIONS.items():
        if values[key] != expected_version:
            raise ValueError(f"{key} must be {expected_version}")
    for prefix, _, _ in CHARTS:
        archive = Path(values[f"{prefix}_CHART_PATH"])
        if not archive.is_absolute() or archive.suffix != ".tgz":
            raise ValueError(f"{prefix}_CHART_PATH must be an absolute release .tgz path")
        digest = values[f"{prefix}_CHART_SHA256"]
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or digest == "0" * 64:
            raise ValueError(f"{prefix}_CHART_SHA256 is not a non-sentinel SHA-256")
    for key, value in values.items():
        if key.endswith("_DIGEST"):
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", value) or value == "sha256:" + "0" * 64:
                raise ValueError(f"{key} is not a non-sentinel OCI digest")
    image = values["KSERVE_RBAC_PROXY_IMAGE"]
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image) or image.endswith("0" * 64):
        raise ValueError("KSERVE_RBAC_PROXY_IMAGE must be digest pinned")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lock", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    lock_path = args.lock.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")
    try:
        values = read_env(lock_path)
        validate_lock(values)
        script_dir = Path(__file__).resolve().parent
        install_dir = script_dir.parent / "install"
        with tempfile.TemporaryDirectory(prefix="pertinence-kserve-") as temporary:
            temp = Path(temporary)
            values_paths = {
                "CERT_MANAGER": temp / "cert-manager-values.yaml",
                "KSERVE_RESOURCES": temp / "kserve-values.yaml",
            }
            render_values(install_dir / "cert-manager-values.yaml.tmpl", values,
                          values_paths["CERT_MANAGER"])
            render_values(install_dir / "kserve-values.yaml.tmpl", values,
                          values_paths["KSERVE_RESOURCES"])

            archives: dict[str, Path] = {}
            for prefix, chart_name, _ in CHARTS:
                archive = Path(values[f"{prefix}_CHART_PATH"]).resolve(strict=True)
                actual = hashlib.sha256(archive.read_bytes()).hexdigest()
                expected = values[f"{prefix}_CHART_SHA256"]
                if actual != expected:
                    raise ValueError(
                        f"{chart_name} package SHA-256 mismatch: expected {expected}, got {actual}"
                    )
                archives[prefix] = archive

            output_dir.mkdir(mode=0o700, parents=False)
            (output_dir / "00-namespaces.yaml").write_text(
                "apiVersion: v1\nkind: Namespace\nmetadata:\n  name: cert-manager\n"
                "---\napiVersion: v1\nkind: Namespace\nmetadata:\n  name: kserve\n"
                "---\napiVersion: v1\nkind: Namespace\nmetadata:\n  name: pertinence-serving\n",
                encoding="utf-8",
            )
            run([
                "helm", "template", "cert-manager", str(archives["CERT_MANAGER"]),
                "--namespace", "cert-manager", "--include-crds", "--values",
                str(values_paths["CERT_MANAGER"]),
            ], stdout=output_dir / "cert-manager.yaml")
            run([
                "helm", "template", "kserve-crd", str(archives["KSERVE_CRD"]),
                "--namespace", "kserve",
            ], stdout=output_dir / "kserve-crd.yaml")
            run([
                "helm", "template", "kserve", str(archives["KSERVE_RESOURCES"]),
                "--namespace", "kserve", "--values", str(values_paths["KSERVE_RESOURCES"]),
            ], stdout=output_dir / "kserve-resources.yaml")
    except (OSError, ValueError, subprocess.CalledProcessError, yaml.YAMLError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
