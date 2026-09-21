from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).parents[1]
KSERVE = ROOT / "cloud" / "kserve"


def _chart(tmp_path: Path, name: str, template: str) -> Path:
    root = tmp_path / name
    (root / "templates").mkdir(parents=True)
    (root / "Chart.yaml").write_text(
        yaml.safe_dump({"apiVersion": "v2", "name": name, "version": "0.1.0"})
    )
    (root / "values.yaml").write_text("{}\n")
    (root / "templates" / "workload.yaml").write_text(template)
    subprocess.run(
        ["helm", "package", str(root), "--destination", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return tmp_path / f"{name}-0.1.0.tgz"


def test_local_release_archives_render_with_all_pinned_images(tmp_path: Path) -> None:
    cert = _chart(
        tmp_path,
        "cert-manager",
        """apiVersion: apps/v1
kind: Deployment
metadata: {name: cert-manager}
spec:
  template:
    spec:
      containers:
        - {name: controller, image: \"{{ .Values.image.repository }}@{{ .Values.image.digest }}\"}
        - {name: webhook, image: \"{{ .Values.webhook.image.repository }}@{{ .Values.webhook.image.digest }}\"}
        - {name: cainjector, image: \"{{ .Values.cainjector.image.repository }}@{{ .Values.cainjector.image.digest }}\"}
        - {name: startup, image: \"{{ .Values.startupapicheck.image.repository }}@{{ .Values.startupapicheck.image.digest }}\"}
        - {name: solver, image: \"{{ .Values.acmesolver.image.repository }}@{{ .Values.acmesolver.image.digest }}\"}
""",
    )
    crd = _chart(tmp_path, "kserve-crd", "apiVersion: v1\nkind: ConfigMap\nmetadata: {name: crd}\n")
    resources = _chart(
        tmp_path,
        "kserve-resources",
        """apiVersion: apps/v1
kind: Deployment
metadata: {name: kserve}
spec:
  template:
    spec:
      containers:
        - {name: controller, image: \"{{ .Values.kserve.controller.image }}:{{ .Values.kserve.controller.tag }}\"}
        - {name: proxy, image: \"{{ .Values.kserve.controller.rbacProxyImage }}\"}
        - {name: storage, image: \"{{ .Values.kserve.storage.image }}:{{ .Values.kserve.storage.tag }}\"}
        - {name: agent, image: \"{{ .Values.kserve.agent.image }}:{{ .Values.kserve.agent.tag }}\"}
        - {name: router, image: \"{{ .Values.kserve.router.image }}:{{ .Values.kserve.router.tag }}\"}
        - {name: art, image: \"{{ .Values.kserve.servingruntime.art.image }}:{{ .Values.kserve.servingruntime.art.defaultVersion }}\"}
""",
    )
    digest = "a" * 64
    lock = tmp_path / "lock.env"
    lines = []
    for prefix, archive, version in (
        ("CERT_MANAGER", cert, "v1.20.4"),
        ("KSERVE_CRD", crd, "v0.20.0"),
        ("KSERVE_RESOURCES", resources, "v0.20.0"),
    ):
        lines.extend(
            [
                f"{prefix}_CHART_PATH={archive}",
                f"{prefix}_CHART_VERSION={version}",
                f"{prefix}_CHART_SHA256={hashlib.sha256(archive.read_bytes()).hexdigest()}",
            ]
        )
    for key in (
        "CERT_MANAGER_CONTROLLER_DIGEST",
        "CERT_MANAGER_WEBHOOK_DIGEST",
        "CERT_MANAGER_CAINJECTOR_DIGEST",
        "CERT_MANAGER_STARTUPAPICHECK_DIGEST",
        "CERT_MANAGER_ACMESOLVER_DIGEST",
        "KSERVE_CONTROLLER_DIGEST",
        "KSERVE_STORAGE_INITIALIZER_DIGEST",
        "KSERVE_AGENT_DIGEST",
        "KSERVE_ROUTER_DIGEST",
        "KSERVE_ART_EXPLAINER_DIGEST",
    ):
        lines.append(f"{key}=sha256:{digest}")
    lines.append(f"KSERVE_RBAC_PROXY_IMAGE=quay.io/proxy@sha256:{digest}")
    lock.write_text("\n".join(lines) + "\n")
    output = tmp_path / "rendered"
    subprocess.run(
        [str(KSERVE / "scripts" / "render-install.sh"), str(lock), str(output)],
        check=True,
        env=dict(os.environ, PYTHON_BIN=sys.executable),
        capture_output=True,
        text=True,
    )
    namespaces = {
        item["metadata"]["name"]
        for item in yaml.safe_load_all((output / "00-namespaces.yaml").read_text())
    }
    assert namespaces == {"cert-manager", "kserve", "pertinence-serving"}
