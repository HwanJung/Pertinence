#!/usr/bin/env python3
"""Submit a minimal demo build context and save its immutable image reference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = "python:3.11-slim@sha256:174bec68e0451bffabbb08c7d5d21c6b253f772d81d52b9558af97bb3159b761"
REPOSITORY = "asia-northeast3-docker.pkg.dev/pertinence-509116/pertinence/demo"


def run(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--base-image", default=BASE)
    args = parser.parse_args()
    if not re.fullmatch(r"python:3\.11-slim@sha256:[0-9a-f]{64}", args.base_image):
        parser.error("base image must pin Python 3.11 slim by digest")
    directory = args.directory.resolve()
    if not directory.is_relative_to(ROOT / "artifacts" / "cifar10"):
        parser.error("output must be under artifacts/cifar10/")
    directory.mkdir(parents=True, exist_ok=False)
    tag = "demo-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ").lower()
    with tempfile.TemporaryDirectory(prefix="pertinence-demo-build-") as temporary:
        context = Path(temporary)
        for file in ("pyproject.toml", "README.md"):
            shutil.copyfile(ROOT / file, context / file)
        shutil.copytree(ROOT / "src", context / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
        shutil.copytree(ROOT / "cloud" / "demo", context / "cloud" / "demo",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        submitted = json.loads(run(
            "gcloud", "builds", "submit", str(context), "--project=pertinence-509116",
            "--config=" + str(ROOT / "cloud/demo/cloudbuild.yaml"),
            "--gcs-source-staging-dir=gs://pertinence-509116_cloudbuild/source",
            f"--substitutions=_IMAGE={REPOSITORY}:{tag},_PYTHON_BASE_IMAGE={args.base_image}",
            "--async", "--format=json", "--quiet",
        ))
    build_id = submitted["id"]
    (directory / "submitted.json").write_text(json.dumps(submitted, indent=2) + "\n")
    print(f"Cloud Build {build_id}", flush=True)
    while True:
        result = json.loads(run("gcloud", "builds", "describe", build_id,
                                "--project=pertinence-509116", "--format=json"))
        (directory / "build.json").write_text(json.dumps(result, indent=2) + "\n")
        status = result["status"]
        if status not in {"QUEUED", "PENDING", "WORKING"}:
            break
        time.sleep(5)
    if status != "SUCCESS":
        raise SystemExit(f"Build {status}. Inspect gcloud builds log {build_id}.")
    image = next(item for item in result["results"]["images"]
                 if item["name"] == f"{REPOSITORY}:{tag}")
    reference = f"{REPOSITORY}@{image['digest']}"
    (directory / "image.txt").write_text(reference + "\n")
    print(reference, flush=True)


if __name__ == "__main__":
    main()
