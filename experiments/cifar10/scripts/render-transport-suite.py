#!/usr/bin/env python3
"""Render one immutable suite arm; applying it does not start load generation."""

import argparse
import json
from pathlib import Path

import yaml

from pertinence.serving.cli.benchmark import _write_json_exclusive
from pertinence.serving.transport_deployment import render_services


def main():
    parser = argparse.ArgumentParser()
    for name in ("base-services", "manifest", "promotion", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--fused-modelcar", required=True)
    parser.add_argument("--hybrid-modelcar", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    promotion = json.loads(args.promotion.read_text())
    services = render_services(list(yaml.safe_load_all(args.base_services.read_text())),
                               manifest, promotion, fused_modelcar=args.fused_modelcar,
                               hybrid_modelcar=args.hybrid_modelcar)
    config = {"apiVersion": "v1", "kind": "ConfigMap", "immutable": True,
              "metadata": {"name": "transport-promotion-" + manifest["fingerprint"][:16],
                           "namespace": "pertinence-serving"},
              "data": {"promotion.json": json.dumps(promotion)}}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "workloads.yaml").write_text(yaml.safe_dump_all([config, *services]))
    _write_json_exclusive(args.output / "run.json", manifest)


if __name__ == "__main__":
    main()
