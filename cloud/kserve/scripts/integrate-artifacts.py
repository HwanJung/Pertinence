#!/usr/bin/env python3
"""Run the reviewed local serving-artifact integration gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from pertinence.serving.integration import integrate_serving_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-bundle", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expert-bundle", type=Path, required=True)
    parser.add_argument("--evaluated-bundle", type=Path, required=True)
    parser.add_argument("--dispatcher-id", required=True)
    parser.add_argument("--approved-dispatcher-sha256", required=True)
    parser.add_argument("--approved-label-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        integrate_serving_artifacts(
            dataset_bundle=args.dataset_bundle,
            split_manifest=args.split_manifest,
            expert_bundle=args.expert_bundle,
            evaluated_bundle=args.evaluated_bundle,
            dispatcher_id=args.dispatcher_id,
            approved_dispatcher_sha256=args.approved_dispatcher_sha256,
            approved_label_sha256=args.approved_label_sha256,
            output=args.output,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
