#!/usr/bin/env python3
"""Run read-only environment, asset, dataset, device, and model checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pertinence.preflight import run_preflight  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", help="override experiment.device for this check")
    parser.add_argument("--smoke-samples", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = run_preflight(
            config_path=arguments.config,
            manifest_path=arguments.manifest,
            artifact_root=arguments.artifacts_dir,
            dataset_root=arguments.dataset_dir,
            source_root=arguments.source_dir,
            cache_directory=arguments.cache_dir,
            device_override=arguments.device,
            smoke_samples=arguments.smoke_samples,
        )
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"preflight failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
