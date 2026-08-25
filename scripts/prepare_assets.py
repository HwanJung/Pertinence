#!/usr/bin/env python3
"""Prepare the pinned CIFAR-10 and chenyaofo assets without running experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pertinence.assets import (  # noqa: E402
    AssetError,
    load_asset_manifest,
    prepare_assets,
    write_report_atomically,
)


def _workspace_artifact_root(value: str) -> Path:
    workspace_artifacts = (WORKSPACE_ROOT / "artifacts").resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace_artifacts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"asset destination must be below {workspace_artifacts}, got {resolved}"
        ) from error
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download or verify the CIFAR-10 tarball, exactly 10 chenyaofo Pareto "
            "checkpoints, and an immutable upstream source archive. This command "
            "only prepares assets; it does not run inference or training."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON/YAML manifest containing explicit HTTPS URLs and full SHA-256 hashes",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=_workspace_artifact_root,
        default=_workspace_artifact_root("artifacts"),
        help="destination below workspace artifacts/ (default: artifacts)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the plan without filesystem or network access",
    )
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="verify existing files without creating or downloading anything",
    )
    parser.add_argument(
        "--receipt-name",
        default="asset-manifest.json",
        help="verified output manifest filename (prepare mode only)",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if Path(arguments.receipt_name).name != arguments.receipt_name:
        parser.error("--receipt-name must be a filename, not a path")
    try:
        manifest = load_asset_manifest(arguments.manifest)
        report = prepare_assets(
            manifest,
            arguments.artifacts_dir,
            dry_run=arguments.dry_run,
            verify_only=arguments.verify_only,
            timeout_seconds=arguments.timeout,
        )
        if not arguments.dry_run and not arguments.verify_only:
            write_report_atomically(
                report, arguments.artifacts_dir / arguments.receipt_name
            )
    except (AssetError, OSError) as error:
        print(f"asset preparation failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
