#!/usr/bin/env python3
"""Build offline expert predictions, routing labels, and extractor features."""

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
    load_asset_manifest,
    prepare_assets,
    verify_extracted_source,
)
from pertinence.cache import (  # noqa: E402
    CacheError,
    build_prediction_caches,
    make_cache_request,
)
from pertinence.config import load_config  # noqa: E402
from pertinence.data import evaluation_transform, load_cifar10, make_splits  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the three CIFAR-10 PERTINENCE caches from already prepared local "
            "datasets, pinned upstream source, and checkpoints. No downloads occur."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="root against which manifest checkpoint paths are resolved",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="local torchvision CIFAR-10 root containing cifar-10-batches-py",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="extracted local chenyaofo source root (never a GitHub repository name)",
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", help="override config experiment.device")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print paths and fingerprints without loading data, models, or checkpoints",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config = load_config(arguments.config)
        manifest = load_asset_manifest(arguments.manifest)
        if not arguments.dry_run:
            prepare_assets(manifest, arguments.artifacts_dir, verify_only=True)
            verify_extracted_source(
                manifest, arguments.artifacts_dir, arguments.source_dir
            )
        request = make_cache_request(
            config,
            manifest,
            artifact_root=arguments.artifacts_dir,
            source_root=arguments.source_dir,
            output_directory=arguments.cache_dir,
            device=arguments.device,
        )
        splits = make_splits(
            search_size=int(config.raw["data"]["ga_search_size"]),
            final_size=int(config.raw["data"]["final_evaluation_size"]),
            seed=int(config.raw["data"]["split_seed"]),
        )
        if arguments.dry_run:
            report = build_prediction_caches(request, splits, dry_run=True)
        else:
            transform = evaluation_transform(
                config.raw["data"]["normalize_mean"],
                config.raw["data"]["normalize_std"],
            )
            train_dataset = load_cifar10(arguments.dataset_dir, train=True, transform=transform)
            test_dataset = load_cifar10(arguments.dataset_dir, train=False, transform=transform)
            report = build_prediction_caches(
                request,
                splits,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
            )
    except (
        CacheError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"cache build failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
