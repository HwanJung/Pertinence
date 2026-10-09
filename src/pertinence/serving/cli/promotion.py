"""Promotion command registration and execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..contracts import PromotionManifestV1, create_promotion, seal_manifest


def add_commands(subparsers: argparse._SubParsersAction) -> None:
    promotion = subparsers.add_parser("promotion", help="create or verify a serving promotion")
    commands = promotion.add_subparsers(dest="promotion_command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--manifest", type=Path, required=True, help="JSON draft or sealed manifest")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--evaluated-bundle", type=Path, required=True)
    create.set_defaults(serving_action="promotion-create")
    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--evaluated-bundle", type=Path, required=True)
    verify.set_defaults(serving_action="promotion-verify")


def run_command(args: argparse.Namespace) -> int:
    from ..promotion import validate_promotion_bundle

    value = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.serving_action == "promotion-create":
        if "fingerprint" not in value:
            value = seal_manifest(value)
        validate_promotion_bundle(value, args.evaluated_bundle)
        created = create_promotion(args.output, value)
        print("created" if created else "already exists with identical content")
    else:
        manifest = PromotionManifestV1.from_dict(value)
        validate_promotion_bundle(manifest, args.evaluated_bundle)
        print(manifest.to_dict()["fingerprint"])
    return 0
