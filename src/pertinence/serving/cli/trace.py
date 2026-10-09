"""Trace command registration and execution."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..trace import (
    build_cifar101_v6_trace,
    build_final_trace_from_npz,
    create_trace,
    source_object,
    verify_trace,
)


def add_commands(subparsers: argparse._SubParsersAction) -> None:
    trace = subparsers.add_parser("trace", help="create or verify an immutable benchmark trace")
    commands = trace.add_subparsers(dest="trace_command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--input", type=Path, required=True, help="NPZ with images/labels/sample_ids")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--trace-id", required=True)
    create.add_argument("--dataset-id", required=True)
    create.add_argument("--dataset-version", required=True)
    create.add_argument("--source-uri")
    provenance_group = create.add_mutually_exclusive_group(required=True)
    provenance_group.add_argument("--source-generation", type=int)
    provenance_group.add_argument("--source-revision")
    create.add_argument("--provenance", required=True)
    create.add_argument("--usage-conditions", required=True)
    create.add_argument("--require-class-balanced", action="store_true")
    create.set_defaults(serving_action="trace-create")
    final = commands.add_parser("create-final")
    final.add_argument("--evaluation-npz", type=Path, required=True)
    final.add_argument("--split-manifest", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--trace-id", default="cifar10-final-20260824")
    final.add_argument("--expected-label-sha256", required=True)
    final.add_argument("--evaluation-uri", required=True)
    final.add_argument("--evaluation-generation", type=int, required=True)
    final.add_argument("--split-uri", required=True)
    final.add_argument("--split-generation", type=int, required=True)
    final.set_defaults(serving_action="trace-create-final")
    cifar101 = commands.add_parser("create-cifar10.1-v6")
    cifar101.add_argument("--data-npy", type=Path, required=True)
    cifar101.add_argument("--labels-npy", type=Path, required=True)
    cifar101.add_argument("--output", type=Path, required=True)
    cifar101.add_argument("--trace-id", default="cifar10.1-v6")
    cifar101.set_defaults(serving_action="trace-create-cifar101")
    publish = commands.add_parser("publish")
    publish.add_argument("--trace", type=Path, required=True)
    publish.add_argument("--uri", required=True)
    publish.set_defaults(serving_action="trace-publish")
    download = commands.add_parser("download")
    download.add_argument("--uri", required=True)
    download.add_argument("--output", type=Path, required=True)
    download.set_defaults(serving_action="trace-download")
    verify = commands.add_parser("verify")
    verify.add_argument("--trace", type=Path, required=True)
    verify.set_defaults(serving_action="trace-verify")


def run_command(args: argparse.Namespace) -> int:
    if args.serving_action == "trace-create-final":
        manifest = build_final_trace_from_npz(
            args.evaluation_npz,
            args.split_manifest,
            args.output,
            expected_label_sha256=args.expected_label_sha256,
            evaluation_uri=args.evaluation_uri,
            evaluation_generation=args.evaluation_generation,
            split_uri=args.split_uri,
            split_generation=args.split_generation,
            trace_id=args.trace_id,
        )
        print(manifest["fingerprint"])
        return 0
    if args.serving_action == "trace-create-cifar101":
        manifest = build_cifar101_v6_trace(
            args.data_npy, args.labels_npy, args.output, trace_id=args.trace_id
        )
        print(manifest["fingerprint"])
        return 0
    if args.serving_action == "trace-publish":
        from ..gcs import publish_create_only

        verify_trace(args.trace)
        publish_create_only(args.trace, args.uri, kind="trace")
        return 0
    if args.serving_action == "trace-download":
        from ..gcs import download_trace

        download_trace(args.uri, args.output)
        print(verify_trace(args.output)[0]["fingerprint"])
        return 0
    if args.serving_action == "trace-verify":
        manifest, _, _, _ = verify_trace(args.trace)
        print(manifest["fingerprint"])
        return 0
    with np.load(args.input, allow_pickle=False) as archive:
        if set(archive.files) != {"images", "labels", "sample_ids"}:
            raise ValueError("trace input NPZ must contain exactly images, labels, sample_ids")
        images = archive["images"]
        labels = archive["labels"]
        sample_ids = archive["sample_ids"].tolist()
    if args.require_class_balanced:
        counts = np.bincount(labels, minlength=10)
        if len(counts) != 10 or len(set(counts.tolist())) != 1:
            raise ValueError("trace is not class balanced")
    manifest = create_trace(
        args.output,
        trace_id=args.trace_id,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        images=images,
        labels=labels,
        sample_ids=sample_ids,
        source_objects=[
            source_object(
                args.input,
                uri=args.source_uri,
                generation=args.source_generation,
                revision=args.source_revision,
            )
        ],
        provenance=args.provenance,
        usage_conditions=args.usage_conditions,
    )
    print(manifest["fingerprint"])
    return 0
