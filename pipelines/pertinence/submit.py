"""Submit the compiled CPU pipeline to an OSS KFP API running on GKE."""

from __future__ import annotations

import argparse
import re

from .pipeline import _gcs_uri


def _sha256(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://127.0.0.1:8888")
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--pipeline-root", required=True, type=_gcs_uri)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--experiment-name", default="pertinence-cifar10")
    parser.add_argument("--run-name", default="pertinence-cifar10-cpu")
    for name in (
        "cifar10",
        "shufflenet-checkpoint",
        "mobilenet-checkpoint",
        "resnet56-checkpoint",
        "run-config",
    ):
        parser.add_argument(f"--{name}-uri", required=True, type=_gcs_uri)
        parser.add_argument(f"--{name}-sha256", required=True, type=_sha256)
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--enable-cache", action="store_true", dest="enable_cache")
    cache.add_argument("--disable-cache", action="store_false", dest="enable_cache")
    parser.set_defaults(enable_cache=False)
    args = parser.parse_args()

    from kfp import Client

    result = Client(host=args.host, namespace=args.namespace).create_run_from_pipeline_package(
        pipeline_file=args.pipeline,
        arguments={
            "cifar10_uri": args.cifar10_uri,
            "cifar10_sha256": args.cifar10_sha256,
            "shufflenet_checkpoint_uri": args.shufflenet_checkpoint_uri,
            "shufflenet_checkpoint_sha256": args.shufflenet_checkpoint_sha256,
            "mobilenet_checkpoint_uri": args.mobilenet_checkpoint_uri,
            "mobilenet_checkpoint_sha256": args.mobilenet_checkpoint_sha256,
            "resnet56_checkpoint_uri": args.resnet56_checkpoint_uri,
            "resnet56_checkpoint_sha256": args.resnet56_checkpoint_sha256,
            "run_config_uri": args.run_config_uri,
            "run_config_sha256": args.run_config_sha256,
        },
        run_name=args.run_name,
        experiment_name=args.experiment_name,
        namespace=args.namespace,
        pipeline_root=args.pipeline_root,
        enable_caching=args.enable_cache,
        service_account=args.service_account,
    )
    print(result.run_id)


if __name__ == "__main__":
    main()
