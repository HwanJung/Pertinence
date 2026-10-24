"""Submit the generic bundle-driven CPU pipeline to an OSS KFP API on GKE."""

from __future__ import annotations

import argparse
import re
from urllib.parse import urlparse

from .pipeline import _gcs_uri


def _sha256(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def _gcs_artifact_uri(value: str) -> str:
    uri = _gcs_uri(value)
    if not urlparse(uri).path.strip("/"):
        raise argparse.ArgumentTypeError("artifact URI must include an object or prefix path")
    return uri


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://127.0.0.1:8888")
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--pipeline-root", required=True, type=_gcs_uri)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--experiment-name", default="pertinence-routing-training")
    parser.add_argument("--run-name", default="pertinence-routing-training-cpu")
    for name in ("dataset-bundle", "expert-bundle", "run-config"):
        parser.add_argument(f"--{name}-uri", required=True, type=_gcs_artifact_uri)
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
            "dataset_bundle_uri": args.dataset_bundle_uri,
            "dataset_bundle_sha256": args.dataset_bundle_sha256,
            "expert_bundle_uri": args.expert_bundle_uri,
            "expert_bundle_sha256": args.expert_bundle_sha256,
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
