"""KServe runtime command registration and execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from ...reproducibility import sha256_file
from .common import _key_values, _load_promotion, _verified_model_path, _wait_for_modelcar


def add_commands(subparsers: argparse._SubParsersAction) -> None:
    serve_parser = subparsers.add_parser("serve", help="run the single-worker KServe V2 runtime")
    commands = serve_parser.add_subparsers(dest="serve_command", required=True)
    expert = commands.add_parser("expert")
    expert.add_argument("--model-name", required=True)
    expert.add_argument("--model-path", type=Path, required=True)
    expert.add_argument("--model-sha256", required=True)
    expert.add_argument("--features-name")
    expert.add_argument("--feature-dim", type=int)
    expert.add_argument("--host", default="0.0.0.0")
    expert.add_argument("--port", type=int, default=8080)
    expert.set_defaults(serving_action="serve-expert")

    router = commands.add_parser("router")
    router.add_argument("--promotion", type=Path, required=True)
    router.add_argument("--bundle-path", type=Path)
    router.add_argument("--router-mode", choices=("dynamic", "static-resnet"), required=True)
    router.add_argument("--expert-url", action="append", default=[], metavar="NAME=URL")
    router.add_argument("--timeout-seconds", type=float, default=10.0)
    router.add_argument("--host", default="0.0.0.0")
    router.add_argument("--port", type=int, default=8080)
    router.set_defaults(serving_action="serve-router")


def _load_dispatcher(path: Path):
    from ..runtime import load_dispatcher_npz

    return load_dispatcher_npz(path)


def run_command(args: argparse.Namespace) -> int:
    from ..http import ExpertModel, V2Application, serve
    from ..runtime import DynamicRouter, HttpExpertClient, OnnxExpertRunner, StaticRouter

    if args.serving_action == "serve-expert":
        model_path = _wait_for_modelcar(
            lambda: _verified_model_path(args.model_path, args.model_sha256)
        )
        runner = OnnxExpertRunner(
            model_path,
            args.model_sha256,
            features_name=args.features_name,
            feature_dim=args.feature_dim,
        )
        model = ExpertModel(args.model_name, runner, feature_dim=args.feature_dim)
        serve(V2Application(model), args.host, args.port)
        return 0

    promotion = _load_promotion(args.promotion)
    experts = [item["id"] for item in promotion["experts"]]
    endpoints = _key_values(args.expert_url, "expert-url")
    required_experts = {experts[-1]} if args.router_mode == "static-resnet" else set(experts)
    if set(endpoints) != required_experts:
        raise ValueError("expert URLs must exactly match the router's required promoted experts")
    clients = {
        name: HttpExpertClient(
            endpoint,
            name,
            timeout_seconds=args.timeout_seconds,
            feature_dim=1024 if name == experts[0] else None,
        )
        for name, endpoint in ((name, endpoints[name]) for name in experts if name in endpoints)
    }
    if args.router_mode == "static-resnet":
        model = StaticRouter(
            "static-resnet-router",
            promotion["promotion_id"],
            experts[-1],
            len(experts) - 1,
            clients[experts[-1]],
        )
    else:
        from ..promotion import validate_promotion_bundle

        if args.bundle_path is None:
            raise ValueError("dynamic router requires --bundle-path")
        verified = _wait_for_modelcar(
            lambda: validate_promotion_bundle(promotion, args.bundle_path)
        )
        checkpoint = args.bundle_path / verified["checkpoint"]
        expected = promotion["dispatcher"]["checkpoint_sha256"]
        if sha256_file(checkpoint) != expected:
            raise ValueError("dispatcher checkpoint SHA-256 differs from promotion")
        dispatcher = _load_dispatcher(checkpoint)
        model = DynamicRouter(
            "dispatcher-000733",
            promotion["promotion_id"],
            promotion["dispatcher"]["id"],
            experts,
            experts[0],
            clients,
            dispatcher,
            feature_dim=1024,
        )
    serve(V2Application(model, router=True), args.host, args.port)
    return 0
