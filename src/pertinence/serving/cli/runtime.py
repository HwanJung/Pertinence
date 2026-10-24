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
    expert.add_argument("--serving-mode", choices=("direct",))
    expert.add_argument("--host", default="0.0.0.0")
    expert.add_argument("--port", type=int, default=8080)
    expert.add_argument("--keep-alive", action="store_true")
    expert.add_argument("--idle-timeout", type=float, default=10.0)
    expert.add_argument("--diagnostics-path", type=Path)
    expert.add_argument("--run-fingerprint")
    expert.set_defaults(serving_action="serve-expert")

    router = commands.add_parser("router")
    router.add_argument("--promotion", type=Path, required=True)
    router.add_argument("--bundle-path", type=Path)
    router.add_argument(
        "--router-mode",
        choices=("dynamic", "fused-dynamic", "hybrid-dynamic", "static-resnet"),
        required=True,
    )
    router.add_argument("--expert-url", action="append", default=[], metavar="NAME=URL")
    router.add_argument("--timeout-seconds", type=float, default=10.0)
    router.add_argument("--host", default="0.0.0.0")
    router.add_argument("--port", type=int, default=8080)
    router.add_argument("--expert-transport", choices=("json", "binary"), default="json")
    router.add_argument("--diagnostics-path", type=Path)
    router.add_argument("--run-fingerprint")
    router.set_defaults(serving_action="serve-router")


def _load_dispatcher(path: Path):
    from ..runtime import load_dispatcher_npz

    return load_dispatcher_npz(path)


def run_command(args: argparse.Namespace) -> int:
    from ..http import ExpertModel, V2Application, serve
    from ..runtime import (
        DynamicRouter,
        FusedDynamicRouter,
        HybridDynamicRouter,
        HttpExpertClient,
        OnnxExpertRunner,
        StaticRouter,
    )

    from ..diagnostics import Diagnostics

    if args.run_fingerprint is not None:
        from ...contracts.common import digest
        digest(args.run_fingerprint)
    diagnostics = Diagnostics(args.diagnostics_path)
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
        identity = {"serving_mode": args.serving_mode} if args.serving_mode else None
        model = ExpertModel(
            args.model_name,
            runner,
            feature_dim=args.feature_dim,
            identity=identity,
        )
        serve(V2Application(model, diagnostics=diagnostics, run_fingerprint=args.run_fingerprint), args.host, args.port,
              keep_alive=args.keep_alive, idle_timeout=args.idle_timeout)
        return 0

    promotion = _load_promotion(args.promotion)
    experts = [item["id"] for item in promotion["experts"]]
    endpoints = _key_values(args.expert_url, "expert-url")
    required_experts = {
        "static-resnet": {experts[-1]},
        "dynamic": set(experts),
        "fused-dynamic": set(),
        "hybrid-dynamic": set(experts[1:]),
    }[args.router_mode]
    if set(endpoints) != required_experts:
        raise ValueError("expert URLs must exactly match the router's required promoted experts")
    if args.router_mode == "hybrid-dynamic" and args.expert_transport != "binary":
        raise ValueError("hybrid-dynamic requires --expert-transport=binary")
    clients = {
        name: HttpExpertClient(
            endpoint,
            name,
            timeout_seconds=args.timeout_seconds,
            feature_dim=1024 if name == experts[0] else None,
            transport=args.expert_transport,
            diagnostics=diagnostics,
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
        evaluated_path = (
            args.bundle_path / "evaluated"
            if args.router_mode in {"fused-dynamic", "hybrid-dynamic"}
            else args.bundle_path
        )
        verified = _wait_for_modelcar(
            lambda: validate_promotion_bundle(promotion, evaluated_path)
        )
        checkpoint = evaluated_path / verified["checkpoint"]
        expected = promotion["dispatcher"]["checkpoint_sha256"]
        if sha256_file(checkpoint) != expected:
            raise ValueError("dispatcher checkpoint SHA-256 differs from promotion")
        dispatcher = _load_dispatcher(checkpoint)
        if args.router_mode in {"fused-dynamic", "hybrid-dynamic"}:
            promoted = {item["id"]: item for item in promotion["experts"]}
            runners = {
                name: OnnxExpertRunner(
                    _wait_for_modelcar(
                        lambda name=name: _verified_model_path(
                            args.bundle_path / "experts" / name / "model.onnx",
                            promoted[name]["model_sha256"],
                        )
                    ),
                    promoted[name]["model_sha256"],
                    features_name="features" if name == experts[0] else None,
                    feature_dim=1024 if name == experts[0] else None,
                )
                for name in (experts if args.router_mode == "fused-dynamic" else experts[:1])
            }
            router_type = (FusedDynamicRouter if args.router_mode == "fused-dynamic"
                           else HybridDynamicRouter)
            model = router_type(
                "dispatcher-000733",
                promotion["promotion_id"],
                promotion["dispatcher"]["id"],
                experts,
                experts[0],
                {**runners, **clients},
                dispatcher,
                feature_dim=1024,
                diagnostics=diagnostics,
            )
        else:
            model = DynamicRouter(
                "dispatcher-000733",
                promotion["promotion_id"],
                promotion["dispatcher"]["id"],
                experts,
                experts[0],
                clients,
                dispatcher,
                feature_dim=1024,
                diagnostics=diagnostics,
                serving_mode="distributed",
            )
    serve(V2Application(model, router=True, diagnostics=diagnostics,
                        run_fingerprint=args.run_fingerprint), args.host, args.port)
    return 0
