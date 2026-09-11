"""Run individual portable components against explicit local artifact paths."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .bundle_data import package_dataset, package_experts
from .contracts import PlatformProfile
from .pipeline_artifacts import artifact_sha256, local_path
from .pipeline_cache import build_routing_dataset
from .pipeline_registry import SQLiteRegistry, register_pareto_dispatchers
from .pipeline_search import evaluate_final, train_dispatcher
from .pipeline_validation import validate_run_contract


def add_commands(subparsers) -> None:
    for command in ("package-dataset", "package-experts"):
        parser = subparsers.add_parser(command)
        parser.add_argument("--source", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.set_defaults(component_action=command)
    parser = subparsers.add_parser("artifact-sha256")
    parser.add_argument("path", type=Path)
    parser.set_defaults(component_action="artifact-sha256")
    parser = subparsers.add_parser("validate-run", aliases=["validate-run-contract"])
    for bundle in ("dataset-bundle", "expert-bundle", "run-config"):
        parser.add_argument(f"--{bundle}-uri", required=True)
        parser.add_argument(f"--{bundle}-sha256", required=True)
    for output in ("normalized-contract", "split-manifest", "validation-report"):
        parser.add_argument(f"--{output}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--image-digest", default="local")
    parser.set_defaults(component_action="validate-run")
    specifications = {
        "build-routing-dataset": (
            "dataset-bundle",
            "expert-bundle",
            "normalized-contract",
            "split-manifest",
            "train-cache",
            "search-cache",
            "final-cache",
        ),
        "train-dispatcher": (
            "train-cache",
            "search-cache",
            "normalized-contract",
            "search-result",
            "pareto-bundle",
        ),
        "evaluate-final": (
            "final-cache",
            "search-result",
            "pareto-bundle",
            "normalized-contract",
            "final-report",
            "evaluated-bundle",
        ),
        "register-pareto-dispatchers": ("evaluated-bundle", "registration-report", "registry-db"),
    }
    for command, arguments in specifications.items():
        parser = subparsers.add_parser(command)
        for argument in arguments:
            parser.add_argument(f"--{argument}", type=Path, required=True)
        parser.set_defaults(component_action=command)


def run_command(args: argparse.Namespace) -> int:
    values = vars(args).copy()
    action = values.pop("component_action")
    values.pop("command", None)
    if action in {"package-dataset", "package-experts"}:
        function = package_dataset if action == "package-dataset" else package_experts
        print(function(local_path(values["source"]), values["output"]))
    elif action == "artifact-sha256":
        print(artifact_sha256(local_path(values["path"])))
    elif action == "validate-run":
        profile = replace(
            PlatformProfile(), device=values.pop("device"), image_digest=values.pop("image_digest")
        )
        validate_run_contract(**values, profile=profile)
    elif action == "build-routing-dataset":
        build_routing_dataset(**values)
    elif action == "train-dispatcher":
        train_dispatcher(**values)
    elif action == "evaluate-final":
        evaluate_final(**values)
    elif action == "register-pareto-dispatchers":
        registry = SQLiteRegistry(values.pop("registry_db"))
        register_pareto_dispatchers(**values, registry=registry)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_commands(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    try:
        return run_command(args)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        raise SystemExit(f"{args.command} failed: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
