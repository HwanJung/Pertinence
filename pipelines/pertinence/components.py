"""KFP v2 container interfaces; no cluster access occurs when creating these."""

import re

from kfp import dsl
from kfp.dsl import Artifact, Input, Output


def make_components(image: str, *, device: str = "cpu", registry_db: str = "/registry/registry.db"):
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("component image must be pinned by @sha256 digest")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    command = ["python", "-m", "pertinence.pipeline_cli"]

    @dsl.container_component
    def validate_run_contract(
        dataset_bundle: Input[Artifact],
        expert_bundle: Input[Artifact],
        run_config: Input[Artifact],
        dataset_sha256: str,
        expert_sha256: str,
        config_sha256: str,
        normalized_contract: Output[Artifact],
        split_manifest: Output[Artifact],
        validation_report: Output[Artifact],
    ):
        return dsl.ContainerSpec(
            image=image,
            command=command,
            args=[
                "validate-run-contract",
                "--dataset-bundle-uri",
                dataset_bundle.path,
                "--dataset-bundle-sha256",
                dataset_sha256,
                "--expert-bundle-uri",
                expert_bundle.path,
                "--expert-bundle-sha256",
                expert_sha256,
                "--run-config-uri",
                run_config.path,
                "--run-config-sha256",
                config_sha256,
                "--normalized-contract",
                normalized_contract.path,
                "--split-manifest",
                split_manifest.path,
                "--validation-report",
                validation_report.path,
                "--device",
                device,
                "--image-digest",
                image.split("@", 1)[1],
            ],
        )

    @dsl.container_component
    def build_routing_dataset(
        dataset_bundle: Input[Artifact],
        expert_bundle: Input[Artifact],
        normalized_contract: Input[Artifact],
        split_manifest: Input[Artifact],
        train_cache: Output[Artifact],
        search_cache: Output[Artifact],
        final_cache: Output[Artifact],
    ):
        return dsl.ContainerSpec(
            image=image,
            command=command,
            args=[
                "build-routing-dataset",
                "--dataset-bundle",
                dataset_bundle.path,
                "--expert-bundle",
                expert_bundle.path,
                "--normalized-contract",
                normalized_contract.path,
                "--split-manifest",
                split_manifest.path,
                "--train-cache",
                train_cache.path,
                "--search-cache",
                search_cache.path,
                "--final-cache",
                final_cache.path,
            ],
        )

    @dsl.container_component
    def train_dispatcher(
        train_cache: Input[Artifact],
        search_cache: Input[Artifact],
        normalized_contract: Input[Artifact],
        search_result: Output[Artifact],
        pareto_bundle: Output[Artifact],
    ):
        return dsl.ContainerSpec(
            image=image,
            command=command,
            args=[
                "train-dispatcher",
                "--train-cache",
                train_cache.path,
                "--search-cache",
                search_cache.path,
                "--normalized-contract",
                normalized_contract.path,
                "--search-result",
                search_result.path,
                "--pareto-bundle",
                pareto_bundle.path,
            ],
        )

    @dsl.container_component
    def evaluate_final(
        final_cache: Input[Artifact],
        search_result: Input[Artifact],
        pareto_bundle: Input[Artifact],
        normalized_contract: Input[Artifact],
        final_report: Output[Artifact],
        evaluated_bundle: Output[Artifact],
    ):
        return dsl.ContainerSpec(
            image=image,
            command=command,
            args=[
                "evaluate-final",
                "--final-cache",
                final_cache.path,
                "--search-result",
                search_result.path,
                "--pareto-bundle",
                pareto_bundle.path,
                "--normalized-contract",
                normalized_contract.path,
                "--final-report",
                final_report.path,
                "--evaluated-bundle",
                evaluated_bundle.path,
            ],
        )

    @dsl.container_component
    def register_pareto_dispatchers(
        evaluated_bundle: Input[Artifact],
        registration_report: Output[Artifact],
    ):
        return dsl.ContainerSpec(
            image=image,
            command=command,
            args=[
                "register-pareto-dispatchers",
                "--evaluated-bundle",
                evaluated_bundle.path,
                "--registration-report",
                registration_report.path,
                "--registry-db",
                registry_db,
            ],
        )

    return {
        "validate": validate_run_contract,
        "build": build_routing_dataset,
        "train": train_dispatcher,
        "evaluate": evaluate_final,
        "register": register_pareto_dispatchers,
    }
