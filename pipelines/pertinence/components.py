"""KFP v2 CPU container interfaces; creating them never contacts a cluster."""

import re

from kfp import dsl
from kfp.dsl import Artifact, Input, Output, OutputPath


def _validate_component_settings(image: str, cpu_threads: int) -> None:
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("component image must be pinned by @sha256 digest")
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer")


def make_cifar10_bootstrap_components(image: str):
    """Create the two legacy adapters from pinned CIFAR-10 raw assets."""

    _validate_component_settings(image, 1)
    command = ["python", "-m", "pertinence.pipeline_cli"]

    @dsl.container_component
    def prepare_dataset_bundle(
        cifar10_archive: Input[Artifact],
        cifar10_sha256: str,
        dataset_bundle: Output[Artifact],
        dataset_bundle_sha256: OutputPath(str),
    ):
        return dsl.ContainerSpec(
            image=image,
            command=command,
            args=[
                "prepare-dataset-bundle",
                "--cifar10-archive",
                cifar10_archive.path,
                "--cifar10-sha256",
                cifar10_sha256,
                "--dataset-bundle",
                dataset_bundle.path,
                "--bundle-sha256-output",
                dataset_bundle_sha256,
            ],
        )

    @dsl.container_component
    def export_expert_bundle(
        shufflenet_checkpoint: Input[Artifact],
        shufflenet_sha256: str,
        mobilenet_checkpoint: Input[Artifact],
        mobilenet_sha256: str,
        resnet56_checkpoint: Input[Artifact],
        resnet56_sha256: str,
        expert_bundle: Output[Artifact],
        expert_bundle_sha256: OutputPath(str),
    ):
        return dsl.ContainerSpec(
            image=image,
            command=command,
            args=[
                "export-expert-bundle",
                "--shufflenet-checkpoint",
                shufflenet_checkpoint.path,
                "--shufflenet-sha256",
                shufflenet_sha256,
                "--mobilenet-checkpoint",
                mobilenet_checkpoint.path,
                "--mobilenet-sha256",
                mobilenet_sha256,
                "--resnet56-checkpoint",
                resnet56_checkpoint.path,
                "--resnet56-sha256",
                resnet56_sha256,
                "--expert-bundle",
                expert_bundle.path,
                "--bundle-sha256-output",
                expert_bundle_sha256,
            ],
        )

    return {
        "prepare": prepare_dataset_bundle,
        "export": export_expert_bundle,
    }


def make_training_components(image: str, *, cpu_threads: int = 4):
    """Create the five bundle-driven components used by the generic pipeline."""

    _validate_component_settings(image, cpu_threads)
    command = ["python", "-m", "pertinence.pipeline_cli"]

    @dsl.container_component
    def validate_run(
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
                "validate-run",
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
                "cpu",
                "--onnx-intra-op-threads",
                str(cpu_threads),
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
    def register_dispatchers(
        evaluated_bundle: Input[Artifact],
        registry_bundle: Output[Artifact],
        registration_report: Output[Artifact],
    ):
        return dsl.ContainerSpec(
            image=image,
            command=command,
            args=[
                "register-dispatcher-bundle",
                "--evaluated-bundle",
                evaluated_bundle.path,
                "--registry-bundle",
                registry_bundle.path,
                "--registration-report",
                registration_report.path,
            ],
        )

    return {
        "validate": validate_run,
        "build": build_routing_dataset,
        "train": train_dispatcher,
        "evaluate": evaluate_final,
        "register": register_dispatchers,
    }


def make_components(image: str, *, cpu_threads: int = 4):
    """Return the legacy seven-component set for callers that still need it."""

    return {
        **make_cifar10_bootstrap_components(image),
        **make_training_components(image, cpu_threads=cpu_threads),
    }
