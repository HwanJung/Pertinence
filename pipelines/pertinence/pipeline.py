"""Compile the CPU-only GCS-backed KFP DAG; cloud provisioning remains separate."""

import argparse
from urllib.parse import urlparse

from kfp import compiler, dsl

from .components import make_components


def _gcs_uri(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "gs"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or any(character.isspace() for character in value)
    ):
        raise ValueError("pipeline root must use a valid gs:// URI")
    return value.rstrip("/")


def _resources(task, *, cpu: str, memory: str):
    return (
        task.set_cpu_request(cpu)
        .set_cpu_limit(cpu)
        .set_memory_request(memory)
        .set_memory_limit(memory)
    )


def make_pipeline(
    image: str,
    *,
    pipeline_root: str,
    cpu_threads: int = 4,
    enable_cache: bool = False,
):
    pipeline_root = _gcs_uri(pipeline_root)
    components = make_components(image, cpu_threads=cpu_threads)

    @dsl.pipeline(name="pertinence-cifar10-cpu", pipeline_root=pipeline_root)
    def pipeline(
        cifar10_uri: str,
        cifar10_sha256: str,
        shufflenet_checkpoint_uri: str,
        shufflenet_checkpoint_sha256: str,
        mobilenet_checkpoint_uri: str,
        mobilenet_checkpoint_sha256: str,
        resnet56_checkpoint_uri: str,
        resnet56_checkpoint_sha256: str,
        run_config_uri: str,
        run_config_sha256: str,
    ):
        cifar10 = dsl.importer(
            artifact_uri=cifar10_uri,
            artifact_class=dsl.Artifact,
            reimport=False,
        )
        shufflenet = dsl.importer(
            artifact_uri=shufflenet_checkpoint_uri,
            artifact_class=dsl.Model,
            reimport=False,
        )
        mobilenet = dsl.importer(
            artifact_uri=mobilenet_checkpoint_uri,
            artifact_class=dsl.Model,
            reimport=False,
        )
        resnet56 = dsl.importer(
            artifact_uri=resnet56_checkpoint_uri,
            artifact_class=dsl.Model,
            reimport=False,
        )
        config = dsl.importer(
            artifact_uri=run_config_uri,
            artifact_class=dsl.Artifact,
            reimport=False,
        )
        prepare = components["prepare"](
            cifar10_archive=cifar10.output,
            cifar10_sha256=cifar10_sha256,
        )
        export = components["export"](
            shufflenet_checkpoint=shufflenet.output,
            shufflenet_sha256=shufflenet_checkpoint_sha256,
            mobilenet_checkpoint=mobilenet.output,
            mobilenet_sha256=mobilenet_checkpoint_sha256,
            resnet56_checkpoint=resnet56.output,
            resnet56_sha256=resnet56_checkpoint_sha256,
        )
        validate = components["validate"](
            dataset_bundle=prepare.outputs["dataset_bundle"],
            expert_bundle=export.outputs["expert_bundle"],
            run_config=config.output,
            dataset_sha256=prepare.outputs["dataset_bundle_sha256"],
            expert_sha256=export.outputs["expert_bundle_sha256"],
            config_sha256=run_config_sha256,
        )
        build = components["build"](
            dataset_bundle=prepare.outputs["dataset_bundle"],
            expert_bundle=export.outputs["expert_bundle"],
            normalized_contract=validate.outputs["normalized_contract"],
            split_manifest=validate.outputs["split_manifest"],
        )
        train = components["train"](
            train_cache=build.outputs["train_cache"],
            search_cache=build.outputs["search_cache"],
            normalized_contract=validate.outputs["normalized_contract"],
        )
        evaluate = components["evaluate"](
            final_cache=build.outputs["final_cache"],
            search_result=train.outputs["search_result"],
            pareto_bundle=train.outputs["pareto_bundle"],
            normalized_contract=validate.outputs["normalized_contract"],
        )
        register = components["register"](
            evaluated_bundle=evaluate.outputs["evaluated_bundle"]
        )

        _resources(prepare, cpu="2", memory="4Gi")
        _resources(export, cpu=str(cpu_threads), memory="8Gi")
        _resources(validate, cpu=str(cpu_threads), memory="6Gi")
        _resources(build, cpu=str(cpu_threads), memory="12Gi")
        _resources(train, cpu=str(cpu_threads), memory="8Gi")
        _resources(evaluate, cpu="2", memory="4Gi")
        _resources(register, cpu="1", memory="2Gi")
        for task in (prepare, export, validate, build, train, evaluate):
            task.set_caching_options(enable_cache)
        for task in (prepare, export, validate, build, evaluate, register):
            task.set_retry(1)
        train.set_retry(0)
        register.set_caching_options(False)

    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--pipeline-root", required=True, type=_gcs_uri)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--enable-cache", action="store_true", dest="enable_cache")
    cache.add_argument("--disable-cache", action="store_false", dest="enable_cache")
    parser.set_defaults(enable_cache=False)
    args = parser.parse_args()
    compiler.Compiler().compile(
        make_pipeline(
            args.image,
            pipeline_root=args.pipeline_root,
            cpu_threads=args.cpu_threads,
            enable_cache=args.enable_cache,
        ),
        args.output,
    )


if __name__ == "__main__":
    main()
