"""Compile the generic CPU-only, bundle-driven KFP training DAG."""

import argparse
from pathlib import Path
from urllib.parse import urlparse

from kfp import compiler, dsl, kubernetes

from .components import make_training_components


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
        .set_env_variable("OMP_NUM_THREADS", cpu)
        .set_env_variable("MKL_NUM_THREADS", cpu)
    )


def _schedule_on_pipeline_pool(task):
    kubernetes.add_node_selector(
        task,
        label_key="cloud.google.com/gke-nodepool",
        label_value="pipeline-pool",
    )
    kubernetes.add_toleration(
        task,
        key="workload",
        operator="Equal",
        value="pertinence-pipeline",
        effect="NoSchedule",
    )
    return task


def make_pipeline(
    image: str,
    *,
    pipeline_root: str,
    cpu_threads: int = 4,
    train_timeout_seconds: int = 259_200,
    enable_cache: bool = False,
):
    pipeline_root = _gcs_uri(pipeline_root)
    components = make_training_components(image, cpu_threads=cpu_threads)

    @dsl.pipeline(name="pertinence-routing-training-cpu", pipeline_root=pipeline_root)
    def pipeline(
        dataset_bundle_uri: str,
        dataset_bundle_sha256: str,
        expert_bundle_uri: str,
        expert_bundle_sha256: str,
        run_config_uri: str,
        run_config_sha256: str,
    ):
        dataset = dsl.importer(
            artifact_uri=dataset_bundle_uri,
            artifact_class=dsl.Artifact,
            reimport=False,
        )
        experts = dsl.importer(
            artifact_uri=expert_bundle_uri,
            artifact_class=dsl.Artifact,
            reimport=False,
        )
        config = dsl.importer(
            artifact_uri=run_config_uri,
            artifact_class=dsl.Artifact,
            reimport=False,
        )
        validate = components["validate"](
            dataset_bundle=dataset.output,
            expert_bundle=experts.output,
            run_config=config.output,
            dataset_sha256=dataset_bundle_sha256,
            expert_sha256=expert_bundle_sha256,
            config_sha256=run_config_sha256,
        )
        build = components["build"](
            dataset_bundle=dataset.output,
            expert_bundle=experts.output,
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

        for task in (validate, build, train, evaluate, register):
            _schedule_on_pipeline_pool(task)

        _resources(validate, cpu=str(cpu_threads), memory="6Gi")
        _resources(build, cpu=str(cpu_threads), memory="12Gi")
        _resources(train, cpu=str(cpu_threads), memory="8Gi")
        kubernetes.set_timeout(train, seconds=train_timeout_seconds)
        _resources(evaluate, cpu="2", memory="4Gi")
        _resources(register, cpu="1", memory="2Gi")
        for task in (validate, build, train, evaluate):
            task.set_caching_options(enable_cache)
        for task in (validate, build, evaluate, register):
            task.set_retry(1)
        train.set_retry(0)
        register.set_caching_options(False)

    return pipeline


def compile_pipeline(
    *,
    image: str,
    pipeline_root: str,
    output: str | Path,
    cpu_threads: int = 4,
    train_timeout_seconds: int = 259_200,
    enable_cache: bool = False,
) -> None:
    """Compile the versioned pipeline package used for GKE submission."""
    output = Path(output)
    compiler.Compiler().compile(
        make_pipeline(
            image,
            pipeline_root=pipeline_root,
            cpu_threads=cpu_threads,
            train_timeout_seconds=train_timeout_seconds,
            enable_cache=enable_cache,
        ),
        str(output),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--pipeline-root", required=True, type=_gcs_uri)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--train-timeout-seconds", type=int, default=259_200)
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--enable-cache", action="store_true", dest="enable_cache")
    cache.add_argument("--disable-cache", action="store_false", dest="enable_cache")
    parser.set_defaults(enable_cache=False)
    args = parser.parse_args()
    compile_pipeline(
        image=args.image,
        pipeline_root=args.pipeline_root,
        output=args.output,
        cpu_threads=args.cpu_threads,
        train_timeout_seconds=args.train_timeout_seconds,
        enable_cache=args.enable_cache,
    )


if __name__ == "__main__":
    main()
