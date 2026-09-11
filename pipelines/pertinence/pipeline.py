"""Compile the fixed KFP DAG locally; submission and platform provisioning are separate."""

import argparse

from kfp import compiler, dsl

from .components import make_components


def make_pipeline(image: str, *, device: str = "cpu", registry_db: str = "/registry/registry.db"):
    components = make_components(image, device=device, registry_db=registry_db)

    @dsl.pipeline(name="pertinence-portable-dispatchers")
    def pipeline(
        dataset_bundle_uri: str,
        dataset_bundle_sha256: str,
        expert_bundle_uri: str,
        expert_bundle_sha256: str,
        run_config_uri: str,
        run_config_sha256: str,
    ):
        dataset = dsl.importer(artifact_uri=dataset_bundle_uri, artifact_class=dsl.Artifact)
        experts = dsl.importer(artifact_uri=expert_bundle_uri, artifact_class=dsl.Artifact)
        config = dsl.importer(artifact_uri=run_config_uri, artifact_class=dsl.Artifact)
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
        register = components["register"](evaluated_bundle=evaluate.outputs["evaluated_bundle"])
        for task in (validate, build, train, evaluate):
            task.set_caching_options(True)
            if device == "cuda":
                task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
        for task in (validate, build, evaluate, register):
            task.set_retry(1)
        train.set_retry(0)
        register.set_caching_options(False)

    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--registry-db", default="/registry/registry.db")
    args = parser.parse_args()
    compiler.Compiler().compile(
        make_pipeline(args.image, device=args.device, registry_db=args.registry_db), args.output
    )


if __name__ == "__main__":
    main()
