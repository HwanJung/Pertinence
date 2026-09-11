"""Run settings and platform-owned resource limits."""

from dataclasses import dataclass

from .common import Manifest, fields, integer, name, number, version


@dataclass(frozen=True)
class PlatformProfile:
    max_experts: int = 32
    max_classes: int = 10000
    max_samples: int = 1_000_000
    max_feature_dim: int = 65536
    max_cache_bytes: int = 32 * 1024**3
    max_bundle_bytes: int = 100 * 1024**3
    max_model_bytes: int = 2 * 1024**3
    max_dispatcher_state_bytes: int = 2 * 1024**3
    max_evaluations: int = 10000
    max_training_sample_visits: int = 10**12
    max_image_pixels: int = 20_000_000
    max_sample_id_bytes: int = 512
    inference_batch_size: int = 32
    device: str = "cpu"
    image_digest: str = "local"

    def __post_init__(self) -> None:
        for key, value in vars(self).items():
            if key not in {"device", "image_digest"}:
                integer(value, key)
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("platform device must be cpu or cuda")
        if self.image_digest != "local":
            from .common import digest

            digest(self.image_digest.removeprefix("sha256:"))


class RunConfig(Manifest):
    @staticmethod
    def validate(value: dict) -> None:
        fields(
            value, {"schema_version", "run_name", "seed", "routing", "split", "dispatcher", "nsga2"}
        )
        version(value)
        name(value["run_name"], "run_name")
        split = value["split"]
        fields(split, {"search_size", "final_size", "seed"})
        for seed in (value["seed"], split["seed"]):
            if integer(seed, "seed", 0) > 2**32 - 1:
                raise ValueError("seed must fit uint32")
        routing = value["routing"]
        fields(
            routing,
            {
                "expert_ids",
                "feature_extractor_id",
                "fallback_expert_id",
                "cost_key",
                "require_cost_sorted",
            },
            {"router_cost"},
        )
        ids = routing["expert_ids"]
        if not isinstance(ids, list) or not ids:
            raise ValueError("expert_ids must be a non-empty list")
        for item in ids:
            name(item, "expert id")
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate routing expert IDs")
        for key in ("feature_extractor_id", "fallback_expert_id"):
            if routing[key] not in ids:
                raise ValueError(f"{key} must belong to expert_ids")
        if type(routing["require_cost_sorted"]) is not bool:
            raise ValueError("require_cost_sorted must be boolean")
        name(routing["cost_key"], "cost key/unit")
        if "router_cost" in routing:
            number(routing["router_cost"], "router_cost")
        elif routing["cost_key"] != "mflops":
            raise ValueError("non-mflops objectives require router_cost in the same unit")
        integer(split["search_size"], "search_size")
        integer(split["final_size"], "final_size")
        training = value["dispatcher"]
        fields(
            training,
            {
                "epochs",
                "batch_size",
                "optimizer",
                "learning_rate",
                "weight_decay",
                "ens_beta",
                "normalize_class_weights",
            },
        )
        for key in ("epochs", "batch_size"):
            integer(training[key], key)
        number(training["learning_rate"], "learning_rate", 1e-15)
        number(training["weight_decay"], "weight_decay")
        if number(training["ens_beta"], "ens_beta") >= 1:
            raise ValueError("ens_beta must be < 1")
        if training["optimizer"] not in {"adam", "sgd"}:
            raise ValueError("unsupported optimizer")
        if training["normalize_class_weights"] != "mean_one":
            raise ValueError("normalize_class_weights must be mean_one")
        ga = value["nsga2"]
        fields(
            ga,
            {
                "population_size",
                "generations",
                "crossover",
                "crossover_eta",
                "crossover_probability",
                "mutation",
                "mutation_eta",
                "mutation_probability",
                "penalty_min",
                "penalty_max",
                "weighting_schemes",
            },
        )
        integer(ga["population_size"], "population_size", 2)
        integer(ga["generations"], "generations")
        if ga["crossover"] != "SBX" or ga["mutation"] != "polynomial":
            raise ValueError("only SBX crossover and polynomial mutation are supported")
        for key in ("crossover_eta", "mutation_eta"):
            number(ga[key], key, 1e-12)
        for key in ("crossover_probability", "mutation_probability"):
            if ga[key] is None and key == "mutation_probability":
                continue
            if number(ga[key], key) > 1:
                raise ValueError(f"{key} must be <= 1")
        if number(ga["penalty_max"], "penalty_max") <= number(ga["penalty_min"], "penalty_min"):
            raise ValueError("penalty_max must exceed penalty_min")
        schemes = ga["weighting_schemes"]
        if (
            not isinstance(schemes, list)
            or not schemes
            or any(s not in ("INS", "ISNS", "ENS") for s in schemes)
            or len(set(schemes)) != len(schemes)
        ):
            raise ValueError("weighting_schemes must be a unique subset of INS, ISNS, ENS")


class NormalizedRunContract(Manifest):
    @staticmethod
    def validate(value: dict) -> None:
        from ..reproducibility import canonical_json_hash
        from .dataset import DatasetBundle
        from .experts import ExpertBundle

        fields(
            value,
            {
                "schema_version",
                "dataset",
                "experts",
                "config",
                "inputs",
                "dimensions",
                "cost",
                "splits",
                "lineage",
                "platform",
                "fingerprint",
            },
        )
        version(value)
        DatasetBundle.from_dict(value["dataset"])
        ExpertBundle.from_dict(value["experts"])
        RunConfig.from_dict(value["config"])
        body = {key: item for key, item in value.items() if key != "fingerprint"}
        if value["fingerprint"] != canonical_json_hash(body):
            raise ValueError("normalized contract fingerprint mismatch")
