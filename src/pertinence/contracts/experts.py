"""Portable ONNX expert metadata and preprocessing v1 allow-list."""

from .common import Manifest, digest, fields, integer, name, number, version


class ExpertBundle(Manifest):
    @staticmethod
    def validate(value: dict) -> None:
        fields(value, {"schema_version", "bundle_id", "experts"})
        version(value)
        name(value["bundle_id"], "bundle_id")
        if not isinstance(value["experts"], list) or not value["experts"]:
            raise ValueError("experts must be a non-empty list")
        ids = []
        for expert in value["experts"]:
            fields(
                expert,
                {
                    "id",
                    "version",
                    "format",
                    "model_path",
                    "model_sha256",
                    "input",
                    "outputs",
                    "num_classes",
                    "costs",
                },
                {"class_names"},
            )
            ids.append(name(expert["id"], "expert id"))
            name(expert["version"], "expert version")
            if expert["format"] != "onnx":
                raise ValueError("unsupported runtime: only onnx is allowed")
            digest(expert["model_sha256"])
            path = expert["model_path"]
            if (
                not isinstance(path, str)
                or not path.endswith(".onnx")
                or path.startswith("/")
                or ".." in path.split("/")
                or "\\" in path
                or ":" in path
            ):
                raise ValueError("unsafe model_path")
            integer(expert["num_classes"], "expert num_classes")
            inputs = expert["input"]
            fields(inputs, {"layout", "dtype", "shape", "preprocessing"})
            if inputs["layout"] != "NCHW" or inputs["dtype"] != "float32":
                raise ValueError("v1 supports only NCHW float32 RGB inputs")
            shape = inputs["shape"]
            if not isinstance(shape, list) or len(shape) != 3 or shape[0] != 3:
                raise ValueError("input shape must be [3, height, width]")
            for dim in shape:
                integer(dim, "input shape dimension")
            preprocessing = inputs["preprocessing"]
            fields(preprocessing, {"resize", "mean", "std"})
            if preprocessing["resize"] != shape[1:]:
                raise ValueError("resize must match input [height, width]")
            for key in ("mean", "std"):
                values = preprocessing[key]
                if not isinstance(values, list) or len(values) != 3:
                    raise ValueError(f"preprocessing {key} must contain three values")
                for item in values:
                    number(item, key, minimum=-1e6 if key == "mean" else 1e-12)
            fields(expert["outputs"], {"logits"}, {"features"})
            for output in expert["outputs"].values():
                if not isinstance(output, str) or not output:
                    raise ValueError("output names must be non-empty strings")
            costs = expert["costs"]
            if not isinstance(costs, dict) or not costs:
                raise ValueError("expert costs must be a non-empty mapping")
            for key, cost in costs.items():
                name(key, "cost key/unit")
                number(cost, f"cost {key}")
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate expert IDs")
