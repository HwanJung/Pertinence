"""DatasetBundle v1 schema."""

from .common import Manifest, digest, fields, integer, name, version


class DatasetBundle(Manifest):
    @staticmethod
    def validate(value: dict) -> None:
        fields(
            value,
            {
                "schema_version",
                "task",
                "dataset_id",
                "dataset_version",
                "num_classes",
                "class_names",
                "image_codec",
                "train_glob",
                "evaluation_glob",
                "train_samples",
                "evaluation_samples",
                "content_sha256",
            },
        )
        version(value)
        if value["task"] != "image_classification":
            raise ValueError("only image_classification is supported")
        name(value["dataset_id"], "dataset_id")
        name(value["dataset_version"], "dataset_version")
        classes = integer(value["num_classes"], "num_classes")
        names = value["class_names"]
        if (
            not isinstance(names, list)
            or len(names) != classes
            or any(not isinstance(item, str) or not item for item in names)
            or len(set(names)) != classes
        ):
            raise ValueError("class_names must contain num_classes unique strings")
        if value["image_codec"] not in ("jpeg", "png"):
            raise ValueError("image_codec must be jpeg or png")
        for key in ("train_samples", "evaluation_samples"):
            integer(value[key], key)
        for key in ("train_glob", "evaluation_glob"):
            pattern = value[key]
            if (
                not isinstance(pattern, str)
                or not pattern.endswith(".parquet")
                or pattern.startswith("/")
                or ".." in pattern.split("/")
                or "\\" in pattern
                or ":" in pattern
            ):
                raise ValueError(f"unsafe dataset shard glob: {key}")
        digest(value["content_sha256"])
