"""Streaming Parquet DatasetBundle reader and local bundle packaging."""

from __future__ import annotations

import io
from pathlib import Path
import shutil
from typing import Iterator

from PIL import Image

from .contracts import DatasetBundle, ExpertBundle, PlatformProfile
from .contracts.common import parse_manifest
from .pipeline_artifacts import atomic_directory, file_inventory, safe_path, write_json
from .reproducibility import canonical_json_hash, sha256_file


def decode_image(encoded: bytes, codec: str, max_pixels: int) -> Image.Image:
    if not isinstance(encoded, bytes) or not encoded:
        raise ValueError("image must be non-empty binary")
    with Image.open(io.BytesIO(encoded)) as image:
        if image.format.lower() != codec or image.width * image.height > max_pixels:
            raise ValueError("image codec mismatch or image exceeds platform pixel limit")
        image.load()
        return image.convert("RGB")


class DatasetBundleReader:
    def __init__(self, root: Path, manifest: DatasetBundle) -> None:
        self.root = root
        self.manifest = manifest.to_dict()

    def shards(self, pool: str) -> list[Path]:
        if pool not in {"train", "evaluation"}:
            raise ValueError("pool must be train or evaluation")
        paths = sorted(self.root.glob(self.manifest[f"{pool}_glob"]))
        if not paths:
            raise ValueError(f"no {pool} Parquet shards matched")
        for path in paths:
            safe_path(self.root, path.relative_to(self.root).as_posix())
            if not path.is_file():
                raise ValueError("Parquet shard must be a regular file")
        return paths

    def batches(self, pool: str, batch_size: int = 32) -> Iterator[list[dict]]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for path in self.shards(pool):
            with pq.ParquetFile(path) as parquet:
                schema = parquet.schema_arrow
                for key, dtype in (
                    ("sample_id", pa.string()),
                    ("image", pa.binary()),
                    ("target", pa.int64()),
                ):
                    if key not in schema.names or schema.field(key).type != dtype:
                        raise ValueError(f"Parquet field {key} must have type {dtype}")
                for batch in parquet.iter_batches(
                    batch_size=batch_size, columns=["sample_id", "image", "target"]
                ):
                    yield batch.to_pylist()

    def validate_rows(self, profile: PlatformProfile) -> dict[str, list[str]]:
        seen: set[str] = set()
        pools = {}
        for pool in ("train", "evaluation"):
            ids = []
            for batch in self.batches(pool, profile.inference_batch_size):
                for row in batch:
                    sample_id, target = row["sample_id"], row["target"]
                    if (
                        not isinstance(sample_id, str)
                        or not sample_id
                        or len(sample_id.encode()) > profile.max_sample_id_bytes
                    ):
                        raise ValueError(
                            "sample_id must be a non-empty string within platform limit"
                        )
                    if sample_id in seen:
                        raise ValueError(f"duplicate/overlapping sample_id: {sample_id}")
                    if type(target) is not int or not 0 <= target < self.manifest["num_classes"]:
                        raise ValueError(f"target outside dataset class range: {sample_id}")
                    decode_image(
                        row["image"], self.manifest["image_codec"], profile.max_image_pixels
                    )
                    seen.add(sample_id)
                    ids.append(sample_id)
                    if len(seen) > profile.max_samples:
                        raise ValueError("dataset exceeds platform sample limit")
            if len(ids) != self.manifest[f"{pool}_samples"]:
                raise ValueError(f"{pool} actual row count differs from manifest")
            pools[pool] = ids
        return pools


def package_dataset(source: Path, output: Path) -> str:
    """Package existing shards and a dataset.yaml template, filling counts/digest."""
    import pyarrow.parquet as pq

    raw = parse_manifest((source / "dataset.yaml").read_text())
    raw["content_sha256"] = "0" * 64
    raw["train_samples"] = raw["evaluation_samples"] = 1
    reader = DatasetBundleReader(source, DatasetBundle.from_dict(raw))
    with atomic_directory(output) as staging:
        for pool in ("train", "evaluation"):
            paths = reader.shards(pool)
            raw[f"{pool}_samples"] = sum(pq.read_metadata(path).num_rows for path in paths)
            for path in paths:
                target = safe_path(staging, path.relative_to(source).as_posix())
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        raw["content_sha256"] = canonical_json_hash(file_inventory(staging))
        manifest = DatasetBundle.from_dict(raw)
        DatasetBundleReader(staging, manifest).validate_rows(PlatformProfile())
        write_json(staging / "dataset.yaml", manifest.to_dict())
        result = canonical_json_hash(file_inventory(staging))
    return result


def package_experts(source: Path, output: Path) -> str:
    """Package ONNX files and an experts.yaml template, filling model digests."""
    raw = parse_manifest((source / "experts.yaml").read_text())
    for expert in raw["experts"]:
        expert["model_sha256"] = sha256_file(safe_path(source, expert["model_path"]))
    manifest = ExpertBundle.from_dict(raw)
    with atomic_directory(output) as staging:
        for expert in raw["experts"]:
            target = safe_path(staging, expert["model_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(source, expert["model_path"]), target)
        write_json(staging / "experts.yaml", manifest.to_dict())
        result = canonical_json_hash(file_inventory(staging))
    return result
