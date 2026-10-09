"""Offline integration gate for the approved CIFAR-10 serving artifacts."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass, field
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from ..pipeline.bundles import DatasetBundleReader, decode_image
from ..contracts import DatasetBundle, ExpertBundle, NormalizedRunContract
from ..pipeline.artifacts import (
    artifact_sha256,
    file_inventory,
    local_path,
    read_sealed,
    safe_path,
    seal,
    verify_artifact,
    write_json,
)
from ..pipeline.search import validate_models, validate_search
from ..reproducibility import canonical_json_hash, sha256_file
from ..runtimes.onnx import ExpertRuntime
from .runtime import load_dispatcher_npz
from .trace import FINAL_SAMPLE_ORDER_SHA256, normalize_cifar_images
from .v2 import parse_infer_request


EXPERT_IDS = ("shufflenetv2_x0_5", "mobilenetv2_x0_5", "resnet56")
EXPERT_MODEL_SHA256 = {
    "shufflenetv2_x0_5": "04c7c69ea710375a3f8e7314c4b58f9ca225dd5bf8f4d13069e1c4b3f3833ea7",
    "mobilenetv2_x0_5": "69225030b26f6de25f85f2b6873b6afb9095cac79b68fa153c59e5a07534d62b",
    "resnet56": "9494958830e238f0cb35ad4f490256a4f07794936403559ac3651c4af4f96858",
}
EXPECTED_CORRECT = {
    "shufflenetv2_x0_5": 1809,
    "mobilenetv2_x0_5": 1858,
    "resnet56": 1892,
    "dispatcher-000733": 1893,
}
EXPECTED_ROUTE_COUNTS = (0, 640, 1360)
APPROVED_LINEAGE = {
    "contract_fingerprint": "5da2c50b3b718923326b55ce21fbdd5b74beed9220df6baecfbfe6091df4bcc0",
    "search_fingerprint": "8c8a2491ed065d6fe100b76273be723c0d8bcb5ef60687f9ad631ef568ef2b07",
    "front_id": "4c31958c59a2c3f4f30b2c4ec86b24b504dd85a5bfbc4616f61f0405d4f52c10",
    "final_fingerprint": "1c1d253694392e54b71e4e44b4e4248e0084da294cc38d160d93a9de35b8423c",
}


@dataclass(frozen=True)
class IntegrationExpectations:
    """Immutable receipts expected from the independently approved run."""

    samples: int = 2_000
    sample_order_sha256: str = FINAL_SAMPLE_ORDER_SHA256
    correct: Mapping[str, int] = field(default_factory=lambda: dict(EXPECTED_CORRECT))
    route_counts: tuple[int, int, int] = EXPECTED_ROUTE_COUNTS
    expert_model_sha256: Mapping[str, str] = field(
        default_factory=lambda: dict(EXPERT_MODEL_SHA256)
    )


def array_sha256(value: np.ndarray) -> str:
    """Hash an array together with its dtype and shape, independent of NPY encoding."""

    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_tensor_interface(
    value: Any, *, name: str, dimensions: Sequence[int], label: str
) -> None:
    if value.name != name or value.type != "tensor(float)" or len(value.shape) != len(dimensions) + 1:
        raise ValueError(f"{label} must be FP32 [batch,{','.join(map(str, dimensions))}]")
    batch = value.shape[0]
    if isinstance(batch, int) and batch != 1:
        raise ValueError(f"{label} must support batch size 1")
    if any(actual != expected for actual, expected in zip(value.shape[1:], dimensions)):
        raise ValueError(f"{label} has an unexpected static shape")


class _ExpertSession:
    def __init__(self, runtime: ExpertRuntime, expert_id: str) -> None:
        self.runtime = runtime
        self.expert_id = expert_id
        inputs = runtime.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"{expert_id} must have exactly one ONNX input")
        _validate_tensor_interface(
            inputs[0], name="image", dimensions=(3, 32, 32), label=f"{expert_id} image"
        )
        outputs = {item.name: item for item in runtime.session.get_outputs()}
        if "logits" not in outputs:
            raise ValueError(f"{expert_id} is missing logits output")
        _validate_tensor_interface(
            outputs["logits"], name="logits", dimensions=(10,), label=f"{expert_id} logits"
        )
        if expert_id == EXPERT_IDS[0]:
            if "features" not in outputs:
                raise ValueError("shufflenetv2_x0_5 is missing features output")
            _validate_tensor_interface(
                outputs["features"],
                name="features",
                dimensions=(1024,),
                label="shufflenetv2_x0_5 features",
            )

    def predict(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        names = ["logits", "features"] if self.expert_id == EXPERT_IDS[0] else ["logits"]
        values = self.runtime.session.run(names, {self.runtime.input_name: images})
        logits = np.asarray(values[0])
        expected = (len(images), 10)
        if logits.dtype != np.float32 or logits.shape != expected or not np.isfinite(logits).all():
            raise ValueError(f"{self.expert_id} emitted invalid FP32 logits")
        features = None if len(values) == 1 else np.asarray(values[1])
        if features is not None and (
            features.dtype != np.float32
            or features.shape != (len(images), 1024)
            or not np.isfinite(features).all()
        ):
            raise ValueError("shufflenetv2_x0_5 emitted invalid FP32 features")
        return logits, features


def _validate_evaluated_bundle(
    root: Path, dispatcher_id: str, approved_sha256: str
) -> tuple[dict, Path, dict[str, str]]:
    contract = NormalizedRunContract.from_dict(
        read_sealed(safe_path(root, "contract.json"))
    ).to_dict()
    search = read_sealed(safe_path(root, "search-result.json"))
    final = read_sealed(safe_path(root, "final-report.json"))
    manifest = read_sealed(safe_path(root, "manifest.json"))
    records = validate_search(search, contract)
    validate_models(root, manifest, search, contract)
    actual_lineage = {
        "contract_fingerprint": contract["fingerprint"],
        "search_fingerprint": search["fingerprint"],
        "front_id": search["front_id"],
        "final_fingerprint": final["fingerprint"],
    }
    if actual_lineage != APPROVED_LINEAGE:
        raise ValueError("evaluated bundle lineage differs from the approved KFP run")
    if (
        manifest.get("kind") != "evaluated-dispatchers"
        or manifest.get("status") != "evaluated"
        or manifest.get("final_report_fingerprint") != final["fingerprint"]
        or final.get("contract_fingerprint") != contract["fingerprint"]
        or final.get("search_fingerprint") != search["fingerprint"]
        or final.get("front_id") != search["front_id"]
    ):
        raise ValueError("evaluated bundle final lineage mismatch")
    models = [item for item in manifest["models"] if item["model_id"] == dispatcher_id]
    solutions = [item for item in final["solutions"] if item["model_id"] == dispatcher_id]
    if [item["model_id"] for item in manifest["models"]] != [
        item["model_id"] for item in final["solutions"]
    ] or any(
        model_item["checkpoint_sha256"] != solution_item["checkpoint_sha256"]
        for model_item, solution_item in zip(manifest["models"], final["solutions"])
    ):
        raise ValueError("evaluated models and final solutions differ")
    if len(models) != 1 or len(solutions) != 1:
        raise ValueError("approved dispatcher must occur exactly once in evaluated artifacts")
    model, solution = models[0], solutions[0]
    expert_order = contract["config"]["routing"]["expert_ids"]
    if (
        contract["dimensions"]["D"] != 1024
        or contract["dimensions"]["K"] != 3
        or expert_order != list(EXPERT_IDS)
        or contract["config"]["routing"]["feature_extractor_id"] != EXPERT_IDS[0]
        or model["dimensions"] != {"D": 1024, "K": 3}
        or model["expert_order"] != list(EXPERT_IDS)
        or model["evaluation_id"] != 733
        or model["evaluation_id"] not in search["pareto_ids"]
        or solution["search_metrics"] != records[model["evaluation_id"]]["metrics"]
    ):
        raise ValueError("approved dispatcher dimensions, order, or search lineage mismatch")
    if model["checkpoint_sha256"] != approved_sha256 or solution["checkpoint_sha256"] != approved_sha256:
        raise ValueError("approved dispatcher SHA-256 differs from evaluated lineage")
    checkpoint = safe_path(root, model["checkpoint"])
    verify_artifact(checkpoint, approved_sha256)
    load_dispatcher_npz(checkpoint)
    lineage = {
        "contract_fingerprint": contract["fingerprint"],
        "search_fingerprint": search["fingerprint"],
        "front_id": search["front_id"],
        "final_fingerprint": final["fingerprint"],
        "evaluated_fingerprint": manifest["fingerprint"],
    }
    return contract, checkpoint, lineage


def _validate_inputs(
    dataset_root: Path,
    split_path: Path,
    expert_root: Path,
    contract: dict,
    expectations: IntegrationExpectations,
) -> tuple[DatasetBundleReader, list[str], list[dict]]:
    dataset = DatasetBundle.from_dict(contract["dataset"])
    experts = ExpertBundle.from_dict(contract["experts"])
    verify_artifact(dataset_root, contract["inputs"]["dataset_sha256"])
    verify_artifact(expert_root, contract["inputs"]["experts_sha256"])
    split = read_sealed(split_path)
    if split["fingerprint"] != contract["splits"]["fingerprint"]:
        raise ValueError("split manifest differs from evaluated contract")
    final_ids = split.get("sample_ids", {}).get("final")
    if not isinstance(final_ids, list) or len(final_ids) != expectations.samples:
        raise ValueError("split manifest final sample count differs from approved receipt")
    if len(set(final_ids)) != len(final_ids):
        raise ValueError("split manifest final sample IDs are not unique")
    if canonical_json_hash(final_ids) != expectations.sample_order_sha256:
        raise ValueError("final sample order differs from approved receipt")
    if contract["splits"]["final"] != {
        "samples": expectations.samples,
        "sample_ids_sha256": expectations.sample_order_sha256,
    }:
        raise ValueError("contract final split differs from approved receipt")
    if contract["dataset"]["content_sha256"] != canonical_json_hash(
        file_inventory(dataset_root, exclude=("dataset.yaml",))
    ):
        raise ValueError("dataset content SHA-256 mismatch")
    source_experts = ExpertBundle.from_text(
        safe_path(expert_root, "experts.yaml").read_text(encoding="utf-8")
    ).to_dict()
    if source_experts != experts.to_dict():
        raise ValueError("expert bundle metadata differs from evaluated contract")
    ordered = {item["id"]: item for item in source_experts["experts"]}
    if list(contract["config"]["routing"]["expert_ids"]) != list(EXPERT_IDS):
        raise ValueError("evaluated expert order differs from the approved order")
    selected = [ordered[item] for item in EXPERT_IDS]
    approved_preprocessing = {
        "resize": [32, 32],
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010],
    }
    for expert in selected:
        expected_sha = expectations.expert_model_sha256[expert["id"]]
        if expert["model_sha256"] != expected_sha:
            raise ValueError(f"{expert['id']} model SHA-256 differs from approved receipt")
        verify_artifact(safe_path(expert_root, expert["model_path"]), expected_sha)
        if (
            expert["input"]["shape"] != [3, 32, 32]
            or expert["input"]["layout"] != "NCHW"
            or expert["input"]["dtype"] != "float32"
            or expert["input"]["preprocessing"] != approved_preprocessing
            or expert["outputs"]["logits"] != "logits"
            or expert["num_classes"] != 10
        ):
            raise ValueError(f"{expert['id']} metadata differs from serving interface")
    if selected[0]["outputs"].get("features") != "features":
        raise ValueError("shufflenetv2_x0_5 must declare the features output")
    return DatasetBundleReader(dataset_root, dataset), final_ids, selected


def _load_final_rows(
    reader: DatasetBundleReader, final_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    positions = {sample_id: index for index, sample_id in enumerate(final_ids)}
    images = np.empty((len(final_ids), 32, 32, 3), dtype=np.uint8)
    labels = np.empty(len(final_ids), dtype=np.int64)
    visited: set[str] = set()
    for batch in reader.batches("evaluation", 128):
        for row in batch:
            sample_id = row["sample_id"]
            if sample_id not in positions:
                continue
            if sample_id in visited:
                raise ValueError(f"duplicate final dataset sample: {sample_id}")
            image = decode_image(row["image"], reader.manifest["image_codec"], 32 * 32)
            if image.size != (32, 32):
                raise ValueError("final CIFAR-10 image must be exactly 32x32")
            position = positions[sample_id]
            images[position] = np.asarray(image, dtype=np.uint8)
            labels[position] = row["target"]
            visited.add(sample_id)
    missing = set(final_ids) - visited
    if missing:
        raise ValueError(f"final dataset samples are missing: {len(missing)}")
    if np.any(labels < 0) or np.any(labels >= 10):
        raise ValueError("final labels must be CIFAR-10 class indices")
    return images, labels


def _infer(
    normalized: np.ndarray,
    sessions: Sequence[_ExpertSession],
    dispatcher: Any,
    *,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    logits_parts: list[list[np.ndarray]] = [[] for _ in sessions]
    feature_parts: list[np.ndarray] = []
    for offset in range(0, len(normalized), batch_size):
        batch = normalized[offset : offset + batch_size]
        for index, session in enumerate(sessions):
            logits, features = session.predict(batch)
            logits_parts[index].append(logits)
            if index == 0:
                if features is None:
                    raise ValueError("extractor inference omitted features")
                feature_parts.append(features)
    expert_logits = np.stack([np.concatenate(items) for items in logits_parts], axis=1)
    features = np.concatenate(feature_parts)
    route_logits = features @ dispatcher.weight.T + dispatcher.bias
    if route_logits.dtype != np.float32 or not np.isfinite(route_logits).all():
        raise ValueError("dispatcher emitted invalid route logits")
    routes = route_logits.argmax(axis=1).astype(np.int64)
    predictions = expert_logits.argmax(axis=2).astype(np.int64)
    selected_logits = expert_logits[np.arange(len(routes)), routes]
    selected_predictions = selected_logits.argmax(axis=1).astype(np.int64)
    # The extractor's already-computed logits are the source for route zero; no second
    # extractor invocation is permitted by the serving contract.
    extractor_mask = routes == 0
    if not np.array_equal(selected_logits[extractor_mask], expert_logits[extractor_mask, 0]):
        raise ValueError("extractor route did not reuse the extractor logits")
    if not np.array_equal(selected_predictions, predictions[np.arange(len(routes)), routes]):
        raise ValueError("selected expert class differs from route selection")
    return {
        "expert_logits": expert_logits,
        "extractor_features": features,
        "route_logits": route_logits,
        "routes": routes,
        "expert_predictions": predictions,
        "dispatcher_predictions": selected_predictions,
        "selected_logits": selected_logits,
    }


@contextmanager
def _new_output(path: Path) -> Iterator[Path]:
    """Yield a staging directory and rename it into place without replacement."""

    output = Path(path)
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        yield temporary
        if output.exists() or output.is_symlink():
            raise ValueError(f"output appeared during integration: {output}")
        try:
            _rename_directory_no_replace(temporary, output)
        except FileExistsError as error:
            raise ValueError(f"output appeared during integration: {output}") from error
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing a racing destination."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise OSError(errno.ENOSYS, "renameat2 is required for create-only publication") from error
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _stage_build_bundle(
    stage: Path,
    expert_root: Path,
    experts: Sequence[dict],
    evaluated_root: Path,
    checkpoint: Path,
) -> dict[str, str]:
    """Convert KFP bundles into the exact allowlisted Cloud Build input layout."""

    build_root = stage / "build-bundle"
    for expert in experts:
        destination = build_root / "experts" / expert["id"]
        destination.mkdir(parents=True)
        source = safe_path(expert_root, expert["model_path"])
        shutil.copyfile(source, destination / "model.onnx")
        (destination / "model.sha256").write_text(
            f"{expert['model_sha256']}  model.onnx\n", encoding="utf-8"
        )
    evaluated_destination = build_root / "evaluated"
    evaluated_destination.mkdir(parents=True)
    for name in ("contract.json", "search-result.json", "final-report.json", "manifest.json"):
        shutil.copyfile(safe_path(evaluated_root, name), evaluated_destination / name)
    relative_checkpoint = checkpoint.relative_to(evaluated_root).as_posix()
    if relative_checkpoint != "dispatcher-000733.npz":
        raise ValueError("approved dispatcher checkpoint must be dispatcher-000733.npz")
    checkpoint_destination = safe_path(evaluated_destination, relative_checkpoint)
    checkpoint_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, checkpoint_destination)
    return file_inventory(build_root)


def integrate_serving_artifacts(
    *,
    dataset_bundle: Path,
    split_manifest: Path,
    expert_bundle: Path,
    evaluated_bundle: Path,
    dispatcher_id: str,
    approved_dispatcher_sha256: str,
    approved_label_sha256: str,
    output: Path,
    expectations: IntegrationExpectations | None = None,
    session_factory: Callable[[ExpertRuntime, str], _ExpertSession] = _ExpertSession,
) -> dict[str, Any]:
    """Run the immutable artifacts together and publish a create-only evidence directory."""

    expected = expectations or IntegrationExpectations()
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    if dispatcher_id != "dispatcher-000733":
        raise ValueError("only the explicitly approved dispatcher-000733 may be integrated")
    for value in (approved_dispatcher_sha256, approved_label_sha256):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("approved SHA-256 values must be 64 lowercase hexadecimal characters")
    dataset_root = local_path(dataset_bundle)
    split_path = local_path(split_manifest)
    expert_root = local_path(expert_bundle)
    evaluated_root = local_path(evaluated_bundle)
    if not split_path.is_file():
        raise ValueError("split manifest must be a regular file")
    contract, checkpoint, lineage = _validate_evaluated_bundle(
        evaluated_root, dispatcher_id, approved_dispatcher_sha256
    )
    reader, final_ids, experts = _validate_inputs(
        dataset_root, split_path, expert_root, contract, expected
    )
    raw_images, labels = _load_final_rows(reader, final_ids)
    if array_sha256(labels) != approved_label_sha256:
        raise ValueError("final label values differ from the independent approved receipt")
    normalized = normalize_cifar_images(raw_images)
    sessions = []
    for expert in experts:
        runtime = ExpertRuntime(expert_root, expert, device="cpu", intra_op_num_threads=1)
        sessions.append(session_factory(runtime, expert["id"]))
    result = _infer(normalized, sessions, load_dispatcher_npz(checkpoint))
    accuracy = {
        expert_id: int(np.sum(result["expert_predictions"][:, index] == labels))
        for index, expert_id in enumerate(EXPERT_IDS)
    }
    accuracy[dispatcher_id] = int(np.sum(result["dispatcher_predictions"] == labels))
    if accuracy != dict(expected.correct):
        raise ValueError(f"offline correct counts differ from approved receipt: {accuracy}")
    route_counts = tuple(np.bincount(result["routes"], minlength=3).tolist())
    if route_counts != expected.route_counts:
        raise ValueError(f"offline route counts differ from approved receipt: {route_counts}")

    evaluation_arrays = {
        "images": raw_images,
        "labels": labels,
        "sample_ids": np.asarray(final_ids, dtype=np.str_),
        **{
            f"{expert_id}_predictions": result["expert_predictions"][:, index]
            for index, expert_id in enumerate(EXPERT_IDS)
        },
        f"{dispatcher_id}_predictions": result["dispatcher_predictions"],
        f"{dispatcher_id}_routes": result["routes"],
        "expert_logits": result["expert_logits"],
        "extractor_features": result["extractor_features"],
        "dispatcher_route_logits": result["route_logits"],
        "dispatcher_selected_logits": result["selected_logits"],
    }
    local_arrays = {
        **{
            f"{expert_id}_predictions": result["expert_predictions"][:, index]
            for index, expert_id in enumerate(EXPERT_IDS)
        },
        **{
            f"{expert_id}_logits": result["expert_logits"][:, index]
            for index, expert_id in enumerate(EXPERT_IDS)
        },
        "static-resnet-router_predictions": result["expert_predictions"][:, 2],
        "static-resnet-router_logits": result["expert_logits"][:, 2],
        f"{dispatcher_id}_predictions": result["dispatcher_predictions"],
        f"{dispatcher_id}_routes": result["routes"],
        f"{dispatcher_id}_logits": result["selected_logits"],
    }
    request = {
        "id": f"integration-{final_ids[0]}",
        "inputs": [
            {
                "name": "image",
                "shape": [1, 3, 32, 32],
                "datatype": "FP32",
                "data": normalized[0].reshape(-1).tolist(),
            }
        ],
    }
    _, parsed = parse_infer_request(request)
    if not np.array_equal(parsed, normalized[0:1]):
        raise ValueError("golden request does not preserve the normalized image")

    with _new_output(output) as stage:
        _write_npz(stage / "evaluation-final.npz", evaluation_arrays)
        _write_npz(stage / "local-predictions.npz", local_arrays)
        write_json(stage / "request.json", request)
        build_inventory = _stage_build_bundle(
            stage, expert_root, experts, evaluated_root, checkpoint
        )
        outputs = {
            name: {
                "sha256": sha256_file(stage / name),
                "size_bytes": (stage / name).stat().st_size,
            }
            for name in ("evaluation-final.npz", "local-predictions.npz", "request.json")
        }
        receipt = seal(
            {
                "schema_version": 1,
                "kind": "serving-artifact-integration",
                "valid": True,
                "dispatcher_id": dispatcher_id,
                "dispatcher_sha256": approved_dispatcher_sha256,
                "expert_model_sha256": dict(expected.expert_model_sha256),
                "lineage": lineage,
                "samples": expected.samples,
                "evaluation_images": "raw uint8 NHWC; trace creation performs normalization",
                "sample_order_sha256": expected.sample_order_sha256,
                "label_sha256": approved_label_sha256,
                "accuracy_correct": accuracy,
                "route_counts": list(route_counts),
                "array_sha256": {
                    key: array_sha256(value)
                    for key, value in {
                        "expert_logits": result["expert_logits"],
                        "extractor_features": result["extractor_features"],
                        "dispatcher_route_logits": result["route_logits"],
                        "dispatcher_routes": result["routes"],
                        "dispatcher_selected_logits": result["selected_logits"],
                        "dispatcher_predictions": result["dispatcher_predictions"],
                    }.items()
                },
                "extractor_logits_reused": True,
                "build_bundle": {
                    "path": "build-bundle",
                    "files": build_inventory,
                    "sha256": canonical_json_hash(build_inventory),
                },
                "outputs": outputs,
                "inputs": {
                    "dataset_bundle_sha256": artifact_sha256(dataset_root),
                    "expert_bundle_sha256": artifact_sha256(expert_root),
                    "split_manifest_sha256": sha256_file(split_path),
                    "evaluated_bundle_sha256": artifact_sha256(evaluated_root),
                },
            }
        )
        write_json(stage / "receipt.json", receipt)
    return receipt
