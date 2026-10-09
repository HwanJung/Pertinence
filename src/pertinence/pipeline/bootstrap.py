"""Create portable CIFAR-10 and ONNX expert bundles from pinned raw assets."""

from __future__ import annotations

import io
import pickle
from pathlib import Path
import tarfile
from typing import BinaryIO

import numpy as np
from PIL import Image

from ..contracts import DatasetBundle, ExpertBundle
from ..experts import FrozenFeatureExtractor, get_expert_spec, load_plain_state_dict
from ..reproducibility import canonical_json_hash, sha256_file
from .artifacts import (
    artifact_sha256,
    atomic_directory,
    file_inventory,
    local_path,
    safe_path,
    verify_artifact,
    write_json,
)


CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
EXPERT_IDS = ("shufflenetv2_x0_5", "mobilenetv2_x0_5", "resnet56")


class _RestrictedCifarUnpickler(pickle.Unpickler):
    """Load the pinned CIFAR NumPy payload without allowing arbitrary globals."""

    _ALLOWED_GLOBALS = {
        ("_codecs", "encode"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"unsupported CIFAR pickle global: {module}.{name}")
        return super().find_class(module, name)


def _load_cifar_pickle(handle: BinaryIO) -> dict:
    value = _RestrictedCifarUnpickler(handle, encoding="bytes").load()
    if not isinstance(value, dict):
        raise ValueError("CIFAR archive member must contain a mapping")
    return value


def _archive_member(archive: tarfile.TarFile, relative: str) -> BinaryIO:
    expected = f"cifar-10-batches-py/{relative}"
    matches = [member for member in archive.getmembers() if member.name == expected]
    if len(matches) != 1 or not matches[0].isfile():
        raise ValueError(f"CIFAR archive must contain exactly one regular {expected}")
    handle = archive.extractfile(matches[0])
    if handle is None:
        raise ValueError(f"cannot read CIFAR archive member: {expected}")
    return handle


def _encode_png(row: np.ndarray) -> bytes:
    image = row.reshape(3, 32, 32).transpose(1, 2, 0)
    output = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(output, format="PNG", optimize=False)
    return output.getvalue()


def _write_cifar_shard(
    path: Path,
    payload: dict,
    *,
    pool: str,
    offset: int,
    expected_samples: int,
) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    images = payload.get(b"data")
    labels = payload.get(b"labels")
    if (
        not isinstance(images, np.ndarray)
        or images.dtype != np.uint8
        or images.shape != (expected_samples, 3072)
        or not isinstance(labels, list)
        or len(labels) != expected_samples
        or any(type(label) is not int or not 0 <= label < 10 for label in labels)
    ):
        raise ValueError(f"invalid CIFAR-10 {pool} batch payload")
    rows = {
        "sample_id": [f"{pool}-{offset + index:05d}" for index in range(expected_samples)],
        "image": [_encode_png(row) for row in images],
        "target": np.asarray(labels, dtype=np.int64),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pydict(
            rows,
            schema=pa.schema(
                [("sample_id", pa.string()), ("image", pa.binary()), ("target", pa.int64())]
            ),
        ),
        path,
        compression="zstd",
    )
    return expected_samples


def prepare_cifar10_dataset_bundle(
    cifar10_archive: str | Path,
    cifar10_sha256: str,
    dataset_bundle: Path,
    *,
    train_batches: int = 5,
    samples_per_batch: int = 10_000,
) -> str:
    """Convert a pinned CIFAR-10 Python archive into the portable Parquet contract."""

    source = local_path(cifar10_archive)
    if not source.is_file():
        raise ValueError("CIFAR-10 raw asset must be one archive file")
    verify_artifact(source, cifar10_sha256)
    if train_batches < 1 or samples_per_batch < 1:
        raise ValueError("CIFAR batch counts must be positive")
    with tarfile.open(source, mode="r:gz") as archive, atomic_directory(dataset_bundle) as stage:
        metadata = _load_cifar_pickle(_archive_member(archive, "batches.meta"))
        raw_names = metadata.get(b"label_names")
        class_names = tuple(
            item.decode("ascii") if isinstance(item, bytes) else item for item in raw_names or ()
        )
        if class_names != CIFAR10_CLASS_NAMES:
            raise ValueError("CIFAR-10 class names or ordering differ from the pinned contract")
        train_samples = 0
        for index in range(1, train_batches + 1):
            payload = _load_cifar_pickle(_archive_member(archive, f"data_batch_{index}"))
            train_samples += _write_cifar_shard(
                stage / "train" / f"part-{index - 1:03d}.parquet",
                payload,
                pool="train",
                offset=train_samples,
                expected_samples=samples_per_batch,
            )
        evaluation = _load_cifar_pickle(_archive_member(archive, "test_batch"))
        evaluation_samples = _write_cifar_shard(
            stage / "evaluation" / "part-000.parquet",
            evaluation,
            pool="evaluation",
            offset=0,
            expected_samples=samples_per_batch,
        )
        manifest = {
            "schema_version": 1,
            "task": "image_classification",
            "dataset_id": "cifar10",
            "dataset_version": f"python-tar-{cifar10_sha256[:12]}",
            "num_classes": 10,
            "class_names": list(class_names),
            "image_codec": "png",
            "train_glob": "train/part-*.parquet",
            "evaluation_glob": "evaluation/part-*.parquet",
            "train_samples": train_samples,
            "evaluation_samples": evaluation_samples,
            "content_sha256": canonical_json_hash(file_inventory(stage)),
        }
        write_json(stage / "dataset.yaml", DatasetBundle.from_dict(manifest).to_dict())
    return artifact_sha256(dataset_bundle)


def _build_resnet56():
    import torch
    from torch import nn

    def conv3x3(in_channels: int, out_channels: int, stride: int = 1):
        return nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )

    class BasicBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
            super().__init__()
            self.conv1 = conv3x3(in_channels, out_channels, stride)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(out_channels, out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.downsample = None
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )

        def forward(self, inputs):
            identity = inputs
            outputs = self.relu(self.bn1(self.conv1(inputs)))
            outputs = self.bn2(self.conv2(outputs))
            if self.downsample is not None:
                identity = self.downsample(inputs)
            return self.relu(outputs + identity)

    class CifarResNet56(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inplanes = 16
            self.conv1 = conv3x3(3, 16)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(16, 9)
            self.layer2 = self._make_layer(32, 9, stride=2)
            self.layer3 = self._make_layer(64, 9, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, 10)

        def _make_layer(self, channels: int, blocks: int, stride: int = 1):
            layers = [BasicBlock(self.inplanes, channels, stride)]
            self.inplanes = channels
            layers.extend(BasicBlock(channels, channels) for _ in range(1, blocks))
            return nn.Sequential(*layers)

        def forward(self, inputs):
            outputs = self.relu(self.bn1(self.conv1(inputs)))
            outputs = self.layer3(self.layer2(self.layer1(outputs)))
            outputs = self.avgpool(outputs)
            return self.fc(torch.flatten(outputs, 1))

    return CifarResNet56()


def build_export_expert(expert_id: str):
    """Build the exact CPU architecture expected by the three pinned checkpoints."""

    from torch import nn
    from torchvision.models import mobilenet_v2, shufflenet_v2_x0_5

    if expert_id == "shufflenetv2_x0_5":
        model = shufflenet_v2_x0_5(weights=None, num_classes=10)
        model.conv1[0].stride = (1, 1)
        model.maxpool = nn.Identity()
        return model
    if expert_id == "mobilenetv2_x0_5":
        model = mobilenet_v2(weights=None, width_mult=0.5, num_classes=10)
        model.features[0][0].stride = (1, 1)
        model.features[2].conv[1][0].stride = (1, 1)
        return model
    if expert_id == "resnet56":
        return _build_resnet56()
    raise ValueError(f"unsupported export expert: {expert_id}")


def _export_one_expert(expert_id: str, checkpoint: Path, output: Path) -> None:
    import onnx
    import torch
    import onnxruntime as ort

    model = build_export_expert(expert_id)
    model.load_state_dict(load_plain_state_dict(checkpoint), strict=True)
    model.requires_grad_(False)
    model.eval()
    wrapper = FrozenFeatureExtractor(model).eval()
    example = torch.linspace(-1.0, 1.0, 2 * 3 * 32 * 32, dtype=torch.float32).reshape(
        2, 3, 32, 32
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        example,
        output,
        input_names=["image"],
        output_names=["logits", "features"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    exported = onnx.load(output, load_external_data=False)
    onnx.checker.check_model(exported)
    if any(tensor.external_data for tensor in exported.graph.initializer):
        raise ValueError("exported ONNX model unexpectedly contains external tensor data")
    with torch.no_grad():
        expected_logits, expected_features = wrapper(example)
    session = ort.InferenceSession(
        exported.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    actual_logits, actual_features = session.run(None, {"image": example.numpy()})
    np.testing.assert_allclose(actual_logits, expected_logits.numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(actual_features, expected_features.numpy(), rtol=1e-4, atol=1e-5)
    single_logits, single_features = session.run(None, {"image": example[:1].numpy()})
    assert single_logits.shape == (1, 10)
    assert single_features.shape == (1, wrapper.feature_dim)


def export_cifar10_expert_bundle(
    shufflenet_checkpoint: str | Path,
    shufflenet_sha256: str,
    mobilenet_checkpoint: str | Path,
    mobilenet_sha256: str,
    resnet56_checkpoint: str | Path,
    resnet56_sha256: str,
    expert_bundle: Path,
) -> str:
    """Export the fixed three CPU experts to one self-describing ONNX bundle."""

    inputs = {
        "shufflenetv2_x0_5": (local_path(shufflenet_checkpoint), shufflenet_sha256),
        "mobilenetv2_x0_5": (local_path(mobilenet_checkpoint), mobilenet_sha256),
        "resnet56": (local_path(resnet56_checkpoint), resnet56_sha256),
    }
    for path, expected in inputs.values():
        if not path.is_file():
            raise ValueError("expert checkpoint raw asset must be one file")
        verify_artifact(path, expected)
    normalization = {
        "resize": [32, 32],
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010],
    }
    with atomic_directory(expert_bundle) as stage:
        experts = []
        for expert_id in EXPERT_IDS:
            checkpoint, checkpoint_digest = inputs[expert_id]
            relative = f"models/{expert_id}/model.onnx"
            model_path = safe_path(stage, relative)
            _export_one_expert(expert_id, checkpoint, model_path)
            spec = get_expert_spec(expert_id)
            experts.append(
                {
                    "id": expert_id,
                    "version": f"checkpoint-{checkpoint_digest[:12]}",
                    "format": "onnx",
                    "model_path": relative,
                    "model_sha256": sha256_file(model_path),
                    "input": {
                        "layout": "NCHW",
                        "dtype": "float32",
                        "shape": [3, 32, 32],
                        "preprocessing": normalization,
                    },
                    "outputs": {"logits": "logits", "features": "features"},
                    "num_classes": 10,
                    "costs": {"mflops": spec.mflops},
                }
            )
        manifest = {"schema_version": 1, "bundle_id": "cifar10-pertinence-v1", "experts": experts}
        write_json(stage / "experts.yaml", ExpertBundle.from_dict(manifest).to_dict())
    return artifact_sha256(expert_bundle)
