"""CPU bootstrap components turn pinned raw assets into portable contracts."""

from __future__ import annotations

import io
import pickle
from pathlib import Path
import tarfile

import numpy as np
import pytest
import torch

from pertinence.experts import FrozenFeatureExtractor
from pertinence.pipeline.artifacts import artifact_sha256
from pertinence.pipeline.bootstrap import (
    CIFAR10_CLASS_NAMES,
    EXPERT_IDS,
    build_export_expert,
    prepare_cifar10_dataset_bundle,
)
from pertinence.reproducibility import sha256_file


def _add_pickle(archive: tarfile.TarFile, name: str, value: object) -> None:
    payload = pickle.dumps(value, protocol=2)
    member = tarfile.TarInfo(f"cifar-10-batches-py/{name}")
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def test_prepare_cifar10_bundle_from_raw_archive(tmp_path: Path):
    pytest.importorskip("pyarrow")
    raw = tmp_path / "cifar.tar.gz"
    images = np.arange(2 * 3072, dtype=np.uint8).reshape(2, 3072)
    batch = {b"data": images, b"labels": [0, 9]}
    with tarfile.open(raw, "w:gz") as archive:
        _add_pickle(archive, "batches.meta", {b"label_names": list(CIFAR10_CLASS_NAMES)})
        _add_pickle(archive, "data_batch_1", batch)
        _add_pickle(archive, "test_batch", batch)
    output = tmp_path / "dataset"
    result = prepare_cifar10_dataset_bundle(
        raw,
        sha256_file(raw),
        output,
        train_batches=1,
        samples_per_batch=2,
    )
    assert result == artifact_sha256(output)
    assert (output / "train" / "part-000.parquet").is_file()
    assert (output / "evaluation" / "part-000.parquet").is_file()
    assert b'"train_samples": 2' in (output / "dataset.yaml").read_bytes()


@pytest.mark.parametrize(
    "expert_id,state_items,feature_dim",
    [
        ("shufflenetv2_x0_5", 338, 1024),
        ("mobilenetv2_x0_5", 314, 1280),
        ("resnet56", 344, 64),
    ],
)
def test_embedded_export_architectures(expert_id: str, state_items: int, feature_dim: int):
    assert expert_id in EXPERT_IDS
    model = build_export_expert(expert_id).eval()
    assert len(model.state_dict()) == state_items
    wrapper = FrozenFeatureExtractor(model).eval()
    with torch.no_grad():
        logits, features = wrapper(torch.zeros(2, 3, 32, 32))
    assert logits.shape == (2, 10)
    assert features.shape == (2, feature_dim)
