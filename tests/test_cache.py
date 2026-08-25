from __future__ import annotations

from pathlib import Path
import weakref

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from pertinence.cache import (
    CACHE_SCHEMA_VERSION,
    CacheBuildRequest,
    CacheFingerprintMismatch,
    build_prediction_caches,
    cache_path,
    route_labels_from_predictions,
    save_cache_atomically,
)
from pertinence.data import SplitIndices


class _VectorDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, labels: list[int]) -> None:
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        value = torch.zeros(10)
        value[self.labels[index]] = 1.0
        return value, self.labels[index]


class _LocalExpert(nn.Module):
    def __init__(self, shift: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(10, 10, bias=False)
        weight = torch.zeros(10, 10)
        for label in range(10):
            weight[(label + shift) % 10, label] = 1.0
        self.classifier.weight.data.copy_(weight)

    def forward(self, inputs: Tensor) -> Tensor:
        assert self.training is False
        assert all(parameter.requires_grad is False for parameter in self.parameters())
        return self.classifier(inputs)


def _splits() -> SplitIndices:
    return SplitIndices(
        dispatcher_train=np.asarray([0, 1, 2, 3], dtype=np.int64),
        ga_search=np.asarray([3, 1], dtype=np.int64),
        final_evaluation=np.asarray([2, 0], dtype=np.int64),
        seed=11,
    )


def _request(tmp_path: Path) -> CacheBuildRequest:
    return CacheBuildRequest(
        model_order=("model_0", "model_1"),
        feature_extractor="model_0",
        source_root=tmp_path / "source",
        checkpoint_paths={
            "model_0": tmp_path / "model_0.pt",
            "model_1": tmp_path / "model_1.pt",
        },
        output_directory=tmp_path / "cache",
        asset_fingerprint="a" * 64,
        config_fingerprint="b" * 64,
        batch_size=2,
        num_workers=0,
        device="cpu",
    )


def test_route_labels_use_cheapest_correct_and_cheapest_fallback() -> None:
    predictions = torch.tensor(
        [
            [0, 0, 3],
            [8, 1, 1],
            [2, 7, 2],
            [9, 9, 8],
        ]
    )
    targets = torch.tensor([0, 1, 2, 4])
    assert route_labels_from_predictions(predictions, targets).tolist() == [0, 1, 0, 0]


def test_dry_run_does_not_require_datasets_or_load_models(tmp_path: Path) -> None:
    calls: list[str] = []

    def forbidden_loader(*args: object, **kwargs: object) -> nn.Module:
        calls.append("called")
        raise AssertionError((args, kwargs))

    request = _request(tmp_path)
    report = build_prediction_caches(
        request, _splits(), dry_run=True, model_loader=forbidden_loader
    )

    assert calls == []
    assert {result.status for result in report.results} == {"planned"}
    assert not request.output_directory.exists()


def test_build_writes_complete_fingerprinted_split_caches(tmp_path: Path) -> None:
    request = _request(tmp_path)
    train_dataset = _VectorDataset([0, 1, 2, 3])
    test_dataset = _VectorDataset([0, 1, 2, 3])
    loaded_names: list[str] = []
    previous_model: weakref.ReferenceType[nn.Module] | None = None

    def local_loader(name: str, **kwargs: object) -> nn.Module:
        nonlocal previous_model
        assert previous_model is None or previous_model() is None
        assert kwargs["source_root"] == request.source_root
        assert kwargs["checkpoint"] == request.checkpoint_paths[name]
        assert str(kwargs["device"]) == "cpu"
        loaded_names.append(name)
        model = _LocalExpert(shift=0 if name == "model_0" else 1)
        previous_model = weakref.ref(model)
        return model

    report = build_prediction_caches(
        request,
        _splits(),
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_loader=local_loader,
    )

    assert loaded_names == ["model_0", "model_1"]
    assert {result.status for result in report.results} == {"written"}
    for result in report.results:
        payload = torch.load(result.path, map_location="cpu", weights_only=True)
        assert payload["schema_version"] == CACHE_SCHEMA_VERSION
        assert payload["fingerprint"] == result.fingerprint
        assert payload["model_order"] == ["model_0", "model_1"]
        assert payload["features"].shape == (result.samples, 10)
        assert payload["expert_predictions"].shape == (result.samples, 2)
        assert payload["targets"].shape == (result.samples,)
        assert payload["sample_indices"].shape == (result.samples,)
        assert payload["route_labels"].tolist() == [0] * result.samples


def test_matching_existing_caches_are_reused_without_model_execution(tmp_path: Path) -> None:
    request = _request(tmp_path)
    build_prediction_caches(
        request,
        _splits(),
        train_dataset=_VectorDataset([0, 1, 2, 3]),
        test_dataset=_VectorDataset([0, 1, 2, 3]),
        model_loader=lambda name, **_kwargs: _LocalExpert(
            shift=0 if name == "model_0" else 1
        ),
    )

    def forbidden_loader(*args: object, **kwargs: object) -> nn.Module:
        raise AssertionError((args, kwargs))

    report = build_prediction_caches(
        request, _splits(), model_loader=forbidden_loader
    )
    assert {result.status for result in report.results} == {"reused"}


def test_fingerprint_mismatch_fails_before_model_execution(tmp_path: Path) -> None:
    request = _request(tmp_path)
    destination = cache_path(request.output_directory, "dispatcher_train")
    save_cache_atomically({"fingerprint": "stale"}, destination)
    original = destination.read_bytes()
    calls: list[str] = []

    def forbidden_loader(*args: object, **kwargs: object) -> nn.Module:
        calls.append("called")
        raise AssertionError((args, kwargs))

    with pytest.raises(CacheFingerprintMismatch, match="fingerprint mismatch"):
        build_prediction_caches(
            request,
            _splits(),
            train_dataset=_VectorDataset([0, 1, 2, 3]),
            test_dataset=_VectorDataset([0, 1, 2, 3]),
            model_loader=forbidden_loader,
        )

    assert calls == []
    assert destination.read_bytes() == original


def test_atomic_save_does_not_leave_partial_files(tmp_path: Path) -> None:
    destination = tmp_path / "cache.pt"
    save_cache_atomically({"fingerprint": "first"}, destination)
    payload = torch.load(destination, map_location="cpu", weights_only=True)
    assert payload["fingerprint"] == "first"
    assert not list(tmp_path.glob("*.part"))


def test_atomic_save_failure_preserves_existing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "cache.pt"
    save_cache_atomically({"fingerprint": "original"}, destination)
    original = destination.read_bytes()

    def fail_save(*args: object, **kwargs: object) -> None:
        raise RuntimeError((args, kwargs))

    monkeypatch.setattr("pertinence.cache.torch.save", fail_save)
    with pytest.raises(RuntimeError):
        save_cache_atomically({"fingerprint": "replacement"}, destination)

    assert destination.read_bytes() == original
    assert not list(tmp_path.glob("*.part"))
