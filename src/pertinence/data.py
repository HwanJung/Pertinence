"""CIFAR-10 loading and deterministic dispatcher/search splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .reproducibility import canonical_json_hash


@dataclass(frozen=True)
class SplitIndices:
    dispatcher_train: np.ndarray
    ga_search: np.ndarray
    final_evaluation: np.ndarray
    seed: int

    @property
    def fingerprint(self) -> str:
        return canonical_json_hash(
            {
                "seed": self.seed,
                "dispatcher_train": self.dispatcher_train.tolist(),
                "ga_search": self.ga_search.tolist(),
                "final_evaluation": self.final_evaluation.tolist(),
            }
        )


def make_splits(search_size: int, final_size: int, seed: int) -> SplitIndices:
    if search_size != 8_000 or final_size != 2_000:
        raise ValueError("the fixed reproduction split is GA search 8,000 + final 2,000")
    test_indices = np.random.default_rng(seed).permutation(10_000)
    return SplitIndices(
        dispatcher_train=np.arange(50_000, dtype=np.int64),
        ga_search=test_indices[:search_size],
        final_evaluation=test_indices[search_size:],
        seed=seed,
    )


def save_splits(splits: SplitIndices, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset": "CIFAR10",
        "official_train_size": 50_000,
        "official_test_size": 10_000,
        "seed": splits.seed,
        "dispatcher_train": splits.dispatcher_train.tolist(),
        "ga_search": splits.ga_search.tolist(),
        "final_evaluation": splits.final_evaluation.tolist(),
        "fingerprint": splits.fingerprint,
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_splits(path: str | Path) -> SplitIndices:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = SplitIndices(
        dispatcher_train=np.asarray(payload["dispatcher_train"], dtype=np.int64),
        ga_search=np.asarray(payload["ga_search"], dtype=np.int64),
        final_evaluation=np.asarray(payload["final_evaluation"], dtype=np.int64),
        seed=int(payload["seed"]),
    )
    if payload.get("fingerprint") != splits.fingerprint:
        raise ValueError("split manifest fingerprint mismatch")
    _validate_splits(splits)
    return splits


def _validate_splits(splits: SplitIndices) -> None:
    if not np.array_equal(splits.dispatcher_train, np.arange(50_000)):
        raise ValueError("dispatcher training must preserve the full official train ordering")
    search = set(map(int, splits.ga_search))
    final = set(map(int, splits.final_evaluation))
    if search & final:
        raise ValueError("ga_search and final_evaluation overlap")
    if search | final != set(range(10_000)):
        raise ValueError("ga_search and final_evaluation must cover official test set")
    if len(search) != 8_000 or len(final) != 2_000:
        raise ValueError("expected exactly 8,000 GA-search and 2,000 final samples")


def evaluation_transform(mean: list[float], std: list[float]):
    """Return the exact chenyaofo validation transform (no augmentation)."""
    from torchvision.transforms import Compose, Normalize, ToTensor

    return Compose([ToTensor(), Normalize(tuple(mean), tuple(std))])


def load_cifar10(root: str | Path, train: bool, transform):
    """Load already prepared CIFAR-10 data without implicit network access."""
    from torchvision.datasets import CIFAR10

    return CIFAR10(root=str(root), train=train, transform=transform, download=False)
