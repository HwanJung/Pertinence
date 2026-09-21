from __future__ import annotations

import json

import numpy as np
import pytest

from pertinence.serving.trace import create_trace, verify_trace


def _arguments() -> dict[str, object]:
    return {
        "trace_id": "cifar10-final-2000",
        "dataset_id": "cifar10",
        "dataset_version": "1",
        "images": np.zeros((2, 3, 32, 32), dtype=np.float32),
        "labels": np.asarray([0, 1], dtype=np.int64),
        "sample_ids": ["sample-0", "sample-1"],
        "source_objects": [
            {"uri": "gs://bucket/data", "sha256": "a" * 64, "size_bytes": 1, "generation": 3}
        ],
        "provenance": "official CIFAR-10 test bundle",
        "usage_conditions": "CIFAR-10 upstream terms apply",
    }


def test_trace_is_reproducible_and_idempotent(tmp_path) -> None:
    output = tmp_path / "trace"
    first = create_trace(output, **_arguments())
    second = create_trace(output, **_arguments())
    verified, images, labels, sample_ids = verify_trace(output)

    assert first == second == verified
    assert images.shape == (2, 3, 32, 32)
    assert labels.tolist() == [0, 1]
    assert sample_ids == ["sample-0", "sample-1"]
    assert "labels" not in first
    assert "label_sha256" in first


def test_trace_conflict_is_rejected(tmp_path) -> None:
    output = tmp_path / "trace"
    create_trace(output, **_arguments())
    changed = _arguments()
    changed["labels"] = np.asarray([1, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="different content"):
        create_trace(output, **changed)


def test_trace_payload_tampering_is_rejected(tmp_path) -> None:
    output = tmp_path / "trace"
    create_trace(output, **_arguments())
    manifest = json.loads((output / "trace.json").read_text())
    manifest["samples"] = 3
    (output / "trace.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_trace(output)
