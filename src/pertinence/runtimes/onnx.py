"""ONNX inference with declarative RGB resize/normalization only."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from ..pipeline.artifacts import safe_path, verify_artifact


def _iter_message_children(message: object) -> Iterator[object]:
    """Yield nested protobuf messages without relying on deprecated descriptor fields."""
    for field, value in message.ListFields():
        if field.message_type is None:
            continue
        is_repeated = getattr(field, "is_repeated", None)
        if is_repeated is None:
            is_repeated = field.label == field.LABEL_REPEATED
        if is_repeated:
            yield from value
        else:
            yield value


class ExpertRuntime:
    def __init__(
        self,
        root: Path,
        expert: dict,
        *,
        device: str = "cpu",
        intra_op_num_threads: int = 1,
    ) -> None:
        import onnx
        import onnxruntime as ort

        self.expert = expert
        path = safe_path(root, expert["model_path"])
        verify_artifact(path, expert["model_sha256"])
        model = onnx.load(path, load_external_data=False)

        # Reject external tensor data and custom domains in nested graphs/functions too.
        def check_message(message) -> None:
            if isinstance(message, onnx.TensorProto) and (
                message.data_location == onnx.TensorProto.EXTERNAL or message.external_data
            ):
                raise ValueError("ONNX external tensor data is unsupported; export a single file")
            if isinstance(message, onnx.NodeProto) and message.domain not in ("", "ai.onnx"):
                raise ValueError("ONNX custom operator domains are unsupported")
            for child in _iter_message_children(message):
                check_message(child)

        check_message(model)
        if model.functions:
            raise ValueError("ONNX local functions are unsupported in runtime v1")
        onnx.checker.check_model(model)
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise ValueError("CUDA ONNX Runtime provider is not installed")
            providers.insert(0, "CUDAExecutionProvider")
        if type(intra_op_num_threads) is not int or intra_op_num_threads < 1:
            raise ValueError("ONNX Runtime intra-op thread count must be a positive integer")
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_num_threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model.SerializeToString(), options, providers=providers)
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)" or len(inputs[0].shape) != 4:
            raise ValueError("ONNX expert must accept one rank-4 float32 input")
        self.input_name = inputs[0].name
        declared = expert["input"]["shape"]
        if isinstance(inputs[0].shape[0], int):
            raise ValueError("ONNX expert must support a dynamic batch dimension")
        if any(
            isinstance(actual, int) and actual != expected
            for actual, expected in zip(inputs[0].shape[1:], declared)
        ):
            raise ValueError("ONNX input shape differs from expert metadata")
        outputs = {output.name for output in self.session.get_outputs()}
        if not set(expert["outputs"].values()).issubset(outputs):
            raise ValueError("declared ONNX output names are missing")

    def predict(self, images: list[Image.Image], *, features: bool = False):
        preprocessing = self.expert["input"]["preprocessing"]
        height, width = preprocessing["resize"]
        mean = np.asarray(preprocessing["mean"], dtype=np.float32)[:, None, None]
        std = np.asarray(preprocessing["std"], dtype=np.float32)[:, None, None]
        inputs = np.stack(
            [
                (
                    np.asarray(
                        image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32
                    ).transpose(2, 0, 1)
                    / 255.0
                    - mean
                )
                / std
                for image in images
            ]
        )
        names = [self.expert["outputs"]["logits"]]
        if features:
            if "features" not in self.expert["outputs"]:
                raise ValueError("feature extractor must declare features output")
            names.append(self.expert["outputs"]["features"])
        result = self.session.run(names, {self.input_name: inputs})
        logits = result[0]
        if (
            logits.shape != (len(images), self.expert["num_classes"])
            or logits.dtype.kind != "f"
            or not np.isfinite(logits).all()
        ):
            raise ValueError("ONNX logits must be finite [batch, num_classes] floats")
        feature_values = result[1] if features else None
        if features and (
            feature_values.ndim != 2
            or feature_values.shape[0] != len(images)
            or feature_values.shape[1] == 0
            or feature_values.dtype.kind != "f"
            or not np.isfinite(feature_values).all()
        ):
            raise ValueError("ONNX features must be finite [batch, feature_dim] floats")
        return logits.argmax(axis=1).astype(np.int64), feature_values
