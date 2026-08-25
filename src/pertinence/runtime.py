"""Online dynamic execution with extractor/expert computation sharing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import torch
from torch import Tensor, nn

from .experts import FrozenFeatureExtractor
from .routing import LinearDispatcher


class DynamicOutput(NamedTuple):
    logits: Tensor
    routes: Tensor


class DynamicExecutor(nn.Module):
    """Dispatch each input to exactly one frozen expert at runtime.

    The feature extractor is also an expert. Its classifier logits are produced
    while extracting features and are reused for samples routed back to it.
    Other experts receive only their selected batch subset.
    """

    def __init__(
        self,
        experts: Mapping[str, nn.Module],
        model_order: Sequence[str],
        feature_extractor: str,
        dispatcher: LinearDispatcher,
    ) -> None:
        super().__init__()
        order = tuple(model_order)
        if not order or len(set(order)) != len(order):
            raise ValueError("model_order must contain unique experts")
        if set(experts) != set(order):
            raise ValueError("experts and model_order must contain the same names")
        if feature_extractor not in experts:
            raise ValueError("feature extractor is missing from experts")
        if dispatcher.linear.out_features != len(order):
            raise ValueError("dispatcher output width must equal number of experts")

        self.model_order = order
        self.feature_extractor_name = feature_extractor
        self.feature_extractor_index = order.index(feature_extractor)
        self.experts = nn.ModuleDict({name: experts[name] for name in order})
        for expert in self.experts.values():
            expert.requires_grad_(False)
            expert.eval()
        self.extractor = FrozenFeatureExtractor(self.experts[feature_extractor])
        if dispatcher.linear.in_features != self.extractor.feature_dim:
            raise ValueError("dispatcher input width must equal extractor feature width")
        self.dispatcher = dispatcher.eval()

    def train(self, mode: bool = True) -> DynamicExecutor:
        # Runtime is inference-only even if a parent module calls train().
        super().train(False)
        self.dispatcher.eval()
        for expert in self.experts.values():
            expert.eval()
        return self

    def forward(self, inputs: Tensor) -> DynamicOutput:
        if inputs.ndim != 4:
            raise ValueError("inputs must be a batch [N, C, H, W]")
        with torch.no_grad():
            extracted = self.extractor(inputs)
            routes = self.dispatcher(extracted.features).argmax(dim=1)
            output = torch.empty_like(extracted.logits)
            for expert_index, name in enumerate(self.model_order):
                selected = torch.where(routes == expert_index)[0]
                if selected.numel() == 0:
                    continue
                if expert_index == self.feature_extractor_index:
                    logits = extracted.logits.index_select(0, selected)
                else:
                    logits = self.experts[name](inputs.index_select(0, selected))
                if not isinstance(logits, Tensor) or logits.shape[1:] != output.shape[1:]:
                    raise RuntimeError(f"expert {name!r} returned incompatible logits")
                output.index_copy_(0, selected, logits)
        return DynamicOutput(logits=output, routes=routes)

