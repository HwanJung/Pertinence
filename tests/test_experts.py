from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from pertinence.experts import (
    EXPERT_REGISTRY,
    FrozenFeatureExtractor,
    checkpoint_path,
    get_expert_spec,
    load_expert,
    load_plain_state_dict,
)


EXPECTED_NAMES = (
    "shufflenetv2_x0_5",
    "mobilenetv2_x0_5",
    "shufflenetv2_x1_0",
    "mobilenetv2_x0_75",
    "mobilenetv2_x1_0",
    "resnet44",
    "resnet56",
    "repvgg_a0",
    "repvgg_a1",
    "repvgg_a2",
)


def _write_fake_source(source_root: Path, filename: str, hub_entry: str) -> None:
    package = source_root / "pytorch_cifar_models"
    package.mkdir(parents=True, exist_ok=True)
    (package / filename).write_text(
        "\n".join(
            (
                "from functools import partial",
                "from torch import nn",
                "class FakeExpert(nn.Module):",
                "    def __init__(self):",
                "        super().__init__()",
                "        self.flatten = nn.Flatten()",
                "        self.hidden = nn.Linear(12, 5)",
                "        self.relu = nn.ReLU()",
                "        self.classifier = nn.Linear(5, 10)",
                "        self.conversion_called = False",
                "        self.deploy = False",
                "    def forward(self, value):",
                "        return self.classifier(self.relu(self.hidden(self.flatten(value))))",
                "    def convert_to_inference_model(self, *args, **kwargs):",
                "        self.conversion_called = True",
                "        raise AssertionError('RepVGG deploy conversion is forbidden')",
                "def make_model(*args, pretrained=False, **kwargs):",
                "    if pretrained:",
                "        raise AssertionError('pretrained=True would access the network')",
                "    return FakeExpert()",
                f"{hub_entry} = partial(make_model)",
            )
        ),
        encoding="utf-8",
    )


def _make_local_assets(tmp_path: Path, name: str) -> tuple[Path, Path, nn.Module]:
    spec = get_expert_spec(name)
    source_root = tmp_path / "source"
    _write_fake_source(source_root, spec.source_module, spec.hub_entry)

    # Import through the production builder once to obtain a structurally exact
    # local state_dict, then save only that dictionary as the release format does.
    from pertinence.experts import build_expert

    original = build_expert(name, source_root=source_root)
    checkpoint = tmp_path / spec.checkpoint_filename
    torch.save(original.state_dict(), checkpoint)
    return source_root, checkpoint, original


def test_registry_is_exactly_the_cost_sorted_pareto_front() -> None:
    assert tuple(spec.name for spec in EXPERT_REGISTRY) == EXPECTED_NAMES
    assert len({spec.name for spec in EXPERT_REGISTRY}) == 10
    assert [spec.madds_m for spec in EXPERT_REGISTRY] == sorted(
        spec.madds_m for spec in EXPERT_REGISTRY
    )
    assert all(spec.hub_entry == f"cifar10_{spec.name}" for spec in EXPERT_REGISTRY)
    assert all(
        spec.checkpoint_url.startswith("https://github.com/chenyaofo/")
        for spec in EXPERT_REGISTRY
    )
    assert all(spec.checkpoint_filename.endswith(".pt") for spec in EXPERT_REGISTRY)


def test_catalog_endpoints_and_metrics_are_pinned() -> None:
    cheapest = get_expert_spec("shufflenetv2_x0_5")
    most_accurate = get_expert_spec("repvgg_a2")
    assert (cheapest.top1_accuracy_pct, cheapest.madds_m, cheapest.params_m) == (
        90.13,
        10.90,
        0.35,
    )
    assert cheapest.checkpoint_filename == "cifar10_shufflenetv2_x0_5-1308b4e9.pt"
    assert (most_accurate.top1_accuracy_pct, most_accurate.madds_m) == (94.98, 1850.10)
    assert most_accurate.mflops == 3700.20


def test_checkpoint_path_uses_registry_filename(tmp_path: Path) -> None:
    result = checkpoint_path(tmp_path, "resnet44")
    assert result == tmp_path / "cifar10_resnet44-2a3cabcb.pt"


def test_unknown_expert_has_actionable_error() -> None:
    with pytest.raises(KeyError, match="unknown CIFAR-10 expert"):
        get_expert_spec("resnet8")


def test_plain_state_dict_loader_rejects_wrapped_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "wrapped.pt"
    torch.save({"state_dict": {"weight": torch.ones(1)}}, checkpoint)
    with pytest.raises(TypeError, match="map parameter names directly to tensors"):
        load_plain_state_dict(checkpoint)


def test_load_expert_uses_only_explicit_local_assets(tmp_path: Path) -> None:
    source_root, checkpoint, original = _make_local_assets(tmp_path, "resnet44")
    loaded = load_expert("resnet44", source_root=source_root, checkpoint=checkpoint)

    assert loaded.training is False
    assert all(parameter.requires_grad is False for parameter in loaded.parameters())
    for expected, actual in zip(original.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(expected, actual)


def test_repvgg_is_not_converted_to_deploy_form(tmp_path: Path) -> None:
    source_root, checkpoint, _ = _make_local_assets(tmp_path, "repvgg_a0")
    loaded = load_expert("repvgg_a0", source_root=source_root, checkpoint=checkpoint)
    assert loaded.deploy is False
    assert loaded.conversion_called is False


class _TwoLinearExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(12, 7)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(7, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.hidden(self.flatten(inputs)))
        return self.classifier(hidden)


def test_frozen_adapter_returns_logits_and_final_linear_input_together() -> None:
    torch.manual_seed(7)
    expert = _TwoLinearExpert()
    adapter = FrozenFeatureExtractor(expert)
    inputs = torch.randn(4, 3, 2, 2)

    expected_features = expert.activation(expert.hidden(expert.flatten(inputs))).detach()
    expected_logits = expert.classifier(expected_features).detach()
    output = adapter(inputs)

    assert adapter.feature_dim == 7
    assert torch.equal(output.features, expected_features)
    assert torch.equal(output.logits, expected_logits)
    assert output.features.requires_grad is False
    assert output.logits.requires_grad is False
    assert all(parameter.requires_grad is False for parameter in adapter.parameters())


def test_adapter_keeps_expert_in_eval_mode_when_dispatcher_trains() -> None:
    expert = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(12, 3))
    adapter = FrozenFeatureExtractor(expert)
    adapter.train()
    assert adapter.training is True
    assert adapter.expert.training is False


def test_adapter_requires_a_linear_classifier() -> None:
    with pytest.raises(ValueError, match="at least one nn.Linear"):
        FrozenFeatureExtractor(nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU()))
