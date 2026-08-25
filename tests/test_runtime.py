import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from pertinence.routing import LinearDispatcher  # noqa: E402
from pertinence.runtime import DynamicExecutor  # noqa: E402


class _Expert(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(3 * 2 * 2, 2)
        nn.init.zeros_(self.classifier.weight)
        self.classifier.bias.data[:] = torch.tensor([offset, -offset])
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return self.classifier(self.flatten(value))


def test_runtime_executes_only_selected_expert_and_reuses_extractor() -> None:
    small = _Expert(1.0)
    large = _Expert(-2.0)
    dispatcher = LinearDispatcher(12, 2)
    with torch.no_grad():
        dispatcher.linear.weight.zero_()
        dispatcher.linear.bias[:] = torch.tensor([-1.0, 1.0])
    executor = DynamicExecutor(
        {"small": small, "large": large}, ("small", "large"), "small", dispatcher
    )
    result = executor(torch.zeros(3, 3, 2, 2))
    assert result.routes.tolist() == [1, 1, 1]
    assert small.calls == 1  # feature extraction, not a second expert pass
    assert large.calls == 1
    assert result.logits.tolist() == [[-2.0, 2.0]] * 3

