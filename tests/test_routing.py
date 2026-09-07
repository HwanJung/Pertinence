import pytest

torch = pytest.importorskip("torch")

from pertinence.routing import (  # noqa: E402
    cheapest_correct_labels,
    class_weights,
    directional_penalty_matrix,
    paper_penalty_loss,
)


def test_cheapest_correct_and_no_correct_fallback() -> None:
    predictions = torch.tensor([[1, 1, 2], [0, 2, 3], [0, 0, 0]])
    targets = torch.tensor([1, 2, 9])
    assert cheapest_correct_labels(predictions, targets).tolist() == [0, 1, 0]


@pytest.mark.parametrize("scheme", ["INS", "ISNS", "ENS"])
def test_class_weights_favor_minority(scheme) -> None:
    labels = torch.tensor([0, 0, 0, 0, 1])
    weights = class_weights(labels, 2, scheme)
    assert weights[1] > weights[0]
    assert weights.mean().item() == pytest.approx(1.0)


@pytest.mark.parametrize("scheme", ["INS", "ISNS", "ENS"])
def test_class_weights_zero_unobserved_classes_and_normalize_observed(scheme) -> None:
    labels = torch.tensor([0, 0, 0, 1])
    weights = class_weights(labels, 3, scheme)

    assert weights[2].item() == 0.0
    assert weights[:2].mean().item() == pytest.approx(1.0)
    assert torch.isfinite(weights).all()


def test_paper_loss_has_zero_contribution_for_correct_argmax() -> None:
    logits = torch.tensor([[4.0, 0.0], [3.0, 1.0]], requires_grad=True)
    targets = torch.tensor([0, 1])
    penalty = torch.tensor([[0.0, 0.1], [10.0, 0.0]])
    loss = paper_penalty_loss(logits, targets, penalty)
    expected = torch.nn.functional.cross_entropy(logits[1:2], targets[1:2]) * 10 / 2
    assert loss.item() == pytest.approx(expected.item())


def test_directional_matrix_respects_cost_order() -> None:
    matrix = directional_penalty_matrix(3, underestimation=10, overestimation=0.1)
    torch.testing.assert_close(
        matrix, torch.tensor([[0, 0.1, 0.1], [10, 0, 0.1], [10, 10, 0]])
    )
