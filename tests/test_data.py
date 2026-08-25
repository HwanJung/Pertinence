import numpy as np

from pertinence.data import load_splits, make_splits, save_splits


def test_splits_are_disjoint_repeatable_and_round_trip(tmp_path) -> None:
    first = make_splits(8_000, 2_000, 7)
    second = make_splits(8_000, 2_000, 7)
    assert np.array_equal(first.ga_search, second.ga_search)
    assert not set(first.ga_search) & set(first.final_evaluation)

    path = tmp_path / "splits.json"
    save_splits(first, path)
    loaded = load_splits(path)
    assert loaded.fingerprint == first.fingerprint
