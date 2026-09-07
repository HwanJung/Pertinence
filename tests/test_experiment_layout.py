from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_specific_files_are_partitioned_by_experiment() -> None:
    experiments = ROOT / "experiments"
    for dataset in ("cifar10", "cifar100"):
        experiment = experiments / dataset
        assert experiment.is_dir()
        for directory in ("configs", "data", "docs", "figures", "scripts"):
            assert (experiment / directory).is_dir()

    for legacy_directory in ("configs", "data", "figures"):
        assert not (ROOT / legacy_directory).exists()


def test_cifar10_config_uses_its_dataset_artifact_root() -> None:
    config = (ROOT / "experiments" / "cifar10" / "configs" / "experiment.yaml").read_text(
        encoding="utf-8"
    )
    assert "artifacts/cifar10" in config
    assert "artifacts/cifar100" not in config
