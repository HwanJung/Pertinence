"""Repository-wide ownership boundaries remain explicit."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_and_serving_implementations_are_packaged() -> None:
    pipeline = ROOT / "src" / "pertinence" / "pipeline"
    serving_cli = ROOT / "src" / "pertinence" / "serving" / "cli"
    assert {
        "artifacts.py",
        "bootstrap.py",
        "bundles.py",
        "cache.py",
        "cli.py",
        "registry.py",
        "search.py",
        "validation.py",
    } <= {path.name for path in pipeline.glob("*.py")}
    assert {"benchmark.py", "common.py", "promotion.py", "runtime.py", "trace.py"} <= {
        path.name for path in serving_cli.glob("*.py")
    }


def test_all_tests_are_owned_by_root_test_suite() -> None:
    assert not list((ROOT / "pipelines" / "pertinence" / "tests").glob("test_*.py"))


def test_current_execution_docs_use_dataset_scoped_artifacts() -> None:
    current_docs = (
        "README.md",
        "pipelines/pertinence/README.md",
        "cloud/docs/runbook.ko.md",
        "cloud/kserve/README.md",
        "cloud/docs/kserve-benchmark-runbook.ko.md",
        "cloud/docs/kserve-benchmark-execution.ko.md",
    )
    forbidden = (
        "artifacts/pipeline/",
        "artifacts/cifar10/portable/",
        "artifacts/kserve-session-",
        "artifacts/datasets/",
        "artifacts/checkpoints/",
        "artifacts/sources/",
    )
    for relative_path in current_docs:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for path in forbidden:
            assert path not in text, f"{relative_path}: legacy artifact path {path}"


def test_project_operator_policy_is_versionable() -> None:
    policy = ROOT / ".codex" / "agents" / "gke-operator.toml"
    assert policy.is_file()
    text = policy.read_text(encoding="utf-8")
    assert "pertinence-509116" in text
    assert "pertinence-dispatcher-260919-1708" in text
