"""Compatibility modules keep documented commands and historical imports stable."""

from pertinence import bundle_data, pipeline_artifacts, pipeline_cli, serving_cli
from pertinence.pipeline import artifacts, bundles, cli
from pertinence.serving import cli as serving_commands


def test_pipeline_compatibility_exports_delegate_to_package() -> None:
    assert bundle_data.package_dataset is bundles.package_dataset
    assert pipeline_artifacts.artifact_sha256 is artifacts.artifact_sha256
    assert pipeline_cli.main is cli.main


def test_serving_compatibility_exports_delegate_to_package() -> None:
    assert serving_cli.add_commands is serving_commands.add_commands
    assert serving_cli.run_command is serving_commands.run_command
