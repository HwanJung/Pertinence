from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request

import pytest

from pertinence.assets import (
    AssetIntegrityError,
    ManifestValidationError,
    PARETO_MODEL_NAMES,
    load_asset_manifest,
    parse_asset_manifest,
    prepare_assets,
    write_report_atomically,
)


SOURCE_REVISION = "1234567890abcdef1234567890abcdef12345678"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _payload(content_by_name: dict[str, bytes] | None = None) -> dict[str, object]:
    contents = content_by_name or {}

    def entry(kind: str, name: str, path: str) -> dict[str, str]:
        content = contents.get(name, f"content:{name}".encode())
        suffix = f"/{SOURCE_REVISION}.tar.gz" if kind == "source" else f"/{name}.bin"
        result = {
            "kind": kind,
            "name": name,
            "url": f"https://assets.example.test{suffix}",
            "sha256": _digest(content),
            "path": path,
        }
        if kind == "source":
            result["revision"] = SOURCE_REVISION
        return result

    assets = [
        entry("dataset", "cifar10_python_tar", "datasets/cifar-10-python.tar.gz"),
        entry(
            "source",
            "chenyaofo_pytorch_cifar_models",
            "sources/pytorch-cifar-models.tar.gz",
        ),
    ]
    assets.extend(
        entry("checkpoint", name, f"checkpoints/{name}.pt")
        for name in PARETO_MODEL_NAMES
    )
    return {"schema_version": 1, "assets": assets}


def test_manifest_requires_exact_pareto_inventory() -> None:
    payload = _payload()
    payload["assets"] = payload["assets"][:-1]  # type: ignore[index]

    with pytest.raises(ManifestValidationError, match="exactly the 10 Pareto"):
        parse_asset_manifest(payload)


@pytest.mark.parametrize("sha256", ["", "TODO", "c58f30108f718f92721af3b95e74349a", "0" * 64])
def test_manifest_rejects_missing_placeholder_or_non_sha256_hash(sha256: str) -> None:
    payload = _payload()
    payload["assets"][0]["sha256"] = sha256  # type: ignore[index]

    with pytest.raises(ManifestValidationError, match="SHA-256|placeholder"):
        parse_asset_manifest(payload)


def test_source_must_be_pinned_to_full_commit_in_url() -> None:
    payload = _payload()
    payload["assets"][1]["url"] = "https://example.test/archive/main.tar.gz"  # type: ignore[index]

    with pytest.raises(ManifestValidationError, match="immutable revision"):
        parse_asset_manifest(payload)


def test_load_manifest_is_local_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "assets.json"
    manifest_path.write_text(json.dumps(_payload()), encoding="utf-8")

    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("manifest loading must not access the network")

    monkeypatch.setattr("urllib.request.urlopen", fail_network)
    manifest = load_asset_manifest(manifest_path)

    assert len(manifest.assets) == 12


def test_load_yaml_manifest(tmp_path: Path) -> None:
    import yaml

    manifest_path = tmp_path / "assets.yaml"
    manifest_path.write_text(yaml.safe_dump(_payload()), encoding="utf-8")

    manifest = load_asset_manifest(manifest_path)

    assert len(manifest.assets) == 12


def test_dry_run_does_not_touch_filesystem_or_network(tmp_path: Path) -> None:
    manifest = parse_asset_manifest(_payload())
    artifact_root = tmp_path / "does-not-exist"

    def fail_network(request: Request, timeout: float) -> io.BytesIO:
        raise AssertionError((request, timeout))

    report = prepare_assets(manifest, artifact_root, dry_run=True, opener=fail_network)

    assert not artifact_root.exists()
    assert {asset.status for asset in report.assets} == {"planned"}


def test_verify_only_checks_existing_files_without_network(tmp_path: Path) -> None:
    contents = {name: f"content:{name}".encode() for name in PARETO_MODEL_NAMES}
    contents["cifar10_python_tar"] = b"content:cifar10_python_tar"
    contents["chenyaofo_pytorch_cifar_models"] = (
        b"content:chenyaofo_pytorch_cifar_models"
    )
    manifest = parse_asset_manifest(_payload(contents))
    for spec in manifest.assets:
        destination = tmp_path / spec.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents[spec.name])

    def fail_network(request: Request, timeout: float) -> io.BytesIO:
        raise AssertionError((request, timeout))

    report = prepare_assets(manifest, tmp_path, verify_only=True, opener=fail_network)

    assert {asset.status for asset in report.assets} == {"verified"}


def test_verify_only_reports_missing_asset_without_creating_root(tmp_path: Path) -> None:
    artifact_root = tmp_path / "missing"

    with pytest.raises(AssetIntegrityError, match="is missing"):
        prepare_assets(parse_asset_manifest(_payload()), artifact_root, verify_only=True)

    assert not artifact_root.exists()


class _Response(io.BytesIO):
    pass


def test_downloads_to_temp_verifies_then_atomically_renames(tmp_path: Path) -> None:
    payload = _payload()
    content_by_url = {
        asset["url"]: f"content:{asset['name']}".encode()
        for asset in payload["assets"]  # type: ignore[union-attr]
    }
    manifest = parse_asset_manifest(payload)
    called_urls: list[str] = []

    def opener(request: Request, timeout: float) -> _Response:
        assert timeout == 3.0
        called_urls.append(request.full_url)
        return _Response(content_by_url[request.full_url])

    report = prepare_assets(manifest, tmp_path, timeout_seconds=3.0, opener=opener)

    assert len(called_urls) == 12
    assert {asset.status for asset in report.assets} == {"downloaded"}
    assert not list(tmp_path.rglob("*.part"))
    for spec in manifest.assets:
        assert (tmp_path / spec.path).read_bytes() == content_by_url[spec.url]


def test_bad_download_keeps_existing_file_and_removes_temp(tmp_path: Path) -> None:
    payload = _payload()
    manifest = parse_asset_manifest(payload)
    first = manifest.assets[0]
    destination = tmp_path / first.path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"previous-corrupt-file")

    def bad_opener(request: Request, timeout: float) -> _Response:
        return _Response(b"wrong-download")

    with pytest.raises(AssetIntegrityError, match="failed SHA-256"):
        prepare_assets(manifest, tmp_path, opener=bad_opener)

    assert destination.read_bytes() == b"previous-corrupt-file"
    assert not list(destination.parent.glob("*.part"))


def test_report_is_written_as_json_without_partial_file(tmp_path: Path) -> None:
    report = prepare_assets(parse_asset_manifest(_payload()), tmp_path, dry_run=True)
    target = tmp_path / "receipt.json"

    write_report_atomically(report, target)

    assert json.loads(target.read_text(encoding="utf-8"))["mode"] == "dry-run"
    assert not list(tmp_path.glob("*.part"))
