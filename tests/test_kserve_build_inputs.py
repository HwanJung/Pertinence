from __future__ import annotations

import json
from pathlib import Path

import pytest

from cloud.kserve.scripts import build_inputs


def _bundle(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    root = tmp_path / "build-bundle"
    payloads: dict[str, bytes] = {}
    for index, relative in enumerate(sorted(build_inputs.REQUIRED_PATHS), 1):
        payload = f"payload-{index}-{relative}".encode()
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        payloads[relative] = payload
    return root, payloads


def _versions() -> dict[str, tuple[str, int]]:
    prefix = "gs://pertinence-509116-pertinence-runs/pertinence/build-inputs/build-1"
    return {
        relative: (f"{prefix}/{relative}", index)
        for index, relative in enumerate(sorted(build_inputs.REQUIRED_PATHS), 1)
    }


def _manifest_bytes(value: dict[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def test_seal_and_fetch_exact_generations(tmp_path: Path) -> None:
    bundle, payloads = _bundle(tmp_path)
    versions = _versions()
    manifest = build_inputs.seal_from_bundle(bundle, versions)
    manifest_payload = _manifest_bytes(manifest)
    manifest_uri = (
        "gs://pertinence-509116-pertinence-runs/pertinence/build-inputs/"
        "build-1/build-input-manifest.json#91"
    )
    requested: list[str] = []

    def fetch(source: str, destination: Path) -> None:
        requested.append(source)
        if source == manifest_uri:
            destination.write_bytes(manifest_payload)
            return
        for relative, (uri, generation) in versions.items():
            if source == f"{uri}#{generation}":
                destination.write_bytes(payloads[relative])
                return
        raise AssertionError(f"unexpected fetch: {source}")

    manifest_output = tmp_path / "evidence" / "build-input-manifest.json"
    manifest_output.parent.mkdir()
    output = tmp_path / "input"
    receipt = tmp_path / "evidence" / "build-input-receipt.json"
    result = build_inputs.fetch_inputs(
        manifest_uri,
        manifest_output,
        output,
        receipt,
        fetch=fetch,
    )

    assert requested == [manifest_uri, *(f"{uri}#{generation}" for uri, generation in versions.values())]
    assert result["valid"] is True
    assert result["manifest"]["generation"] == 91
    assert result["manifest"]["fingerprint"] == manifest["fingerprint"]
    assert json.loads(receipt.read_text()) == result
    assert manifest_output.read_bytes() == manifest_payload
    assert {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    } == payloads


def test_manifest_rejects_missing_extra_or_unsafe_inputs(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)
    versions = _versions()
    missing = dict(versions)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="exactly match"):
        build_inputs.seal_from_bundle(bundle, missing)

    manifest = build_inputs.seal_from_bundle(bundle, versions)
    body = {key: value for key, value in manifest.items() if key != "fingerprint"}
    body["objects"][0]["path"] = "../escape"
    changed = {**body, "fingerprint": build_inputs._canonical_hash(body)}
    with pytest.raises(ValueError, match="unsafe build input path"):
        build_inputs.validate_manifest(changed)


def test_seal_rejects_symlink_inside_bundle(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)
    (bundle / "unapproved-link").symlink_to(bundle / "evaluated" / "manifest.json")

    with pytest.raises(ValueError, match="must not contain symlinks"):
        build_inputs.seal_from_bundle(bundle, _versions())


@pytest.mark.parametrize(
    "manifest_uri",
    [
        "gs://bucket/manifest.json",
        "gs://bucket/manifest.json#0",
        "https://storage.googleapis.com/bucket/manifest.json#1",
        "gs://bucket/../manifest.json#1",
    ],
)
def test_fetch_rejects_unpinned_or_unsafe_manifest_uri(
    tmp_path: Path, manifest_uri: str
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    with pytest.raises(ValueError):
        build_inputs.fetch_inputs(
            manifest_uri,
            evidence / "manifest.json",
            tmp_path / "input",
            evidence / "receipt.json",
            fetch=lambda *_: pytest.fail("invalid manifest URI reached the transport"),
        )


def test_fetch_rejects_downloaded_byte_mismatch_and_keeps_outputs_absent(
    tmp_path: Path,
) -> None:
    bundle, payloads = _bundle(tmp_path)
    versions = _versions()
    manifest = build_inputs.seal_from_bundle(bundle, versions)
    manifest_uri = "gs://bucket/manifests/build-inputs.json#2"
    first_object = manifest["objects"][0]

    def fetch(source: str, destination: Path) -> None:
        if source == manifest_uri:
            destination.write_bytes(_manifest_bytes(manifest))
        elif source == f"{first_object['uri']}#{first_object['generation']}":
            destination.write_bytes(b"tampered")
        else:
            relative = next(
                path
                for path, (uri, generation) in versions.items()
                if source == f"{uri}#{generation}"
            )
            destination.write_bytes(payloads[relative])

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    manifest_output = evidence / "manifest.json"
    output = tmp_path / "input"
    receipt = evidence / "receipt.json"
    with pytest.raises(ValueError, match="downloaded bytes differ"):
        build_inputs.fetch_inputs(
            manifest_uri, manifest_output, output, receipt, fetch=fetch
        )
    assert not manifest_output.exists()
    assert not output.exists()
    assert not receipt.exists()


def test_cloudbuild_uses_only_manifested_generation_pinned_inputs() -> None:
    cloudbuild = Path("cloud/kserve/cloudbuild.yaml").read_text(encoding="utf-8")

    assert "_BUILD_INPUT_MANIFEST_URI" in cloudbuild
    assert "build_inputs.py fetch" in cloudbuild
    assert "build-input-receipt.json" in cloudbuild
    assert "build-input-manifest.json" in cloudbuild
    assert "--recursive" not in cloudbuild
    assert "_BUNDLE_URI" not in cloudbuild
