"""Validated, explicit preparation of reproducible experiment assets.

Importing this module never performs network or filesystem I/O.  Call
``load_asset_manifest`` and ``prepare_assets`` explicitly from a command-line
entry point instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import tarfile
from typing import BinaryIO, Callable, Literal, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen


AssetKind = Literal["dataset", "checkpoint", "source"]
SCHEMA_VERSION = 1
DEFAULT_CHUNK_SIZE = 1024 * 1024

PARETO_MODEL_NAMES: tuple[str, ...] = (
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

DATASET_ASSET_NAME = "cifar10_python_tar"
SOURCE_ASSET_NAME = "chenyaofo_pytorch_cifar_models"
EXPECTED_INVENTORY: frozenset[tuple[str, str]] = frozenset(
    {("dataset", DATASET_ASSET_NAME), ("source", SOURCE_ASSET_NAME)}
    | {("checkpoint", model_name) for model_name in PARETO_MODEL_NAMES}
)

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
_PLACEHOLDER_MARKERS = ("todo", "replace", "changeme", "unknown", "<", "${")


class AssetError(RuntimeError):
    """Base class for asset preparation errors."""


class ManifestValidationError(AssetError, ValueError):
    """Raised before any download when the input manifest is unsafe or incomplete."""


class AssetIntegrityError(AssetError):
    """Raised when a local or downloaded asset does not match its SHA-256."""


@dataclass(frozen=True)
class AssetSpec:
    kind: AssetKind
    name: str
    url: str
    sha256: str
    path: str
    revision: str | None = None


@dataclass(frozen=True)
class AssetManifest:
    schema_version: int
    assets: tuple[AssetSpec, ...]


@dataclass(frozen=True)
class PreparedAsset:
    kind: AssetKind
    name: str
    path: str
    url: str
    expected_sha256: str
    actual_sha256: str | None
    size_bytes: int | None
    status: Literal["planned", "verified", "downloaded", "reused"]
    revision: str | None = None


@dataclass(frozen=True)
class PreparationReport:
    schema_version: int
    mode: Literal["dry-run", "verify-only", "prepare"]
    artifact_root: str
    assets: tuple[PreparedAsset, ...]
    generated_at_utc: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "artifact_root": self.artifact_root,
            "generated_at_utc": self.generated_at_utc,
            "assets": [asdict(asset) for asset in self.assets],
        }


def _require_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{context} must be a JSON object")
    return value


def _require_non_placeholder_string(
    value: object, field: str, asset_name: str
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(
            f"asset {asset_name!r} requires a non-empty {field!r} value"
        )
    result = value.strip()
    lowered = result.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        raise ManifestValidationError(
            f"asset {asset_name!r} has a placeholder instead of an explicit {field!r}"
        )
    return result


def _validate_url(value: object, asset_name: str) -> str:
    url = _require_non_placeholder_string(value, "url", asset_name)
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ManifestValidationError(
            f"asset {asset_name!r} URL must be an absolute HTTPS URL: {url!r}"
        )
    if parsed.username or parsed.password or parsed.fragment:
        raise ManifestValidationError(
            f"asset {asset_name!r} URL must not contain credentials or a fragment"
        )
    return url


def _validate_sha256(value: object, asset_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(
            f"asset {asset_name!r} requires an explicit SHA-256 digest"
        )
    digest = _require_non_placeholder_string(value, "sha256", asset_name).lower()
    if not _SHA256_PATTERN.fullmatch(digest) or len(set(digest)) == 1:
        raise ManifestValidationError(
            f"asset {asset_name!r} requires an explicit 64-hex SHA-256 digest; "
            "bootstrap, truncated, MD5, and placeholder hashes are not accepted"
        )
    return digest


def _validate_relative_path(value: object, asset_name: str) -> str:
    raw_path = _require_non_placeholder_string(value, "path", asset_name)
    if "\\" in raw_path:
        raise ManifestValidationError(
            f"asset {asset_name!r} path must use forward slashes"
        )
    path = PurePosixPath(raw_path)
    if path.is_absolute() or raw_path in {".", ".."} or ".." in path.parts:
        raise ManifestValidationError(
            f"asset {asset_name!r} path must remain below the artifact root: {raw_path!r}"
        )
    if any(part in {"", "."} for part in path.parts):
        raise ManifestValidationError(
            f"asset {asset_name!r} has a non-normalized path: {raw_path!r}"
        )
    return path.as_posix()


def _parse_asset(raw: object, index: int) -> AssetSpec:
    item = _require_mapping(raw, f"assets[{index}]")
    allowed_fields = {"kind", "name", "url", "sha256", "path", "revision"}
    unknown_fields = sorted(set(item) - allowed_fields)
    if unknown_fields:
        raise ManifestValidationError(
            f"assets[{index}] contains unsupported fields: {', '.join(unknown_fields)}"
        )

    name = _require_non_placeholder_string(item.get("name"), "name", f"#{index}")
    kind_value = item.get("kind")
    if kind_value not in {"dataset", "checkpoint", "source"}:
        raise ManifestValidationError(
            f"asset {name!r} kind must be dataset, checkpoint, or source"
        )
    kind: AssetKind = kind_value  # type: ignore[assignment]
    revision_value = item.get("revision")
    revision: str | None = None
    if kind == "source":
        revision = _require_non_placeholder_string(revision_value, "revision", name).lower()
        if not _REVISION_PATTERN.fullmatch(revision):
            raise ManifestValidationError(
                f"source asset {name!r} revision must be a full 40-hex commit ID"
            )
    elif revision_value is not None:
        raise ManifestValidationError(
            f"asset {name!r} may specify revision only when kind is source"
        )

    url = _validate_url(item.get("url"), name)
    if revision is not None and revision not in url.lower():
        raise ManifestValidationError(
            f"source asset {name!r} URL must contain its immutable revision {revision}"
        )
    return AssetSpec(
        kind=kind,
        name=name,
        url=url,
        sha256=_validate_sha256(item.get("sha256"), name),
        path=_validate_relative_path(item.get("path"), name),
        revision=revision,
    )


def parse_asset_manifest(payload: object) -> AssetManifest:
    """Validate and parse an in-memory JSON-compatible manifest payload."""

    manifest = _require_mapping(payload, "manifest")
    unknown_fields = sorted(set(manifest) - {"schema_version", "assets"})
    if unknown_fields:
        raise ManifestValidationError(
            f"manifest contains unsupported fields: {', '.join(unknown_fields)}"
        )
    version = manifest.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ManifestValidationError(
            f"manifest schema_version must be {SCHEMA_VERSION}, got {version!r}"
        )
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, Sequence) or isinstance(raw_assets, (str, bytes)):
        raise ManifestValidationError("manifest 'assets' must be a JSON array")
    assets = tuple(_parse_asset(raw, index) for index, raw in enumerate(raw_assets))

    identities = [(asset.kind, asset.name) for asset in assets]
    duplicates = sorted(identity for identity in set(identities) if identities.count(identity) > 1)
    if duplicates:
        raise ManifestValidationError(f"manifest has duplicate assets: {duplicates}")
    actual_inventory = frozenset(identities)
    missing = sorted(EXPECTED_INVENTORY - actual_inventory)
    unexpected = sorted(actual_inventory - EXPECTED_INVENTORY)
    if missing or unexpected or len(assets) != len(EXPECTED_INVENTORY):
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ManifestValidationError(
            "manifest must contain the CIFAR-10 archive, the fixed upstream source "
            f"archive, and exactly the 10 Pareto checkpoints ({'; '.join(details)})"
        )

    paths = [asset.path for asset in assets]
    duplicate_paths = sorted(path for path in set(paths) if paths.count(path) > 1)
    if duplicate_paths:
        raise ManifestValidationError(
            f"multiple assets target the same path: {duplicate_paths}"
        )
    urls = [asset.url for asset in assets]
    duplicate_urls = sorted(url for url in set(urls) if urls.count(url) > 1)
    if duplicate_urls:
        raise ManifestValidationError(
            f"multiple assets use the same URL: {duplicate_urls}"
        )

    return AssetManifest(schema_version=SCHEMA_VERSION, assets=assets)


def load_asset_manifest(path: Path | str) -> AssetManifest:
    """Load a JSON or YAML manifest without initiating any network access."""

    manifest_path = Path(path)
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            if manifest_path.suffix.lower() in {".yaml", ".yml"}:
                try:
                    import yaml
                except ImportError as error:  # pragma: no cover - pinned dependency
                    raise ManifestValidationError(
                        "PyYAML is required to read YAML asset manifests"
                    ) from error
                payload = yaml.safe_load(handle)
            elif manifest_path.suffix.lower() == ".json":
                payload = json.load(handle)
            else:
                raise ManifestValidationError(
                    "asset manifest filename must end in .json, .yaml, or .yml: "
                    f"{manifest_path}"
                )
    except FileNotFoundError as error:
        raise ManifestValidationError(
            f"asset manifest does not exist: {manifest_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ManifestValidationError(
            f"asset manifest is not valid JSON at line {error.lineno}, column {error.colno}: "
            f"{manifest_path}"
        ) from error
    except AssetError:
        raise
    except Exception as error:
        if error.__class__.__module__.startswith("yaml"):
            raise ManifestValidationError(
                f"asset manifest is not valid YAML: {manifest_path}: {error}"
            ) from error
        raise
    return parse_asset_manifest(payload)


def sha256_file(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def verify_extracted_source(
    manifest: AssetManifest,
    artifact_root: Path | str,
    source_root: Path | str,
) -> str:
    """Tie executable local model modules to the verified source archive.

    Only the four modules imported by :mod:`pertinence.experts` are executable
    inputs. Their bytes must exactly match the immutable archive in the asset
    manifest. The returned digest can be recorded in build metadata.
    """
    source_assets = [asset for asset in manifest.assets if asset.kind == "source"]
    if len(source_assets) != 1:
        raise ManifestValidationError("manifest must contain exactly one source archive")
    spec = source_assets[0]
    archive = _destination(Path(artifact_root), spec.path)
    actual_archive_hash, _ = sha256_file(archive)
    if actual_archive_hash != spec.sha256:
        raise AssetIntegrityError(
            f"source archive failed SHA-256 verification: expected {spec.sha256}, "
            f"got {actual_archive_hash}"
        )
    required = ("resnet.py", "mobilenetv2.py", "shufflenetv2.py", "repvgg.py")
    local_package = Path(source_root).resolve() / "pytorch_cifar_models"
    module_hashes: dict[str, str] = {}
    with tarfile.open(archive, mode="r:gz") as bundle:
        for filename in required:
            matches = [
                member
                for member in bundle.getmembers()
                if member.isfile()
                and member.name.endswith(f"/pytorch_cifar_models/{filename}")
            ]
            if len(matches) != 1:
                raise AssetIntegrityError(
                    f"source archive must contain one pytorch_cifar_models/{filename}"
                )
            archived = bundle.extractfile(matches[0])
            if archived is None:
                raise AssetIntegrityError(f"cannot read archived source module {filename}")
            expected_bytes = archived.read()
            local_path = local_package / filename
            if not local_path.is_file() or local_path.is_symlink():
                raise AssetIntegrityError(f"extracted source module is missing or unsafe: {local_path}")
            actual_bytes = local_path.read_bytes()
            if actual_bytes != expected_bytes:
                raise AssetIntegrityError(
                    f"extracted source module differs from pinned archive: {local_path}"
                )
            module_hashes[filename] = hashlib.sha256(actual_bytes).hexdigest()
    digest = hashlib.sha256()
    for filename in sorted(module_hashes):
        digest.update(filename.encode())
        digest.update(module_hashes[filename].encode())
    return digest.hexdigest()


def _destination(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    destination = (resolved_root / relative_path).resolve()
    try:
        destination.relative_to(resolved_root)
    except ValueError as error:
        raise ManifestValidationError(
            f"asset path escapes the artifact root through a symlink: {relative_path!r}"
        ) from error
    return destination


def _default_opener(request: Request, timeout: float) -> BinaryIO:
    return urlopen(request, timeout=timeout)  # noqa: S310 - URL is HTTPS-validated.


def _download_atomically(
    spec: AssetSpec,
    destination: Path,
    *,
    timeout_seconds: float,
    chunk_size: int,
    opener: Callable[[Request, float], BinaryIO],
) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
        )
        temporary_path = Path(temporary_name)
        digest = hashlib.sha256()
        size = 0
        request = Request(
            spec.url,
            headers={"User-Agent": "pertinence-asset-preparer/1"},
            method="GET",
        )
        with os.fdopen(file_descriptor, "wb") as output:
            with opener(request, timeout_seconds) as response:
                while chunk := response.read(chunk_size):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        actual_digest = digest.hexdigest()
        if actual_digest != spec.sha256:
            raise AssetIntegrityError(
                f"downloaded asset {spec.name!r} failed SHA-256 verification: "
                f"expected {spec.sha256}, got {actual_digest}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        return actual_digest, size
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def prepare_assets(
    manifest: AssetManifest,
    artifact_root: Path | str,
    *,
    dry_run: bool = False,
    verify_only: bool = False,
    timeout_seconds: float = 60.0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    opener: Callable[[Request, float], BinaryIO] = _default_opener,
) -> PreparationReport:
    """Plan, verify, or download every explicitly pinned manifest asset.

    ``dry_run`` performs no filesystem or network I/O. ``verify_only`` reads
    existing files but never creates or downloads anything. Normal preparation
    downloads only absent or invalid files and does not replace an existing
    file until the temporary download passes SHA-256 verification.
    """

    if dry_run and verify_only:
        raise ValueError("dry_run and verify_only are mutually exclusive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    root = Path(artifact_root).resolve()
    mode: Literal["dry-run", "verify-only", "prepare"] = (
        "dry-run" if dry_run else "verify-only" if verify_only else "prepare"
    )
    prepared: list[PreparedAsset] = []

    for spec in manifest.assets:
        destination = _destination(root, spec.path)
        if dry_run:
            prepared.append(
                PreparedAsset(
                    kind=spec.kind,
                    name=spec.name,
                    path=spec.path,
                    url=spec.url,
                    expected_sha256=spec.sha256,
                    actual_sha256=None,
                    size_bytes=None,
                    status="planned",
                    revision=spec.revision,
                )
            )
            continue

        if destination.is_file():
            actual_digest, size = sha256_file(destination, chunk_size)
            if actual_digest == spec.sha256:
                prepared.append(
                    PreparedAsset(
                        kind=spec.kind,
                        name=spec.name,
                        path=spec.path,
                        url=spec.url,
                        expected_sha256=spec.sha256,
                        actual_sha256=actual_digest,
                        size_bytes=size,
                        status="verified" if verify_only else "reused",
                        revision=spec.revision,
                    )
                )
                continue
            if verify_only:
                raise AssetIntegrityError(
                    f"local asset {spec.name!r} failed SHA-256 verification: "
                    f"expected {spec.sha256}, got {actual_digest} ({destination})"
                )
        elif verify_only:
            raise AssetIntegrityError(
                f"local asset {spec.name!r} is missing: {destination}"
            )

        actual_digest, size = _download_atomically(
            spec,
            destination,
            timeout_seconds=timeout_seconds,
            chunk_size=chunk_size,
            opener=opener,
        )
        prepared.append(
            PreparedAsset(
                kind=spec.kind,
                name=spec.name,
                path=spec.path,
                url=spec.url,
                expected_sha256=spec.sha256,
                actual_sha256=actual_digest,
                size_bytes=size,
                status="downloaded",
                revision=spec.revision,
            )
        )

    return PreparationReport(
        schema_version=SCHEMA_VERSION,
        mode=mode,
        artifact_root=str(root),
        assets=tuple(prepared),
        generated_at_utc=(
            None if dry_run else datetime.now(timezone.utc).isoformat(timespec="seconds")
        ),
    )


def write_report_atomically(report: PreparationReport, path: Path | str) -> None:
    """Write the verified asset receipt with an atomic same-directory rename."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{report_path.name}.", suffix=".part", dir=report_path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, report_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
