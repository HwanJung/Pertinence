"""Fail-closed GCS transfer helpers for immutable benchmark inputs and outputs."""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen


TRACE_ROOT = "gs://pertinence-509116-pertinence-runs/pertinence/benchmarks/traces"
RUN_ROOT = "gs://pertinence-509116-pertinence-runs/pertinence/benchmarks/runs"
_STORAGE_SCOPE = "https://www.googleapis.com/auth/devstorage.read_write"

Transport = Callable[[str, str, bytes | None], bytes]


@dataclass(frozen=True)
class _LocalObject:
    relative: str
    path: Path
    size: int
    sha256: str


class _AuthResponse:
    """Small google-auth transport response backed by urllib."""

    def __init__(self, status: int, data: bytes, headers: Any) -> None:
        self.status = status
        self.data = data
        self.headers = headers


class _AuthRequest:
    """Let google-auth refresh both local ADC and metadata-based credentials."""

    def __call__(
        self,
        url: str,
        method: str = "GET",
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 120.0,
        **_: Any,
    ) -> _AuthResponse:
        request = Request(url, data=body, method=method, headers=headers or {})
        try:
            with urlopen(request, timeout=timeout) as response:
                return _AuthResponse(response.status, response.read(), response.headers)
        except HTTPError as error:
            return _AuthResponse(error.code, error.read(), error.headers)


class _GcloudCredentials:
    """Fallback for reviewed workstation runs where gcloud login exists but ADC does not."""

    def __init__(self) -> None:
        self.token: str | None = None
        self._valid_until = 0.0

    @property
    def valid(self) -> bool:
        return self.token is not None and time.monotonic() < self._valid_until

    def refresh(self, _request: _AuthRequest) -> None:
        completed = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            check=True,
            capture_output=True,
            text=True,
        )
        token = completed.stdout.strip()
        if not token or any(character.isspace() for character in token):
            raise RuntimeError("gcloud did not return one access token")
        self.token = token
        # gcloud user tokens normally last one hour; refresh conservatively before expiry.
        self._valid_until = time.monotonic() + 50 * 60


@lru_cache(maxsize=1)
def _credentials() -> Any:
    try:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError
    except ImportError as error:  # pragma: no cover - depends on the installation profile
        raise RuntimeError("google-auth is required for GCS transfers") from error

    try:
        credentials, _ = google.auth.default(scopes=(_STORAGE_SCOPE,))
        return credentials
    except DefaultCredentialsError as error:
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            raise RuntimeError(
                "Application Default Credentials are unavailable inside Kubernetes"
            ) from error
        return _GcloudCredentials()


def _default_transport(method: str, url: str, data: bytes | None = None) -> bytes:
    """Call GCS with local ADC or metadata/workload-identity credentials."""

    credentials = _credentials()
    if not credentials.valid:
        credentials.refresh(_AuthRequest())
    if not credentials.token:
        raise RuntimeError("Application Default Credentials did not provide an access token")

    headers = {"Authorization": f"Bearer {credentials.token}"}
    if data is not None:
        headers["Content-Type"] = "application/octet-stream"
    request = Request(url, data=data, method=method, headers=headers)
    with urlopen(request, timeout=120.0) as response:
        return response.read()


def _prefix(uri: str, root: str) -> str:
    if not re.fullmatch(re.escape(root) + r"/[a-z0-9](?:[a-z0-9.-]{0,126}[a-z0-9])?/?", uri):
        raise ValueError(f"GCS URI must be one immutable ID directly below {root}")
    return uri.rstrip("/")


def _bucket_object(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError("invalid GCS object URI")
    return parsed.netloc, parsed.path.lstrip("/")


def _upload(uri: str, data: bytes, transport: Transport) -> None:
    bucket, name = _bucket_object(uri)
    url = (
        f"https://storage.googleapis.com/upload/storage/v1/b/{quote(bucket, safe='')}/o"
        f"?uploadType=media&ifGenerationMatch=0&name={quote(name, safe='')}"
    )
    transport("POST", url, data)


def _download(uri: str, generation: str, transport: Transport) -> bytes:
    if not generation.isdecimal() or int(generation) <= 0:
        raise ValueError("GCS object generation must be a positive integer")
    bucket, name = _bucket_object(uri)
    query = urlencode({"alt": "media", "ifGenerationMatch": generation})
    url = (
        f"https://storage.googleapis.com/storage/v1/b/{quote(bucket, safe='')}/o/"
        f"{quote(name, safe='')}?{query}"
    )
    return transport("GET", url, None)


def _list_prefix(prefix: str, transport: Transport) -> dict[str, dict[str, Any]]:
    bucket, name = _bucket_object(prefix)
    object_prefix = f"{name.rstrip('/')}/"
    objects: dict[str, dict[str, Any]] = {}
    page_token: str | None = None
    while True:
        query: dict[str, str] = {"prefix": object_prefix}
        if page_token is not None:
            query["pageToken"] = page_token
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{quote(bucket, safe='')}/o?"
            f"{urlencode(query)}"
        )
        try:
            payload = json.loads(transport("GET", url, None))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("GCS object inventory returned invalid JSON") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("items", []), list):
            raise RuntimeError("GCS object inventory has an invalid shape")
        for item in payload.get("items", []):
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise RuntimeError("GCS object inventory contains invalid metadata")
            full_name = item["name"]
            if not full_name.startswith(object_prefix):
                raise RuntimeError("GCS object inventory escaped the requested prefix")
            relative = full_name[len(object_prefix) :]
            if not relative or relative in objects:
                raise RuntimeError("GCS object inventory contains an invalid object name")
            objects[relative] = item
        token = payload.get("nextPageToken")
        if token is None:
            return objects
        if not isinstance(token, str) or not token:
            raise RuntimeError("GCS object inventory contains an invalid page token")
        page_token = token


def _digest(path: Path) -> tuple[int, str]:
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            sha256.update(chunk)
    return size, sha256.hexdigest()


def _local_objects(source: Path) -> dict[str, _LocalObject]:
    result: dict[str, _LocalObject] = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source).as_posix()
        size, sha256 = _digest(path)
        result[relative] = _LocalObject(relative, path, size, sha256)
    return result


def _read_unchanged(local: _LocalObject) -> bytes:
    data = local.path.read_bytes()
    if len(data) != local.size or hashlib.sha256(data).hexdigest() != local.sha256:
        raise RuntimeError(f"publish source changed while being uploaded: {local.relative}")
    return data


def _remote_matches(
    prefix: str,
    local: dict[str, _LocalObject],
    remote: dict[str, dict[str, Any]],
    transport: Transport,
) -> bool:
    if set(remote) != set(local):
        return False
    for relative, expected in local.items():
        metadata = remote[relative]
        generation = metadata.get("generation")
        size = metadata.get("size")
        if not isinstance(generation, str) or not generation.isdecimal() or int(generation) <= 0:
            return False
        if not isinstance(size, str) or not size.isdecimal() or int(size) != expected.size:
            return False
        data = _download(f"{prefix}/{relative}", generation, transport)
        if len(data) != expected.size or hashlib.sha256(data).hexdigest() != expected.sha256:
            return False
    return True


def _mark_invalid(prefix: str, reason: str, uploaded: list[str], transport: Transport) -> None:
    receipt = {
        "schema_version": 1,
        "valid": False,
        "reason": reason,
        "uploaded_objects": uploaded,
    }
    marker = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        _upload(f"{prefix}/INVALID.json", marker, transport)
    except Exception:
        # A pre-existing marker or a transport failure must never lead to overwrite/delete.
        pass


def publish_create_only(
    source: Path,
    destination: str,
    *,
    kind: str,
    transport: Transport = _default_transport,
) -> None:
    """Create an immutable prefix, or safely reuse an identical complete prefix."""

    root = TRACE_ROOT if kind == "trace" else RUN_ROOT if kind == "run" else None
    if root is None:
        raise ValueError("publish kind must be trace or run")
    prefix = _prefix(destination, root)
    source = Path(source)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("publish source must be a regular directory")
    local = _local_objects(source)
    if not local:
        raise ValueError("publish source is empty")

    # Inventory failure is pre-mutation, so it must not create an INVALID marker.
    remote = _list_prefix(prefix, transport)
    if remote:
        if _remote_matches(prefix, local, remote, transport):
            return
        _mark_invalid(prefix, "existing_prefix_conflict", [], transport)
        raise RuntimeError("create-only upload refused a partial or conflicting prefix")

    uploaded: list[str] = []
    try:
        for relative, item in local.items():
            _upload(f"{prefix}/{relative}", _read_unchanged(item), transport)
            uploaded.append(relative)
        remote = _list_prefix(prefix, transport)
        if not _remote_matches(prefix, local, remote, transport):
            raise RuntimeError("uploaded prefix did not verify as an exact byte-for-byte match")
    except Exception as error:
        # A concurrent identical publisher may have won generation-match races.
        try:
            remote = _list_prefix(prefix, transport)
            if _remote_matches(prefix, local, remote, transport):
                return
        except Exception:
            pass
        _mark_invalid(
            prefix,
            f"partial_create_only_upload:{type(error).__name__}",
            uploaded,
            transport,
        )
        raise RuntimeError("create-only upload failed; existing objects were not deleted") from error


def _verify_download(data: bytes, metadata: dict[str, Any], relative: str) -> None:
    size = metadata.get("size")
    if not isinstance(size, str) or not size.isdecimal() or len(data) != int(size):
        raise RuntimeError(f"downloaded GCS object has the wrong size: {relative}")
    md5_hash = metadata.get("md5Hash")
    if not isinstance(md5_hash, str):
        raise RuntimeError(f"trace object has no verifiable MD5 hash: {relative}")
    actual = base64.b64encode(hashlib.md5(data, usedforsecurity=False).digest()).decode("ascii")
    if actual != md5_hash:
        raise RuntimeError(f"downloaded GCS object failed its MD5 check: {relative}")


def download_trace(
    source: str,
    output: Path,
    *,
    transport: Transport = _default_transport,
) -> Path:
    """Download exactly two generation-pinned trace objects into a new directory."""

    prefix = _prefix(source, TRACE_ROOT)
    remote = _list_prefix(prefix, transport)
    expected = {"trace.json", "trace.npz"}
    if set(remote) != expected:
        raise RuntimeError("trace prefix must contain exactly trace.json and trace.npz")
    for name in expected:
        generation = remote[name].get("generation")
        if not isinstance(generation, str) or not generation.isdecimal() or int(generation) <= 0:
            raise RuntimeError(f"trace object has an invalid generation: {name}")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name in ("trace.json", "trace.npz"):
            generation = remote[name]["generation"]
            data = _download(f"{prefix}/{name}", generation, transport)
            _verify_download(data, remote[name], name)
            descriptor = os.open(
                output / name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
    except Exception as error:
        # Keep the partial directory as transfer evidence; callers must choose a new path.
        raise RuntimeError("trace download failed; partial local directory retained") from error
    return output
