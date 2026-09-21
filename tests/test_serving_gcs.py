from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from pertinence.serving import gcs
from pertinence.serving.gcs import download_trace, publish_create_only


TRACE_URI = "gs://pertinence-509116-pertinence-runs/pertinence/benchmarks/traces/trace-1"
OBJECT_PREFIX = "pertinence/benchmarks/traces/trace-1/"


class FakeGCS:
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = dict(objects or {})
        self.calls: list[tuple[str, str, bytes | None]] = []
        self.fail_upload_name: str | None = None
        self.corrupt_download_name: str | None = None

    @staticmethod
    def _metadata(name: str, data: bytes) -> dict[str, str]:
        digest = hashlib.md5(data, usedforsecurity=False).digest()
        return {
            "name": name,
            "generation": "17",
            "size": str(len(data)),
            "md5Hash": base64.b64encode(digest).decode("ascii"),
        }

    def __call__(self, method: str, url: str, data: bytes | None = None) -> bytes:
        self.calls.append((method, url, data))
        parsed = urlsplit(url)
        query = parse_qs(parsed.query)
        if method == "GET" and parsed.path.endswith("/o"):
            prefix = query["prefix"][0]
            items = [
                self._metadata(name, value)
                for name, value in sorted(self.objects.items())
                if name.startswith(prefix)
            ]
            return json.dumps({"items": items}).encode()
        if method == "GET" and "/o/" in parsed.path:
            name = unquote(parsed.path.split("/o/", 1)[1])
            assert query["ifGenerationMatch"] == ["17"]
            value = self.objects[name]
            return b"corrupt" if name == self.corrupt_download_name else value
        if method == "POST":
            assert query["ifGenerationMatch"] == ["0"]
            name = query["name"][0]
            if name == self.fail_upload_name or name in self.objects:
                raise RuntimeError("generation precondition failed")
            assert data is not None
            self.objects[name] = data
            return json.dumps(self._metadata(name, data)).encode()
        raise AssertionError((method, url))


def _trace_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "trace.json").write_text("{}\n", encoding="utf-8")
    (source / "trace.npz").write_bytes(b"npz")
    return source


def test_publish_is_create_only_and_reuses_identical_complete_prefix(tmp_path: Path) -> None:
    source = _trace_source(tmp_path)
    server = FakeGCS()

    publish_create_only(source, TRACE_URI, kind="trace", transport=server)
    first_posts = [call for call in server.calls if call[0] == "POST"]
    assert len(first_posts) == 2
    assert all("ifGenerationMatch=0" in url for _, url, _ in first_posts)

    server.calls.clear()
    publish_create_only(source, TRACE_URI, kind="trace", transport=server)
    assert all(method == "GET" for method, _, _ in server.calls)
    assert all(method != "DELETE" for method, _, _ in server.calls)


@pytest.mark.parametrize(
    "existing",
    [
        {f"{OBJECT_PREFIX}trace.json": b"{}\n"},
        {
            f"{OBJECT_PREFIX}trace.json": b"different",
            f"{OBJECT_PREFIX}trace.npz": b"npz",
        },
    ],
    ids=["partial", "conflicting"],
)
def test_existing_partial_or_conflicting_prefix_fails_closed(
    tmp_path: Path, existing: dict[str, bytes]
) -> None:
    source = _trace_source(tmp_path)
    server = FakeGCS(existing)
    before = dict(server.objects)

    with pytest.raises(RuntimeError, match="partial or conflicting"):
        publish_create_only(source, TRACE_URI, kind="trace", transport=server)

    assert all(server.objects[name] == data for name, data in before.items())
    assert f"{OBJECT_PREFIX}INVALID.json" in server.objects
    assert all(method != "DELETE" for method, _, _ in server.calls)


def test_partial_publish_attempts_create_only_invalid_marker_without_delete(tmp_path: Path) -> None:
    source = _trace_source(tmp_path)
    server = FakeGCS()
    server.fail_upload_name = f"{OBJECT_PREFIX}trace.npz"

    with pytest.raises(RuntimeError, match="not deleted"):
        publish_create_only(source, TRACE_URI, kind="trace", transport=server)

    assert server.objects[f"{OBJECT_PREFIX}trace.json"] == b"{}\n"
    assert f"{OBJECT_PREFIX}INVALID.json" in server.objects
    assert all(method != "DELETE" for method, _, _ in server.calls)


def test_inventory_failure_does_not_attempt_marker(tmp_path: Path) -> None:
    source = _trace_source(tmp_path)
    calls = []

    def transport(method: str, url: str, data: bytes | None = None) -> bytes:
        calls.append((method, url, data))
        raise RuntimeError("inventory unavailable")

    with pytest.raises(RuntimeError, match="inventory unavailable"):
        publish_create_only(source, TRACE_URI, kind="trace", transport=transport)
    assert [method for method, _, _ in calls] == ["GET"]


def test_download_trace_inventories_exact_set_and_pins_generations(tmp_path: Path) -> None:
    objects = {
        f"{OBJECT_PREFIX}trace.json": b"{}\n",
        f"{OBJECT_PREFIX}trace.npz": b"npz",
    }
    server = FakeGCS(objects)

    output = download_trace(TRACE_URI, tmp_path / "trace", transport=server)

    assert (output / "trace.json").read_bytes() == b"{}\n"
    assert (output / "trace.npz").read_bytes() == b"npz"
    media_gets = [url for method, url, _ in server.calls if method == "GET" and "/o/" in url]
    assert len(media_gets) == 2
    assert all("ifGenerationMatch=17" in url for url in media_gets)


def test_download_rejects_extra_object_before_creating_output(tmp_path: Path) -> None:
    objects = {
        f"{OBJECT_PREFIX}trace.json": b"{}\n",
        f"{OBJECT_PREFIX}trace.npz": b"npz",
        f"{OBJECT_PREFIX}INVALID.json": b"invalid",
    }
    server = FakeGCS(objects)
    output = tmp_path / "trace"

    with pytest.raises(RuntimeError, match="exactly"):
        download_trace(TRACE_URI, output, transport=server)
    assert not output.exists()


def test_download_hash_failure_retains_partial_directory(tmp_path: Path) -> None:
    objects = {
        f"{OBJECT_PREFIX}trace.json": b"{}\n",
        f"{OBJECT_PREFIX}trace.npz": b"npz",
    }
    server = FakeGCS(objects)
    server.corrupt_download_name = f"{OBJECT_PREFIX}trace.npz"
    output = tmp_path / "trace"

    with pytest.raises(RuntimeError, match="partial local directory retained"):
        download_trace(TRACE_URI, output, transport=server)
    assert output.is_dir()
    assert (output / "trace.json").is_file()
    assert not (output / "trace.npz").exists()


def test_download_fails_closed_when_object_has_no_hash(tmp_path: Path) -> None:
    class HashlessGCS(FakeGCS):
        @staticmethod
        def _metadata(name: str, data: bytes) -> dict[str, str]:
            return {"name": name, "generation": "17", "size": str(len(data))}

    server = HashlessGCS(
        {
            f"{OBJECT_PREFIX}trace.json": b"{}\n",
            f"{OBJECT_PREFIX}trace.npz": b"npz",
        }
    )
    output = tmp_path / "trace"

    with pytest.raises(RuntimeError, match="partial local directory retained"):
        download_trace(TRACE_URI, output, transport=server)
    assert output.is_dir()


def test_default_transport_uses_adc_without_calling_metadata_directly(monkeypatch) -> None:
    class Credentials:
        valid = False
        token = None

        def refresh(self, request) -> None:
            assert isinstance(request, gcs._AuthRequest)
            self.token = "adc-token"
            self.valid = True

    credentials = Credentials()
    default_calls = []

    def fake_default(*, scopes):
        default_calls.append(scopes)
        return credentials, "project"

    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return b"response"

    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return Response()

    import google.auth

    monkeypatch.setattr(google.auth, "default", fake_default)
    monkeypatch.setattr(gcs, "urlopen", fake_urlopen)
    gcs._credentials.cache_clear()

    try:
        assert gcs._default_transport("GET", "https://storage.googleapis.com/example") == b"response"
        assert default_calls == [(gcs._STORAGE_SCOPE,)]
        assert requests[0][0].get_header("Authorization") == "Bearer adc-token"
        assert "metadata.google.internal" not in requests[0][0].full_url
    finally:
        gcs._credentials.cache_clear()


def test_default_transport_falls_back_to_active_gcloud_login(monkeypatch) -> None:
    from google.auth.exceptions import DefaultCredentialsError

    monkeypatch.setattr(
        "google.auth.default",
        lambda **_: (_ for _ in ()).throw(DefaultCredentialsError("no ADC")),
    )
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(stdout="gcloud-token\n")

    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return b"response"

    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(gcs.subprocess, "run", fake_run)
    monkeypatch.setattr(gcs, "urlopen", fake_urlopen)
    gcs._credentials.cache_clear()
    try:
        assert gcs._default_transport("GET", "https://storage.googleapis.com/example") == b"response"
        assert commands[0][0] == ["gcloud", "auth", "print-access-token"]
        assert requests[0][0].get_header("Authorization") == "Bearer gcloud-token"
    finally:
        gcs._credentials.cache_clear()
