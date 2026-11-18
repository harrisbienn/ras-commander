"""Test trust and transfer policy with fixture bytes, never executable downloads."""

import hashlib
import io
import json
import os
import platform
import stat
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
from typing import ClassVar

import pytest

from ras_commander.dss._hec_monolith import HecMonolithDownloader


def identity(content):
    return {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.delenv("RAS_DSS_ALLOW_DOWNLOADS", raising=False)
    monkeypatch.setattr(
        HecMonolithDownloader,
        "COMMON_JARS",
        [{"artifactId": "fixture", "group": "fixture.group", "version": "1"}],
    )
    native_name = {
        "Windows": "javaHeclib.dll",
        "Linux": "libjavaHeclib.so",
        "Darwin": "libjavaHeclib.dylib",
    }[platform.system()]
    native = b"fixture native bytes, not executable"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(native_name, native)
    jar_name = "jar/fixture-1.jar"
    jar = b"fixture jar bytes, not executable"
    archive_name = "lib/" + HecMonolithDownloader._filename(
        HecMonolithDownloader.NATIVE_LIBS[platform.system()]
    )
    trust = {
        "format_version": 1,
        "platform": platform.system(),
        "artifacts": {jar_name: identity(jar), "lib/" + native_name: identity(native)},
        "downloads": {
            jar_name: identity(jar),
            archive_name: identity(archive.getvalue()),
        },
    }
    manifest = tmp_path / "approved.json"
    manifest.write_text(json.dumps(trust))
    downloader = HecMonolithDownloader(tmp_path / "cache", trust_manifest=manifest)
    return (
        downloader,
        trust,
        {jar_name: jar, archive_name: archive.getvalue(), "lib/" + native_name: native},
    )


def seed(downloader, payloads, names):
    for name in names:
        path = downloader.cache_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[name])


class Response:
    status_code = 200
    headers: ClassVar[dict[str, str]] = {}

    def __init__(self, content):
        self.raw = io.BytesIO(content)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.raw.close()

    def raise_for_status(self):
        pass


def test_missing_or_cache_local_trust_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("RAS_DSS_TRUST_MANIFEST", raising=False)
    with pytest.raises(RuntimeError, match="RAS_DSS_TRUST_MANIFEST"):
        HecMonolithDownloader(tmp_path).is_installed()
    with pytest.raises(ValueError, match="outside"):
        HecMonolithDownloader(
            tmp_path, trust_manifest=tmp_path / "trust.json"
        ).is_installed()


def test_cached_artifacts_verified_on_each_use(runtime):
    downloader, trust, payloads = runtime
    seed(downloader, payloads, trust["artifacts"])
    assert downloader.is_installed()
    assert downloader.get_classpath() == [str(downloader.jar_dir / "fixture-1.jar")]
    (downloader.jar_dir / "fixture-1.jar").write_bytes(
        b"x" * len(payloads["jar/fixture-1.jar"])
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        downloader.get_classpath()


def test_unexpected_jar_rejected(runtime):
    downloader, trust, payloads = runtime
    seed(downloader, payloads, trust["artifacts"])
    (downloader.jar_dir / "unapproved.jar").write_bytes(b"not allowed")
    with pytest.raises(ValueError, match="Unexpected"):
        downloader.get_classpath()


@pytest.mark.parametrize("filename", ["rogue.so.1", "libgfortran.so.5", "OTHER.SO.12"])
@pytest.mark.parametrize(
    "operation", ["is_installed", "get_library_path", "get_classpath"]
)
def test_unexpected_versioned_shared_library_rejected(runtime, filename, operation):
    downloader, trust, payloads = runtime
    seed(downloader, payloads, trust["artifacts"])
    (downloader.lib_dir / filename).write_bytes(b"unapproved fixture bytes")
    with pytest.raises(ValueError, match="Unexpected executable cache artifact"):
        getattr(downloader, operation)()


def test_sealed_cache_never_fetches(runtime, monkeypatch):
    downloader, _, _ = runtime
    monkeypatch.setattr(
        "requests.get", lambda *a, **k: pytest.fail("Network attempted")
    )
    with pytest.raises(RuntimeError, match="downloads are disabled"):
        downloader.install()


def test_verified_download_and_native_extraction(runtime, monkeypatch):
    downloader, trust, payloads = runtime
    sources = downloader._download_sources()
    by_url = {url: payloads[name] for name, url in sources.items()}

    def get(url, **kwargs):
        assert kwargs["allow_redirects"] is False
        assert kwargs["timeout"] == (5, 1)
        return Response(by_url[url])

    monkeypatch.setattr("requests.get", get)
    downloader.allow_downloads = True
    downloader.install()
    assert downloader.is_installed()
    assert all(
        (downloader.cache_dir / name).read_bytes() == payloads[name]
        for name in trust["artifacts"]
    )
    assert not list(downloader.cache_dir.rglob(".download-*"))
    assert not list(downloader.cache_dir.rglob(".extract-*"))


@pytest.mark.parametrize(
    "failure", ["oversize", "corrupt", "interrupted", "redirect", "deadline"]
)
def test_failed_transfer_never_publishes_and_retry_succeeds(
    runtime, monkeypatch, failure
):
    downloader, _, payloads = runtime
    downloader.allow_downloads = True
    content = payloads["jar/fixture-1.jar"]
    response = Response(content)
    if failure == "oversize":
        response.raw = io.BytesIO(content + b"extra")
    elif failure == "corrupt":
        response.raw = io.BytesIO(b"x" * len(content))
    elif failure == "interrupted":

        class Broken:
            def read1(self, size):
                raise OSError("fixture interruption")

            def close(self):
                pass

        response.raw = Broken()
    elif failure == "redirect":
        response.status_code = 302
    else:
        downloader.DOWNLOAD_SECONDS = 0
    monkeypatch.setattr("requests.get", lambda *a, **k: response)
    name, url = next(iter(downloader._download_sources().items()))
    with pytest.raises((ValueError, OSError, TimeoutError)):
        downloader.download_file(url, downloader.cache_dir / name)
    assert not (downloader.cache_dir / name).exists()
    assert not list(downloader.jar_dir.glob(".download-*"))
    downloader.DOWNLOAD_SECONDS = 300
    monkeypatch.setattr("requests.get", lambda *a, **k: Response(content))
    assert (
        downloader.download_file(url, downloader.cache_dir / name).read_bytes()
        == content
    )


@pytest.mark.parametrize("member", ["../escape.dll", "JAVAHECLIB.DLL", "extra.dll"])
def test_native_archive_allowlist_refuses_unexpected_members(
    runtime, monkeypatch, member
):
    downloader, trust, payloads = runtime
    archive_name = next(name for name in trust["downloads"] if name.endswith(".zip"))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, b"fixture")
    payloads[archive_name] = archive.getvalue()
    trust["downloads"][archive_name] = identity(archive.getvalue())
    downloader.trust_manifest.write_text(json.dumps(trust))
    seed(downloader, payloads, [archive_name])
    with pytest.raises(ValueError, match="allowlist|Unsafe"):
        downloader.download_native_library()
    assert not (downloader.cache_dir / "escape.dll").exists()


def test_conflicting_cached_transfer_is_not_overwritten(runtime, monkeypatch):
    downloader, _, payloads = runtime
    name, url = next(iter(downloader._download_sources().items()))
    seed(downloader, payloads, [name])
    destination = downloader.cache_dir / name
    destination.write_bytes(b"corrupt")
    monkeypatch.setattr(
        "requests.get", lambda *a, **k: pytest.fail("Network attempted")
    )
    with pytest.raises(ValueError):
        downloader.download_file(url, destination)
    assert destination.read_bytes() == b"corrupt"


def test_unapproved_cache_refuses_before_acquisition(tmp_path, monkeypatch):
    """A cache's existing files cannot supply their own trust identity."""
    monkeypatch.delenv("RAS_DSS_TRUST_MANIFEST", raising=False)
    with pytest.raises(RuntimeError, match="RAS_DSS_TRUST_MANIFEST"):
        HecMonolithDownloader(tmp_path).is_installed()


def test_sha1_without_expected_digest_is_not_verification(tmp_path):
    """The legacy checksum helper must refuse an absent expected digest."""
    artifact = tmp_path / "artifact.jar"
    artifact.write_bytes(b"untrusted fixture")
    assert HecMonolithDownloader(tmp_path / "cache").verify_sha1(artifact) is False


@pytest.mark.parametrize("value", ["1", "true", 1, 0, [], {}])
def test_download_opt_in_requires_boolean_argument(tmp_path, value):
    with pytest.raises(TypeError, match="allow_downloads"):
        HecMonolithDownloader(tmp_path, allow_downloads=value)


@pytest.mark.parametrize(
    "value,expected", [("1", True), ("0", False), ("true", False), ("", False)]
)
def test_environment_policy_is_ras_specific_and_explicit(
    tmp_path, monkeypatch, value, expected
):
    monkeypatch.setenv("HMS_DSS_ALLOW_DOWNLOADS", "1")
    monkeypatch.setenv("RAS_DSS_ALLOW_DOWNLOADS", value)
    monkeypatch.setenv("RAS_DSS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("RAS_DSS_TRUST_MANIFEST", str(tmp_path / "approved.json"))
    downloader = HecMonolithDownloader()
    assert downloader.allow_downloads is expected
    assert downloader.cache_dir == tmp_path / "cache"
    assert downloader.trust_manifest == tmp_path / "approved.json"


def test_hms_policy_does_not_enable_ras_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("HMS_DSS_ALLOW_DOWNLOADS", "1")
    monkeypatch.delenv("RAS_DSS_ALLOW_DOWNLOADS", raising=False)
    assert HecMonolithDownloader(tmp_path).allow_downloads is False


@pytest.mark.parametrize(
    "kind",
    ["version", "root", "extra", "size", "digest", "duplicate", "oversize", "count"],
)
def test_malformed_trust_refuses_before_cache_creation(runtime, kind):
    downloader, trust, _ = runtime
    if kind == "version":
        trust["format_version"] = True
    elif kind == "root":
        trust = []
    elif kind == "extra":
        trust["accept_observed_bytes"] = True
    elif kind == "size":
        trust["artifacts"]["jar/fixture-1.jar"]["size_bytes"] = True
    elif kind == "digest":
        trust["artifacts"]["jar/fixture-1.jar"]["sha256"] = 1
    elif kind == "count":
        downloader.MAX_ARCHIVE_MEMBERS = 1
    encoded = json.dumps(trust)
    if kind == "duplicate":
        encoded = encoded.replace(
            '"format_version": 1', '"format_version": 1, "format_version": 1'
        )
    elif kind == "oversize":
        downloader.MAX_MANIFEST_BYTES = 8
    downloader.trust_manifest.write_text(encoded)
    with pytest.raises(ValueError):
        downloader.is_installed()
    assert not downloader.cache_dir.exists()


@pytest.mark.parametrize(
    "name",
    [
        "lib/../bad.dll",
        "lib//bad.dll",
        "lib/C:bad.dll",
        "lib/CON.dll",
        "lib/bad.dll.",
        "lib/bad\x00.dll",
        "lib/bad?.dll",
    ],
)
def test_unsafe_manifest_artifact_names_refuse(runtime, name):
    downloader, trust, _ = runtime
    trust["artifacts"][name] = identity(b"fake")
    downloader.trust_manifest.write_text(json.dumps(trust))
    with pytest.raises(ValueError, match="Unsafe"):
        downloader.is_installed()


def test_retained_provisioning_archive_is_verified_but_not_required(runtime):
    downloader, trust, payloads = runtime
    seed(downloader, payloads, trust["artifacts"])
    assert downloader.is_installed()
    name = next(name for name in trust["downloads"] if name.endswith(".zip"))
    seed(downloader, payloads, [name])
    assert downloader.is_installed()
    (downloader.cache_dir / name).write_bytes(b"corrupt archive")
    with pytest.raises(ValueError, match="wrong size"):
        downloader.is_installed()


def test_reparse_cache_directory_and_linked_manifest_are_refused(runtime, monkeypatch):
    downloader, _, _ = runtime
    original = downloader._is_link
    monkeypatch.setattr(
        downloader,
        "_is_link",
        lambda path: path == downloader.trust_manifest or original(path),
    )
    with pytest.raises(ValueError, match="links or junctions"):
        downloader.is_installed()
    monkeypatch.setattr(
        downloader,
        "_is_link",
        lambda path: path == downloader.jar_dir or original(path),
    )
    with pytest.raises(ValueError, match="links are prohibited"):
        downloader.is_installed()


def test_windows_reparse_attribute_is_detected_without_is_junction(
    tmp_path, monkeypatch
):
    path = tmp_path / "junction"
    path.mkdir()
    original = Path.lstat
    flag = 1024
    monkeypatch.setattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", flag, raising=False)
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda p: (
            SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=flag)
            if p == path
            else original(p)
        ),
    )
    assert HecMonolithDownloader._is_link(path)


@pytest.mark.parametrize(
    "failure",
    ["short", "length", "partial_status", "late_eof", "late_chunk", "slow_trickle"],
)
def test_transfer_boundaries_never_publish_unapproved_bytes(
    runtime, monkeypatch, failure
):
    from ras_commander.dss import _hec_monolith as module

    downloader, _, payloads = runtime
    downloader.allow_downloads = True
    name, url = next(iter(downloader._download_sources().items()))
    content = payloads[name]
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    response = Response(content)
    if failure == "short":
        response.raw = io.BytesIO(content[:-1])
    elif failure == "length":
        response.headers = {"content-length": str(len(content) + 1)}
    elif failure == "partial_status":
        response.status_code = 206
    else:

        class Delayed(io.BytesIO):
            def read1(self, size):
                chunk = super().read1(1 if failure == "slow_trickle" else size)
                if failure == "late_chunk" or (failure == "late_eof" and not chunk):
                    clock[0] = downloader.DOWNLOAD_SECONDS + 1
                elif failure == "slow_trickle":
                    clock[0] += downloader.DOWNLOAD_SECONDS / 2
                return chunk

        response.raw = Delayed(content)
    monkeypatch.setattr("requests.get", lambda *a, **k: response)
    with pytest.raises((ValueError, TimeoutError)):
        downloader.download_file(url, downloader.cache_dir / name)
    assert not (downloader.cache_dir / name).exists()
    assert not list(downloader.cache_dir.rglob(".download-*"))


def test_hash_verification_cannot_publish_after_transfer_deadline(runtime, monkeypatch):
    from ras_commander.dss import _hec_monolith as module

    downloader, _, payloads = runtime
    downloader.allow_downloads = True
    name, url = next(iter(downloader._download_sources().items()))
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr("requests.get", lambda *a, **k: Response(payloads[name]))
    original = downloader._verify

    def delayed_verification(path, metadata):
        original(path, metadata)
        clock[0] = downloader.DOWNLOAD_SECONDS + 1

    monkeypatch.setattr(downloader, "_verify", delayed_verification)
    with pytest.raises(TimeoutError, match="total transfer deadline"):
        downloader.download_file(url, downloader.cache_dir / name)
    assert not (downloader.cache_dir / name).exists()
    assert not list(downloader.cache_dir.rglob(".download-*"))


@pytest.mark.parametrize("conflict", [False, True])
def test_concurrent_identical_and_conflicting_promotions_are_immutable(
    runtime, monkeypatch, conflict
):
    downloader, _, payloads = runtime
    downloader.allow_downloads = True
    name, url = next(iter(downloader._download_sources().items()))
    content = payloads[name]
    destination = downloader.cache_dir / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    ready = Barrier(2)
    staged = Event()
    original = downloader._promote

    def promote(temporary, path, metadata, **kwargs):
        staged.set()
        ready.wait(timeout=10)
        return original(temporary, path, metadata, **kwargs)

    monkeypatch.setattr(downloader, "_promote", promote)
    monkeypatch.setattr("requests.get", lambda *a, **k: Response(content))
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(downloader.download_file, url, destination)
        if conflict:

            def competing_writer():
                assert staged.wait(timeout=10)
                destination.write_bytes(b"prior conflicting bytes")
                ready.wait(timeout=10)

            second = pool.submit(competing_writer)
            second.result(timeout=15)
            with pytest.raises(ValueError):
                first.result(timeout=15)
            assert destination.read_bytes() == b"prior conflicting bytes"
        else:
            second = pool.submit(downloader.download_file, url, destination)
            assert first.result(timeout=15) == second.result(timeout=15) == destination
            assert destination.read_bytes() == content
    assert not list(downloader.cache_dir.rglob(".download-*"))


@pytest.mark.parametrize(
    "case",
    ["traversal_directory", "extra_directory", "symlink", "duplicate", "size", "count"],
)
def test_native_archive_rejects_unsafe_structure_before_extracting(runtime, case):
    downloader, trust, payloads = runtime
    archive_name = next(name for name in trust["downloads"] if name.endswith(".zip"))
    native_name = next(
        name[4:] for name in trust["artifacts"] if name.startswith("lib/")
    )
    content = payloads["lib/" + native_name]
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        member = zipfile.ZipInfo(native_name)
        if case == "symlink":
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(member, content + (b"extra" if case == "size" else b""))
        if case.endswith("directory"):
            bundle.writestr(
                "../escape/" if case == "traversal_directory" else "extra/", b""
            )
        elif case == "duplicate":
            bundle.writestr(native_name.upper(), content)
        elif case == "count":
            for index in range(downloader.MAX_ARCHIVE_MEMBERS):
                bundle.writestr(f"extra{index}/", b"")
    payloads[archive_name] = archive.getvalue()
    trust["downloads"][archive_name] = identity(archive.getvalue())
    downloader.trust_manifest.write_text(json.dumps(trust))
    seed(downloader, payloads, [archive_name])
    with pytest.raises(ValueError):
        downloader.download_native_library()
    assert not (downloader.lib_dir / native_name).exists()
    assert not list(downloader.cache_dir.rglob(".extract-*"))


@pytest.fixture
def configured_runtime(runtime, monkeypatch):
    """Use real cache validation with a Java configuration stub that cannot launch Java."""
    from ras_commander.dss import _hec_monolith as module
    from ras_commander.dss.RasDss import RasDss

    downloader, trust, payloads = runtime
    seed(downloader, payloads, trust["artifacts"])
    monkeypatch.setattr(RasDss, "_monolith", None)
    monkeypatch.setattr(RasDss, "_jvm_configured", False)
    monkeypatch.setattr(module, "HecMonolithDownloader", lambda: downloader)
    configuration = SimpleNamespace(
        vm_running=False, classpath=["unapproved-caller.jar"]
    )
    configuration.set_classpath = lambda *paths: setattr(
        configuration, "classpath", list(paths)
    )
    configuration.add_classpath = lambda *paths: pytest.fail(
        "Appending can retain unapproved entries"
    )
    monkeypatch.setitem(sys.modules, "jnius_config", configuration)
    monkeypatch.setenv("JAVA_HOME", "fixture-java-not-executed")
    monkeypatch.setenv("CLASSPATH", "unapproved-environment.jar")
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.setenv("LD_LIBRARY_PATH", "existing-system-search-path")
    return RasDss, configuration, downloader, payloads


def test_jvm_configuration_uses_only_selected_classpath(configured_runtime):
    ras_dss, configuration, downloader, _ = configured_runtime
    ras_dss._configure_jvm()
    assert configuration.classpath == downloader.get_classpath()
    assert "unapproved-caller.jar" not in configuration.classpath
    assert "unapproved-environment.jar" not in configuration.classpath
    assert configuration.vm_running is False
    assert ras_dss._jvm_configured is True


def test_unowned_running_jvm_refuses_admission(configured_runtime):
    ras_dss, configuration, _, _ = configured_runtime
    configuration.vm_running = True
    with pytest.raises(RuntimeError, match="fresh process"):
        ras_dss._configure_jvm()
    assert configuration.classpath == ["unapproved-caller.jar"]
    assert ras_dss._jvm_configured is False


def test_owned_reentry_revalidates_cached_artifacts(configured_runtime):
    ras_dss, configuration, downloader, _ = configured_runtime
    ras_dss._configure_jvm()
    configuration.vm_running = True
    ras_dss._configure_jvm()
    (downloader.jar_dir / "fixture-1.jar").write_bytes(b"changed")
    with pytest.raises(ValueError, match="wrong size"):
        ras_dss._configure_jvm()


def test_failed_initialization_does_not_cache_success_and_can_retry(configured_runtime):
    ras_dss, _, downloader, payloads = configured_runtime
    jar = downloader.jar_dir / "fixture-1.jar"
    jar.unlink()
    with pytest.raises(RuntimeError, match="downloads are disabled"):
        ras_dss._configure_jvm()
    assert ras_dss._monolith is None
    assert ras_dss._jvm_configured is False
    jar.write_bytes(payloads["jar/fixture-1.jar"])
    ras_dss._configure_jvm()
    assert ras_dss._monolith is downloader
    assert ras_dss._jvm_configured is True


def test_native_extraction_interruption_preserves_retryable_archive(
    runtime, monkeypatch
):
    downloader, trust, payloads = runtime
    archive_name = next(name for name in trust["downloads"] if name.endswith(".zip"))
    seed(downloader, payloads, [archive_name])
    read = zipfile.ZipExtFile.read
    calls = [0]

    def interrupted(source, size=-1):
        calls[0] += 1
        if calls[0] > 1:
            raise OSError("injected extraction interruption")
        return read(source, 3)

    monkeypatch.setattr(zipfile.ZipExtFile, "read", interrupted)
    with pytest.raises(OSError, match="extraction interruption"):
        downloader.download_native_library()
    assert not list(downloader.lib_dir.glob(".extract-*"))
    native_name = next(name for name in trust["artifacts"] if name.startswith("lib/"))
    assert not (downloader.cache_dir / native_name).exists()
    assert (downloader.cache_dir / archive_name).read_bytes() == payloads[archive_name]
    monkeypatch.setattr(zipfile.ZipExtFile, "read", read)
    downloader.download_native_library()
    assert (downloader.cache_dir / native_name).read_bytes() == payloads[native_name]


def test_policy_identity_is_reported_with_verified_inventory(runtime):
    downloader, trust, payloads = runtime
    seed(downloader, payloads, trust["artifacts"])
    assert downloader.TRUST_POLICY == "ras-commander/dss-runtime-trust/1.0.0"
    info = downloader.get_info()
    assert info["trust_policy"] == downloader.TRUST_POLICY
    assert info["installed"] is True
    assert info["downloads_allowed"] is False
