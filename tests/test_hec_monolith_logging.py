import io
import json
import logging
import types
from typing import ClassVar

from test_monolith_integrity import Response, identity, seed
from test_monolith_integrity import (
    runtime as runtime,  # noqa: PLC0414 - pytest fixture re-export
)

from ras_commander.dss import _hec_monolith
from ras_commander.dss._hec_monolith import HecMonolithDownloader

LOGGER_NAME = "ras_commander.dss._hec_monolith"


def test_install_is_quiet_by_default(monkeypatch, caplog, capsys, tmp_path):
    downloader = HecMonolithDownloader(cache_dir=tmp_path)
    calls = []

    states = iter((False, True))
    monkeypatch.setattr(downloader, "is_installed", lambda: next(states))
    monkeypatch.setattr(downloader, "download_jars", lambda: calls.append("jars"))
    monkeypatch.setattr(
        downloader,
        "download_native_library",
        lambda: calls.append("native"),
    )
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    downloader.install()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert calls == ["jars", "native"]

    debug_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER_NAME and record.levelno == logging.DEBUG
    ]
    assert any("Installing HEC Monolith libraries in:" in msg for msg in debug_messages)
    assert "HEC Monolith installation complete" in debug_messages


def test_install_already_installed_is_debug_only(monkeypatch, caplog, capsys, tmp_path):
    downloader = HecMonolithDownloader(cache_dir=tmp_path)
    monkeypatch.setattr(downloader, "is_installed", lambda: True)
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    downloader.install()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    debug_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER_NAME and record.levelno == logging.DEBUG
    ]
    assert "HEC Monolith already installed" in debug_messages


def test_download_file_cached_is_debug_only(monkeypatch, caplog, capsys, runtime):
    downloader, _, payloads = runtime
    seed(downloader, payloads, ["jar/fixture-1.jar"])
    cached = downloader.jar_dir / "fixture-1.jar"
    url = downloader._download_sources()["jar/fixture-1.jar"]

    def fail_request(*_args, **_kwargs):
        raise AssertionError("download_file should not request cached files")

    monkeypatch.setattr(_hec_monolith.requests, "get", fail_request)
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    result = downloader.download_file(url, cached)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert result == cached
    debug_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER_NAME and record.levelno == logging.DEBUG
    ]
    assert "Using cached HEC Monolith file: fixture-1.jar" in debug_messages


def test_download_file_disables_progress_for_non_tty(monkeypatch, capsys, runtime):
    tqdm_calls = []
    updates = []

    class FakeResponse(Response):
        headers: ClassVar[dict[str, str]] = {"content-length": "4"}

        def raise_for_status(self):
            pass

        def __init__(self):
            self.raw = self
            self.chunks = iter((b"ab", b"cd", b""))

        def read1(self, size):
            return next(self.chunks)

        def close(self):
            pass

    class FakeTqdm:
        def __init__(self, **kwargs):
            tqdm_calls.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, amount):
            updates.append(amount)

    monkeypatch.setattr(
        _hec_monolith.requests,
        "get",
        lambda url, **kwargs: FakeResponse(),
    )
    monkeypatch.setattr(_hec_monolith, "tqdm", FakeTqdm)
    monkeypatch.setattr(
        _hec_monolith.sys,
        "stderr",
        types.SimpleNamespace(isatty=lambda: False),
    )

    downloader, trust, _ = runtime
    downloader.allow_downloads = True
    name = "jar/fixture-1.jar"
    trust["artifacts"][name] = trust["downloads"][name] = identity(b"abcd")
    downloader.trust_manifest.write_text(json.dumps(trust))
    dest = downloader.jar_dir / "fixture-1.jar"

    result = downloader.download_file(downloader._download_sources()[name], dest)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert result == dest
    assert dest.read_bytes() == b"abcd"
    assert updates == [2, 2]
    assert tqdm_calls == [
        {
            "total": 4,
            "unit": "B",
            "unit_scale": True,
            "unit_divisor": 1024,
            "leave": False,
            "disable": True,
        }
    ]


def test_download_file_can_explicitly_show_progress(monkeypatch, runtime):
    tqdm_calls = []

    class FakeResponse(Response):
        headers: ClassVar[dict[str, str]] = {"content-length": "1"}

        def raise_for_status(self):
            pass

        def __init__(self):
            self.raw = io.BytesIO(b"x")

    class FakeTqdm:
        def __init__(self, **kwargs):
            tqdm_calls.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, _amount):
            pass

    monkeypatch.setattr(
        _hec_monolith.requests,
        "get",
        lambda url, **kwargs: FakeResponse(),
    )
    monkeypatch.setattr(_hec_monolith, "tqdm", FakeTqdm)

    downloader, trust, _ = runtime
    downloader.allow_downloads = True
    downloader.show_progress = True
    name = "jar/fixture-1.jar"
    trust["artifacts"][name] = trust["downloads"][name] = identity(b"x")
    downloader.trust_manifest.write_text(json.dumps(trust))

    downloader.download_file(
        downloader._download_sources()[name],
        downloader.jar_dir / "fixture-1.jar",
    )

    assert tqdm_calls[0]["disable"] is False
    assert tqdm_calls[0]["leave"] is False
