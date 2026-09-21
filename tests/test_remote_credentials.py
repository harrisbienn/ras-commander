"""Credential diagnostics use synthetic values and never contact a remote host."""

import logging
import subprocess
from types import SimpleNamespace

import pytest

from ras_commander.remote import Utils


@pytest.mark.parametrize("failure", ["timeout", "nonzero", "launch", "called_process"])
def test_share_authentication_errors_do_not_disclose_password(
    monkeypatch, caplog, failure
):
    password = "review-password-sentinel'\"\\\n\u00e9"

    def run(command, **kwargs):
        if "/delete" in command:
            return SimpleNamespace(returncode=0)
        assert command[-1] == password
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                command, 30, output=password, stderr=password
            )
        if failure == "launch":
            raise OSError(repr(command))
        if failure == "called_process":
            raise subprocess.CalledProcessError(2, command, stderr=password)
        return SimpleNamespace(returncode=2, stdout=password, stderr=repr(command))

    monkeypatch.setattr(Utils.subprocess, "run", run)
    with caplog.at_level(logging.DEBUG):
        assert not Utils.authenticate_network_share(
            r"\\review-host\share", "review-user", password
        )
    assert password not in caplog.text
    assert repr(password)[1:-1] not in caplog.text
    assert "review-host" in caplog.text


@pytest.mark.parametrize("returncode,stderr", [(0, ""), (2, "System error 1219")])
def test_share_authentication_retains_success_behavior(monkeypatch, returncode, stderr):
    monkeypatch.setattr(
        Utils.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=returncode,
            stdout="",
            stderr=stderr,
        ),
    )
    assert Utils.authenticate_network_share(
        r"\\review-host\share", "review-user", "synthetic"
    )
