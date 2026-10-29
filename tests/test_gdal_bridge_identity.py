"""Runtime identity gates use real directories and Windows junctions."""

import importlib
import os
import subprocess
import sys

import pytest

from ras_commander import _gdal_runtime

clr_bootstrap = importlib.import_module("ras_commander.dotnet.clr_bootstrap")


def make_runtime(root):
    (root / "GDAL/bin64").mkdir(parents=True)
    (root / "GDAL/common/data").mkdir(parents=True)
    (root / "RasMapperLib.dll").touch()
    return root


def link_directory(link, target):
    if os.name == "nt":
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "$ErrorActionPreference='Stop'; New-Item -ItemType Junction "
             "-Path $env:TEST_GDAL_LINK -Target $env:TEST_GDAL_TARGET | Out-Null"],
            env=dict(os.environ, TEST_GDAL_LINK=str(link), TEST_GDAL_TARGET=str(target)),
            check=True, capture_output=True,
        )
    else:
        link.symlink_to(target, target_is_directory=True)


@pytest.mark.parametrize("kind", ["other_runtime", "copied_directory", "dangling", "file"])
def test_wrong_existing_bridge_preserved_before_configuration(tmp_path, monkeypatch, kind):
    selected = make_runtime(tmp_path / "selected")
    other = make_runtime(tmp_path / "other")
    python = tmp_path / "python"
    python.mkdir()
    bridge = python / "GDAL"
    if kind == "other_runtime":
        link_directory(bridge, other / "GDAL")
    elif kind == "copied_directory":
        make_runtime(python)
    elif kind == "dangling":
        missing = tmp_path / "missing"
        missing.mkdir()
        link_directory(bridge, missing)
        missing.rmdir()
    else:
        bridge.write_text("preserve this file")
    before = bridge.lstat()
    calls = []
    monkeypatch.setattr(_gdal_runtime.platform, "system", lambda: "Windows")
    def record_configuration(path):
        calls.append(path)
        return _gdal_runtime.resolve_hecras_gdal_paths(path)

    monkeypatch.setattr(_gdal_runtime, "configure_hecras_gdal_runtime", record_configuration)
    monkeypatch.setattr(_gdal_runtime.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not replace bridge"))

    with pytest.raises(RuntimeError, match="GDAL bridge.*isolated interpreter"):
        _gdal_runtime.configure_rasmapper_gdal_bridge(selected, python)

    assert calls == []
    assert bridge.lstat() == before


def test_all_interpreter_bridges_checked_before_creating_any(tmp_path, monkeypatch):
    selected = make_runtime(tmp_path / "selected")
    other = make_runtime(tmp_path / "other")
    venv = tmp_path / "venv"
    base = tmp_path / "base"
    venv.mkdir()
    base.mkdir()
    link_directory(base / "GDAL", other / "GDAL")
    monkeypatch.setattr(_gdal_runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(_gdal_runtime, "_candidate_python_dirs", lambda value: [venv, base])
    monkeypatch.setattr(_gdal_runtime.subprocess, "run", lambda *args, **kwargs: pytest.fail("preflight all bridges"))
    with pytest.raises(RuntimeError, match="isolated interpreter"):
        _gdal_runtime.configure_rasmapper_gdal_bridge(selected)
    assert not (venv / "GDAL").exists()


def test_matching_bridge_is_idempotent_and_accepts_runtime_alias(tmp_path, monkeypatch):
    selected = make_runtime(tmp_path / "selected")
    alias = tmp_path / "alias"
    link_directory(alias, selected)
    python = tmp_path / "python"
    python.mkdir()
    link_directory(python / "GDAL", selected / "GDAL")
    monkeypatch.setattr(_gdal_runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(_gdal_runtime.subprocess, "run", lambda *args, **kwargs: pytest.fail("already configured"))
    assert _gdal_runtime.ensure_python_gdal_junction(alias, python) is True
    assert _gdal_runtime.ensure_python_gdal_junction(selected, python) is True


def test_clr_refuses_mismatch_before_loading_assemblies(tmp_path, monkeypatch):
    selected = make_runtime(tmp_path / "selected")
    other = make_runtime(tmp_path / "other")
    python = tmp_path / "python"
    python.mkdir()
    link_directory(python / "GDAL", other / "GDAL")
    references = []

    class FakeClr:
        AddReference = staticmethod(references.append)

    monkeypatch.setitem(sys.modules, "clr", FakeClr)
    monkeypatch.setattr(clr_bootstrap, "_CLR_LOADED", False)
    monkeypatch.setattr(_gdal_runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(_gdal_runtime, "_candidate_python_dirs", lambda value: [python])
    with pytest.raises(RuntimeError, match="GDAL") as caught:
        clr_bootstrap.load_clr(selected)
    assert "isolated interpreter" in str(caught.value.__cause__)
    assert references == []


@pytest.mark.parametrize("package", ["geom.GeomMesh", "terrain.RasTerrainMod"])
def test_public_setup_preserves_search_paths_on_mismatch(tmp_path, monkeypatch, package):
    selected = make_runtime(tmp_path / "selected")
    other = make_runtime(tmp_path / "other")
    python = tmp_path / "python"
    python.mkdir()
    link_directory(python / "GDAL", other / "GDAL")
    module = importlib.import_module(f"ras_commander.{package}")
    monkeypatch.setattr(_gdal_runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(module, "configure_hecras_gdal_runtime", lambda path: pytest.fail("guard must run first"))
    before = dict(os.environ)
    if package.startswith("geom"):
        result = module.GeomMesh.setup_gdal_bridge(hecras_dir=selected, python_dir=python)
    else:
        monkeypatch.setattr(module.RasTerrainMod, "_find_hecras_path", staticmethod(lambda: selected))
        result = module.RasTerrainMod.setup_gdal_bridge(python_dir=python)
    assert result is False
    assert dict(os.environ) == before


def test_boolean_helper_preserves_mismatched_bridge(tmp_path, monkeypatch):
    selected = make_runtime(tmp_path / "selected")
    other = make_runtime(tmp_path / "other")
    python = tmp_path / "python"
    python.mkdir()
    link_directory(python / "GDAL", other / "GDAL")
    monkeypatch.setattr(_gdal_runtime.platform, "system", lambda: "Windows")
    assert _gdal_runtime.ensure_python_gdal_junction(selected, python) is False
    assert (python / "GDAL").samefile(other / "GDAL")


@pytest.mark.skipif(os.name != "nt", reason="PowerShell junction creation on Windows")
def test_create_missing_bridge_with_literal_path_characters(tmp_path):
    selected = make_runtime(tmp_path / "RAS space $value 'quoted'")
    python = tmp_path / "Python space $value 'quoted'"
    python.mkdir()
    assert _gdal_runtime.ensure_python_gdal_junction(selected, python) is True
    assert (python / "GDAL").samefile(selected / "GDAL")
    assert _gdal_runtime.ensure_python_gdal_junction(selected, python) is True


@pytest.mark.parametrize("package", ["geom.GeomMesh", "terrain.RasTerrainMod"])
def test_explicit_search_path_only_option_preserved(tmp_path, monkeypatch, package):
    selected = make_runtime(tmp_path / "selected")
    module = importlib.import_module(f"ras_commander.{package}")
    configured = []
    monkeypatch.setattr(_gdal_runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(module, "configure_hecras_gdal_runtime", configured.append)
    monkeypatch.setattr(module, "configure_rasmapper_gdal_bridge",
                        lambda *args: pytest.fail("explicit search-path-only option must be preserved"))
    if package.startswith("geom"):
        assert module.GeomMesh.setup_gdal_bridge(hecras_dir=selected, create_junction=False)
    else:
        monkeypatch.setattr(module.RasTerrainMod, "_find_hecras_path", staticmethod(lambda: selected))
        assert module.RasTerrainMod.setup_gdal_bridge(create_junction=False)
    assert configured == [selected]
