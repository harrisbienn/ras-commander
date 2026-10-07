"""Explicit-file IC preparation must not depend on or mutate global projects."""

import importlib
from pathlib import Path

import pandas as pd
import pytest

from ras_commander import RasPrj, RasUnsteady, RasUtils
from ras_commander.usgs.initial_conditions import InitialConditions


@pytest.fixture(autouse=True)
def uninitialized_global(monkeypatch):
    module = importlib.import_module("ras_commander.RasUnsteady")
    monkeypatch.setattr(module, "ras", RasPrj())
    monkeypatch.setattr(importlib.import_module("ras_commander.RasPlan"), "ras", RasPrj())


def write_file(tmp_path, newline="\r\n"):
    path = tmp_path / "Example.u01"
    path.write_bytes(newline.join([
        "Flow Title=Example", "Program Version=6.60", "Use Restart=-1",
        "Restart Filename=original.rst", "Prior WS Filename=original.p01",
        "Prior WS Profile=Original", "Initial Storage Elev=Lake,10",
        "IC Point Elev=Reference Point,11,-1", "Boundary Location=Downstream",
        "Friction Slope=0.001", "",
    ]).encode())
    return path


@pytest.mark.parametrize("newline", ["\r\n", "\n"])
@pytest.mark.parametrize("as_frame", [False, True])
def test_explicit_path_wrapper_updates_rows_and_method(tmp_path, newline, as_frame):
    path = write_file(tmp_path, newline)
    entries = [{"type": "storage", "area_name": "Lake", "value": 12.5}]
    RasUnsteady.set_initial_conditions(str(path), pd.DataFrame(entries) if as_frame else entries)
    assert RasUnsteady.get_initial_flow_method(path)["method"] == "initial_flow_distribution"
    assert RasUnsteady.get_initial_storage_elevations(path).iloc[0]["value"] == 12.5
    assert RasUnsteady.get_initial_point_elevations(path).elevation.tolist() == [11.0]
    assert "Restart Filename=" not in path.read_text()
    assert "Prior WS Filename=" not in path.read_text()
    assert "Friction Slope=0.001" in path.read_text()
    assert RasUtils._detect_text_newline(path) == newline
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("entries,auto", [([], True), ([{"type": "storage", "area_name": "Lake", "value": 12}], False)])
def test_empty_or_disabled_auto_selection_preserves_method(tmp_path, entries, auto):
    path = write_file(tmp_path)
    RasUnsteady.set_initial_conditions(path, entries, auto_set_method=auto)
    assert RasUnsteady.get_initial_flow_method(path)["method"] == "restart_file"
    assert "Restart Filename=original.rst" in path.read_text()
    assert len(RasUnsteady.get_initial_conditions(path)) == len(entries)


@pytest.mark.parametrize("newline", ["\r\n", "\n"])
@pytest.mark.parametrize("method,kwargs", [
    ("restart_file", {"restart_filename": "other.rst"}),
    ("prior_ws", {"prior_ws_filename": "other.p01", "prior_ws_profile": "Profile"}),
    ("initial_flow_distribution", {}), ("none", {}),
])
def test_explicit_method_roundtrip(tmp_path, newline, method, kwargs):
    path = write_file(tmp_path, newline)
    RasUnsteady.set_initial_flow_method(path, method, **kwargs)
    assert RasUnsteady.get_initial_flow_method(path)["method"] == method
    assert RasUtils._detect_text_newline(path) == newline
    assert RasUnsteady.get_initial_point_elevations(path).elevation.tolist() == [11.0]


@pytest.mark.parametrize("failure", ["bad-entry", "mixed-newlines", "missing-header", "method-failure", "replace-failure"])
def test_failure_leaves_original_bytes_and_no_temporary_files(tmp_path, monkeypatch, failure):
    path = write_file(tmp_path)
    entries = [{"type": "storage", "area_name": "Lake", "value": 12}]
    if failure == "bad-entry":
        entries.append({"type": "storage", "value": 15})
    elif failure == "mixed-newlines":
        path.write_bytes(path.read_bytes() + b"Additional=Value\n")
    elif failure == "missing-header":
        path.write_bytes(b"Flow Title=MissingHeaders\r\nBoundary Location=Downstream\r\n")
    elif failure == "method-failure":
        def fail(*args, **kwargs):
            raise ValueError("injected method preparation failure")
        monkeypatch.setattr(RasUnsteady, "set_initial_flow_method", fail)
    else:
        import os
        original_replace = os.replace

        def fail_final(source, destination):
            if Path(destination) == path:
                raise PermissionError("injected replacement failure")
            original_replace(source, destination)
        monkeypatch.setattr(os, "replace", fail_final)
    original = path.read_bytes()
    with pytest.raises((ValueError, PermissionError)):
        RasUnsteady.set_initial_conditions(path, entries)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_direct_writer_rejects_mixed_input_without_changing_it(tmp_path):
    path = write_file(tmp_path)
    path.write_bytes(path.read_bytes() + b"Additional=Value\n")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Mixed newline"):
        InitialConditions.write_initial_conditions(path, [])
    assert path.read_bytes() == original


@pytest.mark.parametrize("explicit_context", [False, True])
def test_number_resolution_refreshes_only_selected_project(tmp_path, monkeypatch, explicit_context):
    path = write_file(tmp_path)
    project = RasPrj()
    project.prj_file = tmp_path / "Example.prj"
    project.prj_file.write_text("Proj Title=Example\nUnsteady File=u01\n", encoding="utf-8")
    project.project_folder = tmp_path
    project.project_name = "Example"
    project.initialized = True
    if not explicit_context:
        monkeypatch.setattr(importlib.import_module("ras_commander.RasUnsteady"), "ras", project)
        monkeypatch.setattr(importlib.import_module("ras_commander.RasPlan"), "ras", project)
    context = project if explicit_context else None
    RasUnsteady.set_initial_conditions("01", [{"type": "storage", "area_name": "Lake", "value": 14}], ras_object=context)
    assert RasUnsteady.get_initial_flow_method("01", ras_object=context)["method"] == "initial_flow_distribution"
    assert RasUnsteady.get_initial_conditions(path).iloc[0]["value"] == 14
    assert project.unsteady_df.iloc[0]["unsteady_number"] == "01"


def test_explicit_file_does_not_refresh_unrelated_initialized_global(tmp_path, monkeypatch):
    path = write_file(tmp_path)
    unrelated = RasPrj()
    unrelated.initialized = True

    def unexpected_refresh():
        pytest.fail("explicit path must not refresh an unrelated global project")

    monkeypatch.setattr(unrelated, "get_unsteady_entries", unexpected_refresh)
    monkeypatch.setattr(importlib.import_module("ras_commander.RasUnsteady"), "ras", unrelated)
    RasUnsteady.set_initial_conditions(path, [{"type": "storage", "area_name": "Lake", "value": 13}])
    assert RasUnsteady.get_initial_flow_method(path)["method"] == "initial_flow_distribution"


@pytest.mark.parametrize("method,kwargs", [("unknown", {}), ("restart_file", {}), ("prior_ws", {})])
def test_invalid_method_arguments_do_not_change_file(tmp_path, method, kwargs):
    path = write_file(tmp_path)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        RasUnsteady.set_initial_flow_method(path, method, **kwargs)
    assert path.read_bytes() == original
