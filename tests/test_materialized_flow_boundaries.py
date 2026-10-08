"""Regression coverage for exact-target flow materialization and readback."""

from dataclasses import replace
from datetime import datetime
import pytest

from ras_commander import RasBoundaryLink, RasScenario, RasUnsteady
from test_ras_scenario import RAS_EXE, _write_project
from test_rasunsteady_dss_link_selectors import _write_unsteady


def _source(tmp_path, newline="\r\n", modifier="Flow Hydrograph QMult= 0.5\n"):
    path = _write_unsteady(tmp_path / "Model.u01")
    text = path.read_text(encoding="utf-8").replace(
        "Flow Hydrograph= 0\n", "Flow Hydrograph= 0\n" + modifier
    )
    path.write_bytes(text.replace("\n", newline).encode())
    return path


def _link(path, **kwargs):
    return RasUnsteady.set_boundary_dss_link(
        path, river=None, reach=None, station=None, dss_file="materialized.dss",
        dss_path="//SPLIT/FLOW//5MIN/RUN/", sa_2d_name="OuaRiW2",
        bc_line="J_ByDeLout_2", expected_bc_type="Flow Hydrograph", **kwargs,
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("qmult", [None, 0.5, 1.0])
def test_materialized_flow_is_applied_once_and_preserves_other_blocks(tmp_path, newline, qmult):
    modifier = "" if qmult is None else f"Flow Hydrograph QMult= {qmult}\n"
    path = _source(tmp_path, newline, modifier)
    original = path.read_bytes()
    assert _link(path, flow_multiplier_policy="materialized")
    state = RasUnsteady.inspect_boundary_flow(path, sa_2d_name="OuaRiW2", bc_line="J_ByDeLout_2")
    assert state["qmult"] == 1.0
    assert state["qmult_explicit"] == (qmult is not None)
    # The incoming handoff already has half the original flow.
    assert (100.0 * 0.5) * state["qmult"] == 50.0
    other = RasUnsteady.inspect_boundary_flow(path, sa_2d_name="OuaRiW2", bc_line="Other")
    assert other["qmult"] == (1.0 if qmult is None else qmult)
    result = path.read_bytes()
    separator = b"Boundary Location="
    assert result.split(separator)[1] == original.split(separator)[1]
    assert result.split(separator)[3] == original.split(separator)[3]
    assert result.replace(newline.encode(), b"").count(b"\n") == 0
    assert _link(path, flow_multiplier_policy="materialized")
    assert path.read_bytes() == result


def test_default_dss_link_preserves_source_qmult(tmp_path):
    path = _source(tmp_path)
    assert _link(path)
    assert RasUnsteady.inspect_boundary_flow(path, boundary_index=1)["qmult"] == 0.5


@pytest.mark.parametrize("modifier", [
    "Flow Hydrograph QMult= nan\n", "Flow Hydrograph QMult= -1\n",
    "Flow Hydrograph QMult= bad\n", "Flow Hydrograph QMult= inf\n",
    "Flow Hydrograph QMult=0.5\nFlow Hydrograph QMult=0.5\n",
    "Flow Hydrograph QMin= 1\n", "Flow Hydrograph Offset= 2\n",
    "Use Fixed Start Time=True\n", "Is Critical Boundary=True\n",
])
def test_materialized_rejects_unsupported_or_malformed_modifier_without_writing(tmp_path, modifier):
    path = _source(tmp_path, modifier=modifier)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        _link(path, flow_multiplier_policy="materialized")
    assert path.read_bytes() == original


def test_materialized_rejects_mixed_newlines_and_wrong_index_without_writing(tmp_path):
    path = _source(tmp_path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="does not match"):
        _link(path, flow_multiplier_policy="materialized", boundary_index=2)
    assert path.read_bytes() == original
    path.write_bytes(original.replace(b"\r\n", b"\n", 1))
    mixed = path.read_bytes()
    with pytest.raises(ValueError, match="[Mm]ixed newline"):
        _link(path, flow_multiplier_policy="materialized")
    assert path.read_bytes() == mixed


@pytest.mark.skipif(not RAS_EXE.is_file(), reason="HEC-RAS 6.6 not installed")
@pytest.mark.parametrize("interpolation", ["", "Bilinear"])
@pytest.mark.parametrize("inactive_first", [False, True])
def test_workspace_materialized_readback_detects_multiplier_and_link_tampering(tmp_path, interpolation, inactive_first):
    source = _write_project(tmp_path / "source")
    unsteady = source.with_suffix(".u01")
    text = unsteady.read_text(encoding="utf-8").replace(
        "\nFlow Hydrograph= 0\n", "\nFlow Hydrograph= 0\nFlow Hydrograph QMult= 0.5\n"
    )
    if inactive_first:
        parts = text.split("Boundary Location=")
        last, trailer = parts[3].split("Met Point Raster Parameters=")
        text = parts[0] + "Boundary Location=" + last + "Boundary Location=" + parts[1] + "Boundary Location=" + parts[2] + "Met Point Raster Parameters=" + trailer
    text += "Precipitation Mode=Enable\nMet BC=Precipitation|Mode=Gridded\n"
    text += f"Met BC=Precipitation|Gridded Source=DSS\nMet BC=Precipitation|Gridded Interpolation={interpolation}\n"
    unsteady.write_text(text, encoding="utf-8")
    original = unsteady.read_bytes()
    hydrology = tmp_path / "materialized.dss"
    hydrology.write_bytes(b"preparation-only fixture")
    link = RasBoundaryLink(
        mapping_id="junction", dss_path="//JUNCTION/FLOW//5MIN/RUN/",
        expected_bc_type="Flow Hydrograph", sa_2d_name="Area2D", bc_line="Junction",
        boundary_index=2 if inactive_first else 1, flow_multiplier_policy="materialized",
    )
    excess = tmp_path / "excess.dss"
    excess.write_bytes(b"preparation-only forcing fixture")
    workspace = RasScenario.prepare_workspace(
        source, tmp_path / "workspace", "materialized-test", "01", hydrology,
        [link], datetime(2020, 1, 1), datetime(2020, 1, 2), ras_exe_path=RAS_EXE,
        forcing_excess_dss=excess, forcing_excess_pathname="/SHG/AREA/PRECIPITATION///EXCESS/",
        forcing_excess_interpolation="preserve-source",
    )
    assert (RasUnsteady.get_met_precipitation_config(workspace.unsteady_file)["interpolation"] or "") == interpolation
    assert unsteady.read_bytes() == original
    audit = workspace.boundary_flow_preparation[0]
    assert audit["before"]["qmult"] == 0.5
    assert audit["after"]["qmult"] == 1.0
    assert RasScenario.validate_workspace(workspace, [link])["materialized_flow_bindings_match"]
    prepared = workspace.unsteady_file.read_bytes()
    workspace.unsteady_file.write_bytes(prepared.replace(b"QMult= 1 ", b"QMult= 0.5 "))
    with pytest.raises(ValueError, match="materialized_flow_bindings_match"):
        RasScenario.validate_workspace(workspace, [link])
    workspace.unsteady_file.write_bytes(prepared)
    with pytest.raises(ValueError, match="materialized_flow_bindings_match"):
        RasScenario.validate_workspace(workspace, [replace(link, dss_path="//WRONG/FLOW//5MIN/RUN/")])
