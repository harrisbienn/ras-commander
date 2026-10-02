"""Malformed-layout regression cases; retained native DeLoutre validates semantics."""

import hashlib

import h5py
import numpy as np
import pytest

from ras_commander import HdfResultsPlan

BASE = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
LAT = f"{BASE}/Lateral Structures/River Reach 2"
XS = f"{BASE}/Cross Sections"


@pytest.fixture
def coupling_hdf(tmp_path):
    path = tmp_path / "project.p01.hdf"
    with h5py.File(path, "w") as source:
        source.create_dataset(f"{BASE}/Time Date Stamp (ms)", data=[
            b"02Jan3000 00:00:00:000", b"02Jan3000 00:05:00:000"
        ])
        variables = source.create_dataset(f"{LAT}/Structure Variables", data=[[-10000., 10., 11.], [2., 12., 11.]])
        variables.attrs["Variable_Unit"] = np.array([
            [b"Total Flow", b"cfs"], [b"Stage HW", b"ft"], [b"Stage TW", b"ft"]
        ])
        seg = source.create_group(f"{LAT}/HW TW Segments")
        seg.create_dataset("HW TW Station", data=[0., 100.])
        seg.create_dataset("Tailwater Cells", data=[b"15"])
        seg.create_dataset("Headwater River Stations", data=[b"3", b"1"])
        for label, unit in (("Flow", "cfs"), ("Water Surface HW", "ft"), ("Water Surface TW", "ft")):
            seg.create_dataset(label, data=[[1., 2.], [2., 3.]]).attrs["Units"] = unit
        source.create_dataset(f"{XS}/Cross Section Attributes", data=np.array([
            (b"River", b"Reach", b"3"), (b"Other", b"Reach", b"3")
        ], dtype=[(key, "S16") for key in ("River", "Reach", "Station")]))
        source.create_dataset(f"{XS}/Flow", data=[[1000., 9.], [1000., 9.]]).attrs["Variable Units"] = "cfs"
        source.create_dataset(f"{XS}/Flow Volume Cumulative", data=[[0., 0.], [300000., 2700.]]).attrs[
            "Cumulative Volumetric Flow"
        ] = "Feet^3"
    return path


def read(path):
    return HdfResultsPlan.get_coupling_diagnostics(path, ["River Reach 2"], [("River", "Reach", "3")])


def test_preserves_units_connectivity_signed_flow_and_synthetic_time(coupling_hdf):
    before = hashlib.sha256(coupling_hdf.read_bytes()).hexdigest()
    result = read(coupling_hdf)
    assert result["time"][-1] == "3000-01-02T00:05:00"
    lateral = result["laterals"][0]
    assert lateral["tailwater_cell_labels"] == ["15"]
    assert lateral["variables"]["Total Flow"] == {"units": "cfs", "values": [-10000., 2.]}
    assert result["cross_sections"][0]["cumulative_flow_ft3"] == [0., 300000.]
    assert hashlib.sha256(coupling_hdf.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("target,key,value", [
    (f"{XS}/Flow", "Variable Units", "m3/s"),
    (f"{XS}/Flow Volume Cumulative", "Cumulative Volumetric Flow", "m3"),
    (f"{LAT}/HW TW Segments/Water Surface TW", "Units", "m"),
])
def test_rejects_incompatible_units(coupling_hdf, target, key, value):
    with h5py.File(coupling_hdf, "a") as source:
        source[target].attrs[key] = value
    with pytest.raises(ValueError, match="units"):
        read(coupling_hdf)


@pytest.mark.parametrize("case", ["time", "connectivity", "nonfinite", "missing_stage", "missing_output"])
def test_rejects_incomplete_or_ambiguous_output(coupling_hdf, case):
    with h5py.File(coupling_hdf, "a") as source:
        if case == "time":
            source[f"{BASE}/Time Date Stamp (ms)"][1] = source[f"{BASE}/Time Date Stamp (ms)"][0]
        elif case == "connectivity":
            del source[f"{LAT}/HW TW Segments/Tailwater Cells"]
            source.create_dataset(f"{LAT}/HW TW Segments/Tailwater Cells", data=[b"15", b"16"])
        elif case == "nonfinite":
            source[f"{XS}/Flow"][1, 0] = np.nan
        elif case == "missing_stage":
            source[f"{LAT}/HW TW Segments/Water Surface TW"][1, 0] = -9999
        else:
            del source[f"{XS}/Flow Volume Cumulative"]
    with pytest.raises((ValueError, KeyError)):
        read(coupling_hdf)


def test_rejects_unknown_and_duplicate_selection(coupling_hdf):
    with pytest.raises(ValueError, match="Unknown cross section"):
        HdfResultsPlan.get_coupling_diagnostics(coupling_hdf, ["River Reach 2"], [("River", "Reach", "4")])
    with pytest.raises(ValueError, match="distinct lateral"):
        HdfResultsPlan.get_coupling_diagnostics(coupling_hdf, ["River Reach 2"] * 2, [("River", "Reach", "3")])
