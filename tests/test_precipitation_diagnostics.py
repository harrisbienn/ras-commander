"""Reader invariants; separate real DeLoutre runs validate the native layout."""

import hashlib

import h5py
import numpy as np
import pytest

from ras_commander import HdfResultsPlan

MET = "Event Conditions/Meteorology/Precipitation"
BASE = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
CV = f"{BASE}/Cross Sections Control Volume"
ACCOUNTING = "Results/Unsteady/Summary/Volume Accounting/Volume Accounting 1D"


@pytest.fixture
def precipitation_hdf(tmp_path):
    path = tmp_path / "project.p01.hdf"
    with h5py.File(path, "w") as source:
        met = source.create_group(MET)
        met.attrs.update(Mode="Gridded", **{"Data Type": "per-cum", "Units": "in"})
        met.create_dataset("Values", data=[[1.0, 2.0]])
        met.create_dataset("Timestamp", data=[b"02Jan3000 00:05:00"], dtype="S22")
        met.create_dataset("Cross Sections/Weights", data=[0.5, 0.5])
        source.create_dataset(f"{BASE}/Time Date Stamp (ms)", data=[
            b"02Jan3000 00:00:00:000", b"02Jan3000 00:05:00:000"
        ])
        source.create_dataset(f"{CV}/Cumulative Precipitation Depth", data=[[0.0, 1.0], [12.0, 7.0]])
        source[f"{CV}/Cumulative Precipitation Depth"].attrs["Units"] = "in"
        dtype = [(name, "S16") for name in ("River", "Reach", "Station US", "Station DS")]
        dtype.append(("Surface Area (ft^2)", "f4"))
        source.create_dataset(f"{CV}/XS CV Attributes", data=np.array([
            (b"River", b"Reach", b"3", b"2", 43560.0),
            (b"River", b"Reach", b"2", b"1", 87120.0),
        ], dtype=dtype))
        source.create_group(ACCOUNTING).attrs.update({
            "Vol Accounting in": "Acre Feet", "Precip Excess (acre feet)": 3.0
        })
    return path


def test_area_weighted_depth_is_independent_of_summary(precipitation_hdf):
    before = hashlib.sha256(precipitation_hdf.read_bytes()).hexdigest()
    result = HdfResultsPlan.get_precipitation_diagnostics(precipitation_hdf)
    assert result["saved_final_volume_af"] == pytest.approx(13 / 6)
    assert result["saved_window_volume_af"] == pytest.approx(2)
    assert result["saved_final_minus_native_af"] == pytest.approx(-5 / 6)
    assert result["native_1d_precipitation_af"] == 3
    assert result["time"][-1] == "3000-01-02T00:05:00"
    assert len(result["control_volumes"]) == 2
    assert hashlib.sha256(precipitation_hdf.read_bytes()).hexdigest() == before


def test_spatial_weight_change_changes_fingerprint(precipitation_hdf):
    before = HdfResultsPlan.get_precipitation_diagnostics(precipitation_hdf)
    with h5py.File(precipitation_hdf, "a") as source:
        source[f"{MET}/Cross Sections/Weights"][0] = 0.6
    after = HdfResultsPlan.get_precipitation_diagnostics(precipitation_hdf)
    assert before["meteorology_datasets"]["Values"] == after["meteorology_datasets"]["Values"]
    assert before["meteorology_datasets"]["Cross Sections/Weights"] != after["meteorology_datasets"]["Cross Sections/Weights"]


@pytest.mark.parametrize("target,key,value", [
    (MET, "Units", "mm"), (MET, "Data Type", "per-aver"),
    (f"{CV}/Cumulative Precipitation Depth", "Units", "ft"),
    (ACCOUNTING, "Vol Accounting in", "m3"),
    (ACCOUNTING, "Precip Excess (acre feet)", np.nan),
])
def test_rejects_ambiguous_units_and_nonfinite_summary(precipitation_hdf, target, key, value):
    with h5py.File(precipitation_hdf, "a") as source:
        source[target].attrs[key] = value
    with pytest.raises(ValueError):
        HdfResultsPlan.get_precipitation_diagnostics(precipitation_hdf)


@pytest.mark.parametrize("target", [f"{MET}/Values", f"{CV}/Cumulative Precipitation Depth"])
def test_rejects_nonfinite_depths(precipitation_hdf, target):
    with h5py.File(precipitation_hdf, "a") as source:
        source[target][0, 0] = np.nan
    with pytest.raises(ValueError, match="[Nn]onfinite|Invalid precipitation"):
        HdfResultsPlan.get_precipitation_diagnostics(precipitation_hdf)


def test_missing_output_is_not_zero(precipitation_hdf):
    with h5py.File(precipitation_hdf, "a") as source:
        del source[f"{CV}/Cumulative Precipitation Depth"]
    with pytest.raises(KeyError):
        HdfResultsPlan.get_precipitation_diagnostics(precipitation_hdf)


def test_duplicate_control_volumes_rejected(precipitation_hdf):
    with h5py.File(precipitation_hdf, "a") as source:
        source[f"{CV}/XS CV Attributes"][1] = source[f"{CV}/XS CV Attributes"][0]
    with pytest.raises(ValueError, match="Duplicate"):
        HdfResultsPlan.get_precipitation_diagnostics(precipitation_hdf)


def test_repeated_time_rejected(precipitation_hdf):
    with h5py.File(precipitation_hdf, "a") as source:
        source[f"{BASE}/Time Date Stamp (ms)"][1] = source[f"{BASE}/Time Date Stamp (ms)"][0]
    with pytest.raises(ValueError, match="strictly increase"):
        HdfResultsPlan.get_precipitation_diagnostics(precipitation_hdf)


def test_variable_length_forcing_not_hashed_as_memory_addresses(precipitation_hdf):
    with h5py.File(precipitation_hdf, "a") as source:
        del source[f"{MET}/Timestamp"]
        source.create_dataset(f"{MET}/Timestamp", data=[b"02Jan3000 00:05:00"])
    with pytest.raises(ValueError, match="variable-length"):
        HdfResultsPlan.get_precipitation_diagnostics(precipitation_hdf)


@pytest.mark.parametrize("area", [0.0, -1.0, np.nan])
def test_invalid_control_volume_area_rejected(precipitation_hdf, area):
    with h5py.File(precipitation_hdf, "a") as source:
        row = source[f"{CV}/XS CV Attributes"][0]
        row["Surface Area (ft^2)"] = area
        source[f"{CV}/XS CV Attributes"][0] = row
    with pytest.raises(ValueError, match="Invalid precipitation"):
        HdfResultsPlan.get_precipitation_diagnostics(precipitation_hdf)


def test_depth_time_shape_mismatch_rejected(precipitation_hdf):
    with h5py.File(precipitation_hdf, "a") as source:
        del source[f"{CV}/Cumulative Precipitation Depth"]
        source.create_dataset(f"{CV}/Cumulative Precipitation Depth", data=[[0.0, 1.0]])
        source[f"{CV}/Cumulative Precipitation Depth"].attrs["Units"] = "in"
    with pytest.raises(ValueError, match="shape/time mismatch"):
        HdfResultsPlan.get_precipitation_diagnostics(precipitation_hdf)
