"""Accounting arithmetic against the HEC-RAS 6.6 HDF layout.

Small fixtures test the reader contract; the real DeLoutre diagnostic provides
separate domain validation without putting client model bytes in CI.
"""

import hashlib

import h5py
import numpy as np
import pytest

from ras_commander import HdfResultsPlan

BASE = "Results/Unsteady/Summary/Volume Accounting"


@pytest.fixture
def accounting_hdf(tmp_path):
    path = tmp_path / "project.p01.hdf"
    with h5py.File(path, "w") as source:
        group = source.create_group(BASE)
        group.attrs.update(
            {
                "Vol Accounting in": np.bytes_("Acre Feet"),
                "Volume Starting": 10.0,
                "Volume Ending": 22.0,
                "Total Boundary Flux of Water In": 30.0,
                "Total Boundary Flux of Water Out": 20.0,
                "Precipitation Excess (acre feet)": 12.0,
                "Error": 2.0,
                "Error Percent": 5.0,
            }
        )
        area = source.create_group(f"{BASE}/Volume Accounting 2D/Basin")
        area.attrs.update(
            {
                "Vol Accounting in": np.bytes_("Acre Feet"),
                "Vol Starting": 0.0,
                "Vol Ending": 11.0,
                "Cum Inflow": 30.0,
                "Cum Outflow": 20.0,
                "Precip Excess (acre feet)": 12.0,
                "Error": 1.0,
                "Error Percent": 100 / 30,
            }
        )
    return path


def test_accounting_preserves_precipitation_and_does_not_double_count(accounting_hdf):
    before = hashlib.sha256(accounting_hdf.read_bytes()).hexdigest()
    result = HdfResultsPlan.get_volume_accounting_diagnostics(accounting_hdf)
    assert result["units"] == "Acre Feet"
    assert result["overall_reconstructed_error"] == 2
    assert result["two_d"][0]["reconstructed_error"] == 1
    assert result["sum_reported_two_d_errors"] == 1
    assert result["overall_minus_two_d_errors"] == 1
    assert result["two_d"][0]["raw"]["Precip Excess (acre feet)"] == 12
    assert result["one_d"] is None
    assert result["saved_computation_series"] == {"Basin": {}}
    assert hashlib.sha256(accounting_hdf.read_bytes()).hexdigest() == before


def test_one_d_only_is_not_a_measured_one_d_error(accounting_hdf):
    with h5py.File(accounting_hdf, "a") as source:
        del source[f"{BASE}/Volume Accounting 2D"]
        source.create_group(f"{BASE}/Volume Accounting 1D").attrs["Flow US In"] = 30.0
    result = HdfResultsPlan.get_volume_accounting_diagnostics(accounting_hdf)
    assert result["one_d"]["Flow US In"] == 30
    assert result["two_d"] == []
    assert result["overall_minus_two_d_errors"] is None


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_rejects_nonfinite_balance_values(accounting_hdf, value):
    with h5py.File(accounting_hdf, "a") as source:
        source[BASE].attrs["Volume Ending"] = value
    with pytest.raises(ValueError, match="finite number"):
        HdfResultsPlan.get_volume_accounting_diagnostics(accounting_hdf)


def test_rejects_mixed_units(accounting_hdf):
    with h5py.File(accounting_hdf, "a") as source:
        source[f"{BASE}/Volume Accounting 2D/Basin"].attrs["Vol Accounting in"] = "1000 m3"
    with pytest.raises(ValueError, match="units differ"):
        HdfResultsPlan.get_volume_accounting_diagnostics(accounting_hdf)


def test_preserves_signed_losses(accounting_hdf):
    with h5py.File(accounting_hdf, "a") as source:
        area = source[f"{BASE}/Volume Accounting 2D/Basin"]
        area.attrs["Vol Ending"] = 8.0
        area.attrs["Error"] = -2.0
    result = HdfResultsPlan.get_volume_accounting_diagnostics(accounting_hdf)
    assert result["two_d"][0]["reconstructed_error"] == -2
    assert result["overall_minus_two_d_errors"] == 4


def test_missing_accounting_is_not_zero_error(accounting_hdf):
    with h5py.File(accounting_hdf, "a") as source:
        del source[BASE]
    with pytest.raises(KeyError, match="Missing HDF group"):
        HdfResultsPlan.get_volume_accounting_diagnostics(accounting_hdf)


def test_missing_required_area_term_fails(accounting_hdf):
    with h5py.File(accounting_hdf, "a") as source:
        del source[f"{BASE}/Volume Accounting 2D/Basin"].attrs["Cum Outflow"]
    with pytest.raises(KeyError):
        HdfResultsPlan.get_volume_accounting_diagnostics(accounting_hdf)


def test_zero_saved_error_is_not_substituted_for_final_error(accounting_hdf):
    with h5py.File(accounting_hdf, "a") as source:
        info = source.create_group("Plan Data/Plan Information")
        info.attrs["Simulation Start Time"] = np.bytes_("02JAN3000 00:00:00")
        series = source.create_group(
            "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
        )
        series.create_dataset("Time", data=[0.0, 0.5, 1.0])
        error = series.create_dataset(
            "2D Flow Areas/Basin/Computations/Volume Error", data=np.zeros((3, 1))
        )
        error.attrs["Units"] = np.bytes_("ft^3")
    result = HdfResultsPlan.get_volume_accounting_diagnostics(accounting_hdf)
    saved = result["saved_computation_series"]["Basin"]["Volume Error"]
    assert saved["times"] == ["3000-01-02T00:00:00", "3000-01-02T12:00:00", "3000-01-03T00:00:00"]
    assert saved["values"] == [0, 0, 0]
    assert saved["units"] == "ft^3"
    assert result["two_d"][0]["raw"]["Error"] == 1
