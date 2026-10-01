"""Analytic sparse-weight budgets and explicit failure cases for imported rain."""

import hashlib

import h5py
import numpy as np
import pytest
from pyproj import CRS

from ras_commander import HdfResultsPlan

MET = "Event Conditions/Meteorology/Precipitation"
BASE = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
SURFACE = "Geometry/Cross Section Interpolation Surfaces"


@pytest.fixture
def rain_hdf(tmp_path):
    path = tmp_path / "rain.p01.hdf"
    with h5py.File(path, "w") as f:
        f.attrs["Projection"] = CRS.from_epsg(2276).to_wkt()
        m = f.create_group(MET)
        m.attrs.update({"Mode": "Gridded", "Data Type": "per-cum", "Units": "in",
                        "Projection": CRS.from_epsg(5070).to_wkt(),
                        "Raster Rows": 2, "Raster Cols": 2, "Raster Cellsize": 1.0})
        m.create_dataset("Values", data=[[1., 2., 3., 4.], [0., 1., 0., 1.]])
        m.create_dataset("Timestamp", data=[b"02Jan3000 00:05:00.000", b"02Jan3000 00:10:00.000"])
        f.create_dataset(f"{BASE}/Time Date Stamp (ms)", data=[
            b"02Jan3000 00:00:00:000", b"02Jan3000 00:05:00:000", b"02Jan3000 00:10:00:000"])
        f.create_dataset(f"{BASE}/Cross Sections/Water Surface", data=[[1, 1]] * 3).attrs["Variable Units"] = "Feet"
        f.create_dataset("Geometry/2D Flow Areas/Attributes",
                         data=np.array([(b"Mesh", 2)], dtype=[("Name", "S16"), ("Cell Count", "i4")]))
        f.create_dataset("Geometry/2D Flow Areas/Mesh/Cells Surface Area", data=[43560., 87120., 999999.])
        m.create_dataset("2D Flow Areas/Mesh/Cell Info", data=[[0, 1], [1, 1]])
        m.create_dataset("2D Flow Areas/Mesh/Cell Indexes", data=[0, 3])
        m.create_dataset("2D Flow Areas/Mesh/Cell Weights", data=[1., 1.])
        f.create_dataset("Geometry/Cross Sections/Attributes", data=np.array(
            [(b"R", b"A", b"20"), (b"R", b"A", b"10")],
            dtype=[(key, "S16") for key in ("River", "Reach", "RS")]))
        f.create_dataset(f"{BASE}/Cross Sections Control Volume/XS CV Attributes", data=np.array(
            [(b"R", b"A", b"20", b"10", 43560.)], dtype=[(key, "S16") for key in
            ("River", "Reach", "Station US", "Station DS")] + [("Surface Area (ft^2)", "f4")]))
        f.create_dataset(f"{SURFACE}/XSIDs", data=[[0, 1]])
        m.create_dataset("Cross Section Interpolation Surfaces/Info", data=[[0, 2]])
        m.create_dataset("Cross Section Interpolation Surfaces/Indexes", data=[1, 2])
        m.create_dataset("Cross Section Interpolation Surfaces/Weights", data=[0.5, 0.5])
    return path


def test_receiving_volume_does_not_need_summary_or_saved_rain(rain_hdf):
    before = hashlib.sha256(rain_hdf.read_bytes()).hexdigest()
    r = HdfResultsPlan.get_precipitation_receiving_diagnostics(rain_hdf, value_semantics="interval_depth")
    assert r["total_two_d_volume_af"] == pytest.approx(11 / 12)
    assert r["one_d"]["total_volume_af"] == pytest.approx(3 / 12)
    assert r["total_receiving_volume_af"] == pytest.approx(14 / 12)
    assert r["grid_total_volume_af"] == pytest.approx(12 * .0254 / 1233.48183754752)
    assert r["two_d"]["Mesh"]["excluded_nonphysical_area_rows"] == 1
    assert r["one_d"]["receivers"][0]["identity"] == ("R", "A", "20", "10")
    assert r["end"] == "3000-01-02T00:10:00"
    assert hashlib.sha256(rain_hdf.read_bytes()).hexdigest() == before


def test_value_semantics_are_explicit_and_cumulative_mode_differences(rain_hdf):
    with pytest.raises(ValueError, match="semantics"):
        HdfResultsPlan.get_precipitation_receiving_diagnostics(rain_hdf, value_semantics="guess")
    with pytest.raises(ValueError, match="decreases"):
        HdfResultsPlan.get_precipitation_receiving_diagnostics(rain_hdf, value_semantics="cumulative_depth")
    with h5py.File(rain_hdf, "a") as f:
        f[f"{MET}/Values"][:] = [[1., 2., 3., 4.], [1., 3., 3., 5.]]
    r = HdfResultsPlan.get_precipitation_receiving_diagnostics(rain_hdf, value_semantics="cumulative_depth")
    assert r["total_receiving_volume_af"] == pytest.approx(14 / 12)


@pytest.mark.parametrize("defect", ["units", "geometry_crs", "nan", "negative", "time", "weights", "index",
                                  "slice", "count", "area", "surface", "identity", "missing_area"])
def test_invalid_units_weights_coverage_and_identities_fail_closed(rain_hdf, defect):
    with h5py.File(rain_hdf, "a") as f:
        if defect == "units":
            f[MET].attrs["Units"] = "mm"
        elif defect == "geometry_crs":
            f.attrs["Projection"] = CRS.from_epsg(5070).to_wkt()
        elif defect in ("nan", "negative"):
            f[f"{MET}/Values"][0, 0] = np.nan if defect == "nan" else -1
        elif defect == "time":
            f[f"{MET}/Timestamp"][1] = b"02Jan3000 00:15:00.000"
        elif defect == "weights":
            f[f"{MET}/2D Flow Areas/Mesh/Cell Weights"][0] = .9
        elif defect == "index":
            f[f"{MET}/2D Flow Areas/Mesh/Cell Indexes"][0] = 4
        elif defect == "slice":
            f[f"{MET}/2D Flow Areas/Mesh/Cell Info"][1] = [0, 1]
        elif defect == "count":
            f["Geometry/2D Flow Areas/Attributes"][0] = (b"Mesh", 3)
        elif defect == "area":
            f["Geometry/2D Flow Areas/Mesh/Cells Surface Area"][0] = 0
        elif defect == "surface":
            f[f"{SURFACE}/XSIDs"][0] = [1, 0]
        elif defect == "identity":
            f["Geometry/Cross Sections/Attributes"][1] = (b"R", b"A", b"20")
        else:
            del f[f"{MET}/2D Flow Areas/Mesh"]
    with pytest.raises(ValueError):
        HdfResultsPlan.get_precipitation_receiving_diagnostics(rain_hdf, value_semantics="interval_depth")
