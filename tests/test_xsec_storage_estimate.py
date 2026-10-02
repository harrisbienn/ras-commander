"""Analytical profiles test storage integration; Muncie supplies real validation."""

import hashlib

import h5py
import numpy as np
import pytest

from ras_commander import HdfResultsXsec

BASE = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
GEOM = "Geometry/Cross Sections"


@pytest.fixture
def storage_hdf(tmp_path):
    path = tmp_path / "storage.p01.hdf"
    with h5py.File(path, "w") as hdf:
        hdf.create_dataset(f"{BASE}/Time Date Stamp (ms)",
                           data=[b"02Jan2020 00:00:00:000", b"02Jan2020 00:05:00:000"])
        attrs = [(name, "S16") for name in ("River", "Reach", "RS")]
        attrs += [(name, "f4") for name in ("Left Bank", "Right Bank", "Len Left", "Len Channel", "Len Right")]
        hdf.create_dataset(f"{GEOM}/Attributes", data=np.array([
            (b"R", b"A", b"2", 0, 10, 100, 100, 100),
            (b"R", b"A", b"1", 0, 10, 0, 0, 0)], dtype=attrs))
        # V-shaped section: area = depth**2 / 2 for depths up to 10.
        hdf.create_dataset(f"{GEOM}/Station Elevation Info", data=[[0, 3], [3, 3]])
        hdf.create_dataset(f"{GEOM}/Station Elevation Values", data=[[0, 10], [5, 0], [10, 10]] * 2)
        hdf.create_dataset(f"{BASE}/Cross Sections/Cross Section Attributes", data=np.array([
            (b"R", b"A", b"1"), (b"R", b"A", b"2")],
            dtype=[(name, "S16") for name in ("River", "Reach", "Station")]))
        stage = hdf.create_dataset(f"{BASE}/Cross Sections/Water Surface", data=[[2., 2.], [4., 4.]])
        stage.attrs["Variable Units"] = "Feet"
        hdf.create_dataset(f"{BASE}/Cross Sections Control Volume/XS CV Attributes", data=np.array([
            (b"R", b"A", b"2", b"1")],
            dtype=[(name, "S16") for name in ("River", "Reach", "Station US", "Station DS")]))
    return path


def test_analytic_storage_and_identity_alignment_without_accounting(storage_hdf):
    before = hashlib.sha256(storage_hdf.read_bytes()).hexdigest()
    result = HdfResultsXsec.estimate_storage_from_geometry(storage_hdf)
    assert result["mean_overbank_length_storage_af"] == pytest.approx([200 / 43560, 800 / 43560])
    assert result["separate_overbank_lengths_storage_af"] == result["mean_overbank_length_storage_af"]
    assert result["vertical_end_wall_assumption"] is False
    assert result["sections_exceeding_surveyed_ends"] == []
    assert hashlib.sha256(storage_hdf.read_bytes()).hexdigest() == before


def test_end_wall_assumption_is_explicit_and_reported(storage_hdf):
    with h5py.File(storage_hdf, "a") as hdf:
        hdf[f"{BASE}/Cross Sections/Water Surface"][-1] = [12, 12]
    with pytest.raises(ValueError, match="surveyed section end"):
        HdfResultsXsec.estimate_storage_from_geometry(storage_hdf)
    result = HdfResultsXsec.estimate_storage_from_geometry(storage_hdf, allow_vertical_end_walls=True)
    assert result["mean_overbank_length_storage_af"][-1] == pytest.approx(7000 / 43560)
    assert len(result["sections_exceeding_surveyed_ends"]) == 2


def test_explicit_pairs_support_dry_outputs_without_guessing_order(storage_hdf):
    expected = HdfResultsXsec.estimate_storage_from_geometry(storage_hdf)
    with h5py.File(storage_hdf, "a") as hdf:
        del hdf[f"{BASE}/Cross Sections Control Volume"]
    with pytest.raises(ValueError, match="Missing control-volume index"):
        HdfResultsXsec.estimate_storage_from_geometry(storage_hdf)
    actual = HdfResultsXsec.estimate_storage_from_geometry(
        storage_hdf, control_volumes=expected["control_volumes"])
    assert actual["mean_overbank_length_storage_af"] == expected["mean_overbank_length_storage_af"]
    assert actual["control_volume_index_source"] == "explicit"
    with pytest.raises(ValueError, match="Unresolved control-volume"):
        HdfResultsXsec.estimate_storage_from_geometry(storage_hdf, control_volumes=[("R", "A", "2", "3")])
    with pytest.raises(ValueError, match="Duplicate control-volume"):
        HdfResultsXsec.estimate_storage_from_geometry(storage_hdf, control_volumes=expected["control_volumes"] * 2)


def test_explicit_pairs_cannot_override_native_index(storage_hdf):
    with pytest.raises(ValueError, match="differ from native"):
        HdfResultsXsec.estimate_storage_from_geometry(storage_hdf, control_volumes=[("R", "A", "1", "2")])
    with pytest.raises(ValueError, match="four nonempty"):
        HdfResultsXsec.estimate_storage_from_geometry(storage_hdf, control_volumes=[("R", "A", "1")])


def test_overbank_length_sensitivity_is_separate(storage_hdf):
    with h5py.File(storage_hdf, "a") as hdf:
        hdf[f"{GEOM}/Station Elevation Values"][:] = [[0, 0], [5, 0], [10, 0]] * 2
        dataset = hdf[f"{GEOM}/Attributes"]
        for index in (0, 1):
            row = dataset[index]
            row["Left Bank"], row["Right Bank"] = 2, 6
            row["Len Left"], row["Len Channel"], row["Len Right"] = 100, 200, 300
            dataset[index] = row
    result = HdfResultsXsec.estimate_storage_from_geometry(storage_hdf, allow_vertical_end_walls=True)
    assert result["mean_overbank_length_storage_af"] == pytest.approx([4000 / 43560, 8000 / 43560])
    assert result["separate_overbank_lengths_storage_af"] == pytest.approx([4400 / 43560, 8800 / 43560])


@pytest.mark.parametrize("defect", ["units", "nan", "sentinel", "identity", "bank", "length", "time"])
def test_rejects_invalid_geometry_or_results(storage_hdf, defect):
    with h5py.File(storage_hdf, "a") as hdf:
        if defect == "units":
            hdf[f"{BASE}/Cross Sections/Water Surface"].attrs["Variable Units"] = "Meters"
        elif defect == "nan":
            hdf[f"{BASE}/Cross Sections/Water Surface"][0, 0] = np.nan
        elif defect == "sentinel":
            hdf[f"{BASE}/Cross Sections/Water Surface"][0, 0] = -9999
        elif defect == "identity":
            dataset = hdf[f"{BASE}/Cross Sections/Cross Section Attributes"]
            dataset[1] = dataset[0]
        elif defect in ("bank", "length"):
            dataset = hdf[f"{GEOM}/Attributes"]
            row = dataset[0]
            row["Right Bank" if defect == "bank" else "Len Channel"] = -1
            dataset[0] = row
        else:
            dataset = hdf[f"{BASE}/Time Date Stamp (ms)"]
            dataset[1] = dataset[0]
    with pytest.raises(ValueError):
        HdfResultsXsec.estimate_storage_from_geometry(storage_hdf)
