"""Manning's n updates must preserve full-width native HDF class identities."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from ras_commander import HdfLandCover


@pytest.mark.parametrize("legacy", [False, True])
def test_roughness_update_preserves_full_width_nullterm_name(tmp_path: Path, legacy: bool):
    path = tmp_path / "landcover.hdf"
    names = [b"Water", b"Emergent Herbaceous Wetlands"]
    width = len(names[-1])
    variables = np.array([(name, 0.12, 7.0) for name in names], dtype=[
        ("Name", f"S{width}"), ("ManningsN", "<f8"), ("Other", "<f8")])
    fields = [("ID", "<i4"), ("Name", f"S{width}")]
    if legacy:
        fields.append(("ManningsN", "<f8"))
    raster = np.array([(i, name, 0.12) if legacy else (i, name)
                       for i, name in enumerate(names)], dtype=fields)
    with h5py.File(path, "w") as hdf:
        for key, values in (("Variables", variables), ("Raster Map", raster)):
            native_type = h5py.h5t.py_create(values.dtype)
            index = values.dtype.names.index("Name")
            name_type = native_type.get_member_type(index)
            name_type.set_strpad(h5py.h5t.STR_NULLTERM)
            # Rebuild the compound type with native HEC string padding.
            compound = h5py.h5t.create(h5py.h5t.COMPOUND, values.dtype.itemsize)
            for field_index, field in enumerate(values.dtype.names):
                member = name_type if field == "Name" else native_type.get_member_type(field_index)
                compound.insert(field.encode(), values.dtype.fields[field][1], member)
            dataset = h5py.h5d.create(hdf.id, key.encode(), compound, h5py.h5s.create_simple(values.shape))
            dataset.write(h5py.h5s.ALL, h5py.h5s.ALL, values, mtype=compound)
    before = HdfLandCover.get_landcover_raster_map(path)
    assert before.mannings_n.notna().all()
    HdfLandCover.set_landcover_raster_map(path, {names[-1].decode(): 0.06})
    after = HdfLandCover.get_landcover_raster_map(path)
    assert after.class_name.tolist() == before.class_name.tolist()
    assert after.mannings_n.tolist() == pytest.approx([0.12, 0.06])
    with h5py.File(path, "r") as hdf:
        assert hdf["Variables"]["Name"].tolist() == names
        assert hdf["Variables"]["Other"].tolist() == [7.0, 7.0]
        if legacy:
            assert hdf["Raster Map"]["Name"].tolist() == names
            assert hdf["Raster Map"]["ManningsN"].tolist() == pytest.approx([0.12, 0.06])
