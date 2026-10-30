"""Explicit product CRS policy on synthetic HDF/RASMapper fixtures; no solver."""

import json

import h5py
import numpy as np
import pytest
import rasterio
from pyproj import CRS

from ras_commander import HdfBase, HdfResultsProducts
from test_hdf_results_products_export import _add_coupled_geometry, _sha256, _write_export_hdf


def _project(folder, *, projection=3451, terrain=6477):
    folder.mkdir()
    source = folder / "Model.p01.hdf"
    _write_export_hdf(source)
    (folder / "project.prj").write_text(CRS.from_epsg(projection).to_wkt())
    with h5py.File(folder / "terrain.hdf", "w") as hdf:
        hdf.attrs["Projection"] = CRS.from_epsg(terrain).to_wkt()
    (folder / "Model.rasmap").write_text(
        '<RASMapper><RASProjectionFilename Filename="project.prj" />'
        '<Terrains><Layer Filename="terrain.hdf" /></Terrains></RASMapper>'
    )
    return source


def test_strict_preflight_reports_conflict_without_outputs(tmp_path):
    source = _project(tmp_path / "source")
    report = HdfResultsProducts.inspect_export_crs(str(source))
    assert report["policy"] == "strict"
    assert report["project_resolution"] == "conflict"
    assert report["export_allowed"] is False
    assert report["crs"] is None
    assert HdfBase.get_projection(source) is None
    output = tmp_path / "new-parent" / "products"
    with pytest.raises(ValueError, match="CRS"):
        HdfResultsProducts.export(source, output, max_dimension=64)
    assert not output.parent.exists()


def test_result_geometry_export_preserves_values_and_default_guard(tmp_path):
    source = _project(tmp_path / "conflict")
    consistent = _project(tmp_path / "consistent", terrain=3451)
    original = {p.name: _sha256(p) for p in source.parent.iterdir()}
    output = tmp_path / "selected-products"
    result = HdfResultsProducts.export(
        source, output, crs_policy="result_geometry", max_dimension=64,
        include_preview=False,
    )
    control = tmp_path / "strict-products"
    HdfResultsProducts.export(consistent, control, max_dimension=64, include_preview=False)
    assert result["spatial"]["crs"] == "EPSG:3451"
    evidence = result["spatial"]["crs_resolution"]
    assert evidence["policy"] == "result_geometry"
    assert evidence["project_resolution"] == "conflict"
    assert evidence["terrain_conflict_retained"] is True
    assert evidence["coordinates_transformed"] is False
    assert result["status"]["hydraulic_qaqc"] == "not_evaluated"
    assert {e["crs"] for e in evidence["evidence"]} == {"EPSG:3451", "EPSG:6477"}
    assert json.loads((output / "result-metadata.json").read_text())["spatial"]["crs_resolution"] == evidence
    for name in ("maximum-depth.tif", "maximum-wse.tif", "maximum-velocity.tif"):
        with rasterio.open(output / name) as a, rasterio.open(control / name) as b:
            assert a.crs == b.crs and a.transform == b.transform
            np.testing.assert_array_equal(a.read(), b.read())
    assert _sha256(output / "hydraulic-hydrographs.parquet") == _sha256(control / "hydraulic-hydrographs.parquet")
    assert {p.name: _sha256(p) for p in source.parent.iterdir()} == original
    assert HdfBase.get_projection(source) is None
    with pytest.raises(FileExistsError):
        HdfResultsProducts.export(source, output, crs_policy="result_geometry")


@pytest.mark.parametrize("damage", ["project_conflict", "missing_project", "invalid_terrain", "missing_map", "missing_embedded"])
def test_result_geometry_refuses_ambiguous_or_invalid_evidence(tmp_path, damage):
    source = _project(tmp_path / "source", projection=6477 if damage == "project_conflict" else 3451)
    if damage == "missing_project":
        (source.parent / "project.prj").unlink()
    elif damage == "invalid_terrain":
        with h5py.File(source.parent / "terrain.hdf", "a") as hdf:
            hdf.attrs["Projection"] = "not a CRS"
    elif damage == "missing_map":
        (source.parent / "Model.rasmap").unlink()
    elif damage == "missing_embedded":
        with h5py.File(source, "a") as hdf:
            del hdf.attrs["Projection"]
    report = HdfResultsProducts.inspect_export_crs(source, crs_policy="result_geometry")
    assert report["export_allowed"] is False
    with pytest.raises(ValueError, match="CRS"):
        HdfResultsProducts.export(source, tmp_path / "products", crs_policy="result_geometry", max_dimension=64)
    assert not (tmp_path / "products").exists()


def test_policy_cannot_supply_arbitrary_crs(tmp_path):
    source = _project(tmp_path / "source")
    with pytest.raises(ValueError, match="crs_policy"):
        HdfResultsProducts.export(source, tmp_path / "products", crs_policy="EPSG:6477")


def test_evidence_change_during_rendering_prevents_publication(tmp_path, monkeypatch):
    source = _project(tmp_path / "source")
    original = HdfResultsProducts._build_products

    def change_projection(*args, **kwargs):
        result = original(*args, **kwargs)
        (source.parent / "project.prj").write_text(CRS.from_epsg(6477).to_wkt())
        return result

    monkeypatch.setattr(HdfResultsProducts, "_build_products", change_projection)
    output = tmp_path / "products"
    with pytest.raises(RuntimeError, match="CRS evidence changed"):
        HdfResultsProducts.export(source, output, crs_policy="result_geometry", max_dimension=64, include_preview=False)
    assert not output.exists()


def test_result_geometry_coupled_domain_uses_result_coordinates(tmp_path):
    source = _project(tmp_path / "source")
    _add_coupled_geometry(source)
    original = _sha256(source)
    manifest = HdfResultsProducts.export(
        source, tmp_path / "products", crs_policy="result_geometry",
        max_dimension=64, include_preview=False,
    )
    assert manifest["spatial"]["bbox"] == [-1., -1., 6., 2.]
    assert manifest["spatial"]["raster_bbox"] == [-1., -1., 2., 2.]
    assert manifest["spatial"]["crs_resolution"]["terrain_conflict_retained"] is True
    assert _sha256(source) == original


@pytest.mark.parametrize("damage", ["geographic", "ambiguous_map", "unreadable_map"])
def test_result_geometry_refuses_nonprojected_or_ambiguous_map(tmp_path, damage):
    source = _project(tmp_path / "source")
    if damage == "geographic":
        wkt = CRS.from_epsg(4326).to_wkt()
        with h5py.File(source, "a") as hdf:
            hdf.attrs["Projection"] = wkt
        (source.parent / "project.prj").write_text(wkt)
    elif damage == "ambiguous_map":
        (source.parent / "Model.rasmap").rename(source.parent / "Other.rasmap")
        (source.parent / "Another.rasmap").write_text("<RASMapper />")
    else:
        (source.parent / "Model.rasmap").write_text("not XML")
    assert HdfResultsProducts.inspect_export_crs(source, crs_policy="result_geometry")["export_allowed"] is False


def test_post_publication_crs_change_removes_new_package(tmp_path, monkeypatch):
    source = _project(tmp_path / "source")
    publish = HdfResultsProducts._publish_package
    output = tmp_path / "products"

    def change_projection(*args, **kwargs):
        publish(*args, **kwargs)
        (source.parent / "project.prj").write_text(CRS.from_epsg(6477).to_wkt())

    monkeypatch.setattr(HdfResultsProducts, "_publish_package", change_projection)
    with pytest.raises(RuntimeError, match="CRS evidence changed"):
        HdfResultsProducts.export(source, output, crs_policy="result_geometry", max_dimension=64, include_preview=False)
    assert not output.exists()


def _add_auxiliary_xsec_tables(source):
    """Synthetic 6.6 layout observed in complete 2D-only producer results."""
    with h5py.File(source, "a") as hdf:
        hdf.attrs["File Version"] = "HEC-RAS 6.6 September 2024"
        hdf["Geometry"].attrs["Complete Geometry"] = True
        hdf["Geometry/Cross Sections/Property Tables/XSEC Info"] = np.array([[0, 2, 0], [2, 2, 0]])
        hdf["Geometry/Cross Sections/Flow Distribution/Info"] = np.array([[0, 1, 0], [1, 1, 0]])


@pytest.mark.parametrize("policy", ["strict", "result_geometry"])
def test_complete_66_2d_result_reports_auxiliary_tables_without_1d_domain(tmp_path, policy):
    source = _project(tmp_path / "source", terrain=3451 if policy == "strict" else 6477)
    _add_auxiliary_xsec_tables(source)
    before = _sha256(source)
    manifest = HdfResultsProducts.export(
        source, tmp_path / "products", crs_policy=policy, max_dimension=64, include_preview=False,
    )
    assert manifest["spatial"]["bbox"] == manifest["spatial"]["raster_bbox"]
    assert manifest["spatial"]["non_domain_geometry_groups"] == [{
        "group": "Cross Sections", "reason": "complete_66_2d_auxiliary_tables_only",
        "subgroups": ["Flow Distribution", "Property Tables"],
    }]
    assert _sha256(source) == before


@pytest.mark.parametrize("damage", ["incomplete", "unknown_version", "spatial_data", "result_identity", "unknown_subgroup"])
def test_auxiliary_table_rule_does_not_hide_incomplete_1d_geometry(tmp_path, damage):
    source = _project(tmp_path / "source", terrain=3451)
    _add_auxiliary_xsec_tables(source)
    with h5py.File(source, "a") as hdf:
        if damage == "incomplete":
            hdf["Geometry"].attrs["Complete Geometry"] = False
        elif damage == "unknown_version":
            hdf.attrs["File Version"] = "unknown producer"
        elif damage == "spatial_data":
            hdf["Geometry/Cross Sections/Polyline Points"] = np.zeros((2, 2))
        elif damage == "result_identity":
            hdf.create_group("Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Cross Sections")
        else:
            hdf.create_group("Geometry/Cross Sections/Unknown")
    with pytest.raises(ValueError, match="footprint identities"):
        HdfResultsProducts.export(source, tmp_path / "products", max_dimension=64, include_preview=False)
    assert not (tmp_path / "products").exists()
