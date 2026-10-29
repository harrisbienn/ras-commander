"""Fork/upstream compatibility contracts; no HEC engine is invoked."""
import hashlib
import json
import subprocess
from types import SimpleNamespace
from importlib import import_module

import h5py
import numpy as np
import pytest
import pyarrow.parquet as pq

from ras_commander import HdfResultsProducts, RasUnsteady
from ras_commander.RasBoundary import delete_boundaries, BoundaryStaleEvidenceError
from ras_commander.RasProject import (
    stage_project, stage_project_for_boundary_edit, ProjectPopulationError,
)
from test_ras_scenario import _write_project
from test_hdf_results_products_export import _write_export_hdf, SERIES_BASE


def _selections(stage, indices):
    inventory = RasUnsteady.inspect_boundary_blocks(stage, unsteady_number="01")
    return tuple({
        "boundary_id": str(row.boundary_id),
        "expected_source_sha256": str(row.owner_sha256),
        "expected_block_sha256": str(row.block_sha256),
        "expected_bc_type": str(row.bc_type),
        "expected_location_raw": str(row.boundary_location_raw),
    } for row in inventory.itertuples() if row.boundary_index in indices)


def test_editor_stage_keeps_execution_unready_and_applies_batch(tmp_path):
    project = _write_project(tmp_path / "source")
    source_bytes = {p.name: p.read_bytes() for p in project.parent.iterdir()}
    with pytest.raises(ProjectPopulationError):
        stage_project(project, tmp_path / "execution")
    stage = stage_project_for_boundary_edit(project, tmp_path / "editor")
    assert stage.execution_readiness == "not_ready"
    manifest = json.loads((stage.destination_root / ".ras-commander/stage.json").read_text())
    assert manifest["purpose"] == "boundary_edit"
    assert manifest["execution_readiness"] == "not_ready"
    original = (stage.destination_root / "Example.u01").read_bytes()
    selections = _selections(stage, {0, 2})
    preview = delete_boundaries(stage, unsteady_number="01", selections=selections)
    assert all(item.state == "previewed" for item in preview)
    assert preview[0].unsteady_file.read_bytes() == original
    applied = delete_boundaries(stage, unsteady_number="01", selections=selections, dry_run=False)
    assert all(item.state == "applied" and item.after_boundary_count == 1 for item in applied)
    updated = applied[0].unsteady_file.read_bytes()
    assert b"Boundary Name=Junction" in updated
    assert b"Boundary Name=Missing Storage" not in updated
    assert b"Boundary Name=Tributary" not in updated
    assert b"Met Point Raster Parameters=,,,," in updated
    assert all(item.result_sha256 == hashlib.sha256(updated).hexdigest() for item in applied)
    assert {p.name: p.read_bytes() for p in project.parent.iterdir()} == source_bytes


def test_batch_stale_evidence_prevents_all_writes(tmp_path):
    project = _write_project(tmp_path / "source")
    stage = stage_project_for_boundary_edit(project, tmp_path / "editor")
    selections = _selections(stage, {0, 2})
    selections[1]["expected_block_sha256"] = "0" * 64
    target = stage.destination_root / "Example.u01"
    original = target.read_bytes()
    with pytest.raises(BoundaryStaleEvidenceError):
        delete_boundaries(stage, unsteady_number="01", selections=selections, dry_run=False)
    assert target.read_bytes() == original


def test_editor_stage_still_requires_geometry_text(tmp_path):
    project = _write_project(tmp_path / "source")
    (project.parent / "Example.g01").unlink()
    with pytest.raises(ProjectPopulationError):
        stage_project_for_boundary_edit(project, tmp_path / "editor")
    assert not (tmp_path / "editor").exists()


def test_credentialed_psexec_failure_does_not_echo_tool_output(caplog):
    module = import_module("ras_commander.remote.PsexecWorker")
    worker = SimpleNamespace(credentials={"username": "fixture", "password": "fixture-secret"})
    result = subprocess.CompletedProcess([], 1, "fixture-secret in stdout", "fixture-secret in stderr")
    module._log_psexec_output(worker, result)
    assert "fixture-secret" not in caplog.text
    assert "omitted" in caplog.text


def test_year_3000_product_inspection_and_export(tmp_path):
    source = tmp_path / "clock.p01.hdf"
    _write_export_hdf(source)
    with h5py.File(source, "a") as hdf:
        hdf["Plan Data/Plan Information"].attrs["Simulation Start Time"] = "18Sep3000 13:00:00"
        if f"{SERIES_BASE}/Time" not in hdf:
            hdf.create_dataset(f"{SERIES_BASE}/Time", data=np.asarray([0.0, 1.0 / 24.0]))
        boundary = hdf.create_dataset(
            f"{SERIES_BASE}/Boundary Conditions/Inflow",
            data=np.asarray([[10.0, 100.0], [11.0, 120.0]]),
        )
        boundary.attrs["Columns"] = np.asarray([b"Stage", b"Flow"])
        boundary.attrs["Stage"] = b"ft"
        boundary.attrs["Flow"] = b"cfs"
        boundary.attrs["2D Area"] = b"Mesh"
        for name in ("Time Date Stamp (ms)", "Time Date Stamp"):
            path = f"{SERIES_BASE}/{name}"
            if path in hdf:
                raw = hdf[path][:]
                hdf[path][:] = np.asarray([value.replace(b"2019", b"3000") for value in raw], dtype=raw.dtype)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    inspection = HdfResultsProducts.inspect_result(source)
    assert inspection["time"]["start"] == "3000-09-18T13:00:00"
    assert inspection["time"]["interval_seconds"] == 3600
    manifest = HdfResultsProducts.export(source, tmp_path / "products", max_dimension=64)
    hydrograph = pq.read_table(tmp_path / "products/hydraulic-hydrographs.parquet")
    assert str(hydrograph.schema.field("time").type) == "timestamp[us]"
    assert len(hydrograph) == 4
    assert all(value.year == 3000 for value in hydrograph.column("time").to_pylist())
    assert set(manifest["assets"]) == set(HdfResultsProducts.FILENAMES)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
