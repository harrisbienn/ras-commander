"""Tests for authenticated RAS precipitation application areas."""

from __future__ import annotations

import json
from copy import deepcopy
from importlib.resources import files

import geopandas as gpd
import pytest
from shapely.geometry import box

from ras_commander.precip import PrecipitationApplicationArea


def _grid() -> dict[str, object]:
    return {
        "definition_id": "asymmetric-target-grid",
        "crs": "EPSG:3857",
        "shape": [2, 3],
        "cell_size_meters": 10.0,
        "cell_area_square_meters": 100.0,
        "origin": [0.0, 0.0],
        "row_order": "south_to_north",
    }


def _mesh_cells() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "mesh_name": ["Receiving Area", "Receiving Area", "Other Area"],
            "cell_id": [9, 2, 1],
        },
        geometry=[
            box(0.0, 0.0, 25.0, 10.0),
            box(10.0, 10.0, 20.0, 20.0),
            box(100.0, 100.0, 110.0, 110.0),
        ],
        crs="EPSG:3857",
    )


def _compile(*, plan_id: str = "p01") -> dict[str, object]:
    return PrecipitationApplicationArea.compile_from_mesh_cells(
        _mesh_cells(),
        "Receiving Area",
        _grid(),
        project_id="fixture-project",
        plan_id=plan_id,
        geometry_id="g01",
        source_geometry_hdf={
            "name": "fixture.g01.hdf",
            "size_bytes": 123,
            "sha256": "a" * 64,
        },
    )


def test_compile_preserves_grid_order_and_effective_area() -> None:
    artifact = _compile()

    assert artifact["schema"] == ("ras-commander/precipitation-application-area/1.0")
    assert artifact["method"] == "ras-mesh-effective-area"
    assert artifact["algorithm"] == ("target-cell-intersection-with-mesh-union-v1")
    assert [cell["cell_id"] for cell in artifact["cells"]] == list(range(6))
    assert [cell["row_index"] for cell in artifact["cells"]] == [0, 0, 0, 1, 1, 1]
    assert [cell["column_index"] for cell in artifact["cells"]] == [0, 1, 2, 0, 1, 2]
    assert [cell["effective_area_square_meters"] for cell in artifact["cells"]] == [
        100.0,
        100.0,
        50.0,
        0.0,
        100.0,
        0.0,
    ]
    assert [cell["membership"] for cell in artifact["cells"]] == [
        "inside",
        "inside",
        "partial",
        "outside",
        "inside",
        "outside",
    ]
    assert artifact["metrics"] == {
        "mesh_cell_count": 2,
        "mesh_union_area_square_meters": 350.0,
        "target_cell_count": 6,
        "inside_target_cell_count": 3,
        "partial_target_cell_count": 1,
        "outside_target_cell_count": 2,
        "receiving_area_square_meters": 350.0,
        "target_grid_area_square_meters": 600.0,
    }
    assert len(artifact["application_area_sha256"]) == 64


def test_compile_is_deterministic_and_write_is_idempotent(tmp_path) -> None:
    first = _compile()
    second = _compile()
    assert first == second

    destination = tmp_path / "application-area.json"
    assert PrecipitationApplicationArea.write(first, destination) == destination
    assert PrecipitationApplicationArea.write(second, destination) == destination
    assert json.loads(destination.read_text(encoding="utf-8")) == first

    with pytest.raises(FileExistsError, match="different artifact"):
        PrecipitationApplicationArea.write(
            _compile(plan_id="p02"),
            destination,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["cells"].reverse(), "row-major"),
        (
            lambda value: value["cells"][2].update(effective_area_square_meters=100.0),
            "partial cell",
        ),
        (
            lambda value: value["metrics"].update(receiving_area_square_meters=999.0),
            "metrics do not agree",
        ),
    ],
)
def test_validate_rejects_internal_drift(mutation, message) -> None:
    artifact = deepcopy(_compile())
    mutation(artifact)
    with pytest.raises(ValueError, match=message):
        PrecipitationApplicationArea.validate(artifact)


def test_compile_rejects_grid_orientation_and_supports_reprojection() -> None:
    north_to_south = _grid()
    north_to_south["row_order"] = "north_to_south"
    with pytest.raises(ValueError, match="south_to_north"):
        PrecipitationApplicationArea.compile_from_mesh_cells(
            _mesh_cells(),
            "Receiving Area",
            north_to_south,
            project_id="fixture-project",
            plan_id="p01",
            geometry_id="g01",
            source_geometry_hdf={
                "name": "fixture.g01.hdf",
                "size_bytes": 123,
                "sha256": "a" * 64,
            },
        )

    reprojected_grid = _grid()
    reprojected_grid["crs"] = "EPSG:3395"
    reprojected = PrecipitationApplicationArea.compile_from_mesh_cells(
        _mesh_cells(),
        "Receiving Area",
        reprojected_grid,
        project_id="fixture-project",
        plan_id="p01",
        geometry_id="g01",
        source_geometry_hdf={
            "name": "fixture.g01.hdf",
            "size_bytes": 123,
            "sha256": "a" * 64,
        },
    )
    assert reprojected["source_mesh_crs"] == "EPSG:3857"
    assert reprojected["target_grid"]["crs"] == "EPSG:3395"


def test_compile_rejects_overlapping_mesh_cells() -> None:
    cells = gpd.GeoDataFrame(
        {"mesh_name": ["Area", "Area"], "cell_id": [0, 1]},
        geometry=[box(0.0, 0.0, 20.0, 20.0), box(10.0, 0.0, 30.0, 20.0)],
        crs="EPSG:3857",
    )
    with pytest.raises(ValueError, match="overlap"):
        PrecipitationApplicationArea.compile_from_mesh_cells(
            cells,
            "Area",
            _grid(),
            project_id="fixture-project",
            plan_id="p01",
            geometry_id="g01",
            source_geometry_hdf={
                "name": "fixture.g01.hdf",
                "size_bytes": 123,
                "sha256": "a" * 64,
            },
        )


def test_packaged_schema_matches_public_contract() -> None:
    schema_path = files("ras_commander.contracts").joinpath(
        "precipitation-application-area-v1.0.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["$id"] == PrecipitationApplicationArea.SCHEMA
    assert schema["properties"]["method"]["const"] == (
        PrecipitationApplicationArea.METHOD
    )
