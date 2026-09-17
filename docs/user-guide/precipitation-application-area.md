# Precipitation Application Areas

`PrecipitationApplicationArea` builds a content-addressed description of how
an ordered regular precipitation grid overlaps one HEC-RAS 2D flow-area mesh.
It is intended for workflows that must distribute precipitation or excess
precipitation onto RAS while preserving an auditable receiving area.

The artifact is geometry evidence, not a rainfall transformation or an
engineering approval. It records every target cell in canonical
south-to-north, west-to-east row-major order and reports the positive-area
intersection between that cell and the selected mesh. A downstream package
can use those effective areas without reopening or reinterpreting the RAS HDF.

## Build From A Geometry HDF

```python
from ras_commander.precip import PrecipitationApplicationArea

target_grid = {
    "definition_id": "basin-500m-precip-grid-v1",
    "crs": "EPSG:5070",
    "shape": [44, 26],
    "cell_size_meters": 500.0,
    "cell_area_square_meters": 250000.0,
    "origin": [352500.0, 1061500.0],
    "row_order": "south_to_north",
}

artifact = PrecipitationApplicationArea.from_geometry_hdf(
    "Basin.g01.hdf",
    "Basin 2D Area",
    target_grid,
    project_id="basin-project",
    plan_id="p01",
    geometry_id="g01",
)

PrecipitationApplicationArea.write(
    artifact,
    "evidence/basin-precipitation-application-area.json",
)
```

The mesh may use a different projected CRS, including a State Plane feet
definition. RAS Commander authenticates the original mesh cells, reprojects
them to the target grid's projected meter CRS, and calculates all reported
areas in square meters.

## Contract And Failure Behavior

The `ras-commander/precipitation-application-area/1.0` artifact binds:

- portable project, plan, geometry, and 2D-flow-area identifiers;
- source geometry-HDF size and SHA-256;
- source mesh CRS and an ordered mesh-cell geometry digest;
- the complete target grid definition and its digest;
- target cell IDs, row/column indices, centers, bounds, membership, and
  effective receiving areas;
- inside, partial, and outside cell counts plus aggregate areas; and
- Shapely/GEOS and intersection-algorithm provenance.

The builder fails on missing or invalid polygons, duplicate mesh-cell IDs,
positive-area mesh overlap, a nonprojected target CRS, non-meter target units,
an invalid grid checksum, or a target grid with no positive-area mesh overlap.
`write()` is idempotent for identical content and refuses to replace a
different artifact.

Any geometry, grid, plan binding, algorithm, or software-provenance change
creates a different artifact identity. Hydraulic or hydrologic suitability
must be assessed separately under the consuming workflow's qualification and
engineering-approval policy.
