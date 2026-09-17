"""Authenticated RAS receiving-area artifacts for precipitation grids."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

from ..Decorators import log_call, standardize_input
from ..hdf.HdfMesh import HdfMesh
from ..LoggingConfig import get_logger

logger = get_logger(__name__)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    content = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_mapping(
    value: Mapping[str, Any],
    required: set[str],
    *,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    missing = sorted(required - set(value))
    unexpected = sorted(set(value) - required)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(unexpected)}")
        raise ValueError(f"{label} has invalid fields: {'; '.join(details)}")


def _non_empty(value: Any, *, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    return normalized


def _normalized_grid(configured: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from pyproj import CRS
    except ImportError as exc:  # pragma: no cover - required package dependency
        raise ImportError("Grid validation requires pyproj") from exc

    required = {
        "definition_id",
        "crs",
        "shape",
        "cell_size_meters",
        "origin",
        "row_order",
    }
    missing = sorted(required - set(configured))
    if missing:
        raise ValueError("target_grid_definition is missing: " + ", ".join(missing))

    try:
        shape = tuple(int(value) for value in configured["shape"])
        origin = tuple(float(value) for value in configured["origin"])
        cell_size = float(configured["cell_size_meters"])
    except (TypeError, ValueError) as exc:
        raise ValueError("target_grid_definition has invalid numeric values") from exc
    if len(shape) != 2 or any(value <= 0 for value in shape):
        raise ValueError("target_grid_definition shape must have two positive integers")
    if len(origin) != 2 or not all(math.isfinite(value) for value in origin):
        raise ValueError("target_grid_definition origin must have two finite values")
    if not math.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("target_grid_definition cell_size_meters must be positive")
    if str(configured["row_order"]) != "south_to_north":
        raise ValueError("target_grid_definition row_order must be 'south_to_north'")

    crs = CRS.from_user_input(_non_empty(configured["crs"], label="target CRS"))
    if not crs.is_projected:
        raise ValueError("target_grid_definition CRS must be projected")
    axis_factors = {
        float(axis.unit_conversion_factor)
        for axis in crs.axis_info
        if axis.unit_conversion_factor is not None
    }
    if axis_factors and any(
        not math.isclose(factor, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
        for factor in axis_factors
    ):
        raise ValueError("target_grid_definition CRS axes must use meters")

    expected_area = cell_size * cell_size
    area = float(configured.get("cell_area_square_meters", expected_area))
    if not math.isclose(area, expected_area, rel_tol=1.0e-12, abs_tol=1.0e-9):
        raise ValueError(
            "target_grid_definition cell area must equal cell size squared"
        )

    normalized = {
        "definition_id": _non_empty(
            configured["definition_id"],
            label="target grid definition_id",
        ),
        "crs": str(configured["crs"]),
        "shape": list(shape),
        "cell_size_meters": cell_size,
        "cell_area_square_meters": area,
        "origin": list(origin),
        "row_order": "south_to_north",
    }
    normalized["definition_sha256"] = _canonical_sha256(normalized)
    supplied_hash = configured.get("definition_sha256")
    if supplied_hash is not None and supplied_hash != normalized["definition_sha256"]:
        raise ValueError("target_grid_definition checksum is invalid")
    return normalized


def _normalized_source_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    _required_mapping(
        source,
        {"name", "size_bytes", "sha256"},
        label="source_geometry_hdf",
    )
    name = _non_empty(source["name"], label="source_geometry_hdf.name")
    if Path(name).name != name:
        raise ValueError("source_geometry_hdf.name must be a portable filename")
    size_bytes = int(source["size_bytes"])
    sha256 = str(source["sha256"])
    if size_bytes <= 0:
        raise ValueError("source_geometry_hdf.size_bytes must be positive")
    if not _SHA256_PATTERN.fullmatch(sha256):
        raise ValueError("source_geometry_hdf.sha256 must be lowercase SHA-256")
    return {"name": name, "size_bytes": size_bytes, "sha256": sha256}


class PrecipitationApplicationArea:
    """Build deterministic RAS precipitation receiving-area evidence.

    The artifact intersects an ordered regular precipitation grid with one
    exact HEC-RAS 2D flow-area mesh. It reports effective receiving area for
    every target cell while preserving the target grid's south-to-north,
    west-to-east row-major order. It does not apply HMS subbasin policy or
    approve the resulting spatial transfer.
    """

    SCHEMA = "ras-commander/precipitation-application-area/1.0"
    METHOD = "ras-mesh-effective-area"
    ALGORITHM = "target-cell-intersection-with-mesh-union-v1"
    AREA_PRECISION_DECIMAL_PLACES = 9

    @staticmethod
    @log_call
    @standardize_input(file_type="geom_hdf")
    def from_geometry_hdf(
        hdf_path: Union[Path, str],
        mesh_name: str,
        target_grid_definition: Mapping[str, Any],
        *,
        project_id: str,
        plan_id: str,
        geometry_id: str,
        ras_object=None,
    ) -> dict[str, Any]:
        """Build an application-area artifact from a RAS geometry HDF.

        Args:
            hdf_path: Geometry HDF path, plan number, or supported RAS input.
            mesh_name: Exact HEC-RAS 2D flow-area name.
            target_grid_definition: Regular target-grid definition with CRS,
                shape, origin, cell size, and south-to-north row ordering.
            project_id: Portable project identifier selected by the caller.
            plan_id: Portable plan identifier selected by the caller.
            geometry_id: Portable geometry identifier selected by the caller.
            ras_object: Optional initialized RasPrj context used by input
                standardization for multi-project workflows.

        Returns:
            JSON-serializable application-area artifact with deterministic
            content identity and per-cell effective receiving areas.

        Raises:
            FileNotFoundError: If the geometry HDF cannot be resolved.
            ValueError: If the mesh, grid, CRS, geometry, or identity is
                incomplete or inconsistent.
        """
        source = Path(hdf_path).resolve()
        before = source.stat()
        mesh_cells = HdfMesh.get_mesh_cell_polygons(source)
        after = source.stat()
        if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
            raise RuntimeError(
                "Geometry HDF changed while precipitation application area was built"
            )
        source_identity = {
            "name": source.name,
            "size_bytes": after.st_size,
            "sha256": _sha256_file(source),
        }
        after_hash = source.stat()
        if (
            after_hash.st_size != after.st_size
            or after_hash.st_mtime_ns != after.st_mtime_ns
        ):
            raise RuntimeError(
                "Geometry HDF changed while precipitation application area was built"
            )
        return PrecipitationApplicationArea.compile_from_mesh_cells(
            mesh_cells,
            mesh_name,
            target_grid_definition,
            project_id=project_id,
            plan_id=plan_id,
            geometry_id=geometry_id,
            source_geometry_hdf=source_identity,
        )

    @staticmethod
    @log_call
    def compile_from_mesh_cells(
        mesh_cells: Any,
        mesh_name: str,
        target_grid_definition: Mapping[str, Any],
        *,
        project_id: str,
        plan_id: str,
        geometry_id: str,
        source_geometry_hdf: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Compile an artifact from already extracted RAS mesh-cell polygons.

        This entry point keeps geometry math independently testable while
        ``from_geometry_hdf`` remains the normal HEC-RAS-facing workflow.

        Args:
            mesh_cells: GeoDataFrame with ``mesh_name``, ``cell_id``, and
                polygon ``geometry`` columns plus a projected CRS.
            mesh_name: Exact 2D flow-area name to select.
            target_grid_definition: Regular target-grid definition.
            project_id: Portable project identifier.
            plan_id: Portable plan identifier.
            geometry_id: Portable geometry identifier.
            source_geometry_hdf: Portable ``name``, ``size_bytes``, and
                ``sha256`` identity for the source geometry HDF.

        Returns:
            Validated JSON-serializable application-area artifact.

        Raises:
            ValueError: If geometry, CRS, grid, ordering, or identities are
                incomplete or inconsistent.
        """
        try:
            import shapely
            from pyproj import CRS
            from shapely.geometry import box
            from shapely.ops import unary_union
        except ImportError as exc:  # pragma: no cover - package dependencies
            raise ImportError(
                "Precipitation application areas require geopandas, pyproj, and shapely"
            ) from exc

        selected_name = _non_empty(mesh_name, label="mesh_name")
        model = {
            "project_id": _non_empty(project_id, label="project_id"),
            "plan_id": _non_empty(plan_id, label="plan_id"),
            "geometry_id": _non_empty(geometry_id, label="geometry_id"),
            "two_d_flow_area": selected_name,
        }
        source_identity = _normalized_source_identity(source_geometry_hdf)
        target_grid = _normalized_grid(target_grid_definition)

        required_columns = {"mesh_name", "cell_id", "geometry"}
        if mesh_cells is None or not required_columns.issubset(
            set(getattr(mesh_cells, "columns", []))
        ):
            raise ValueError("mesh_cells must contain mesh_name, cell_id, and geometry")
        selected = mesh_cells.loc[mesh_cells["mesh_name"] == selected_name].copy()
        if selected.empty:
            raise ValueError(f"2D flow area {selected_name!r} has no mesh cells")
        if selected.crs is None:
            raise ValueError("mesh_cells must declare a projected CRS")
        mesh_crs = CRS.from_user_input(selected.crs)
        target_crs = CRS.from_user_input(target_grid["crs"])
        if not mesh_crs.is_projected:
            raise ValueError("mesh CRS must be projected")
        if selected["cell_id"].duplicated().any():
            raise ValueError("mesh cell identifiers must be unique within the area")
        if selected.geometry.isna().any() or selected.geometry.is_empty.any():
            raise ValueError("mesh cell geometry must not be missing or empty")
        if not selected.geometry.is_valid.all():
            raise ValueError("mesh cell geometry must be valid")
        if not all(
            geometry.geom_type in {"Polygon", "MultiPolygon"}
            for geometry in selected.geometry
        ):
            raise ValueError("mesh cell geometry must contain only polygons")

        selected = selected.sort_values("cell_id", kind="stable")
        mesh_records = [
            {
                "cell_id": int(row.cell_id),
                "geometry_wkb_hex": row.geometry.wkb_hex.lower(),
            }
            for row in selected.itertuples(index=False)
        ]
        source_mesh_crs = mesh_crs.to_string()
        if not mesh_crs.equals(target_crs):
            selected = selected.to_crs(target_crs)
            if selected.geometry.isna().any() or selected.geometry.is_empty.any():
                raise ValueError("mesh reprojection produced missing or empty geometry")
            if not selected.geometry.is_valid.all():
                raise ValueError("mesh reprojection produced invalid geometry")
        individual_area = float(selected.geometry.area.sum())
        mesh_union = unary_union(selected.geometry.tolist())
        if mesh_union.is_empty or not mesh_union.is_valid:
            raise ValueError("selected mesh did not produce a valid area")
        union_area = float(mesh_union.area)
        overlap_area = individual_area - union_area
        overlap_tolerance = max(1.0e-6, union_area * 1.0e-12)
        if overlap_area > overlap_tolerance:
            raise ValueError(
                "mesh cells overlap by a positive area and cannot define one receiving support"
            )

        rows, columns = target_grid["shape"]
        origin_x, origin_y = target_grid["origin"]
        cell_size = target_grid["cell_size_meters"]
        cell_area = target_grid["cell_area_square_meters"]
        area_tolerance = max(1.0e-9, cell_area * 1.0e-12)
        cells = []
        receiving_area = 0.0
        membership_counts = {"inside": 0, "partial": 0, "outside": 0}
        for row_index in range(rows):
            minimum_y = origin_y + row_index * cell_size
            maximum_y = minimum_y + cell_size
            for column_index in range(columns):
                minimum_x = origin_x + column_index * cell_size
                maximum_x = minimum_x + cell_size
                target_cell = box(minimum_x, minimum_y, maximum_x, maximum_y)
                effective_area = float(target_cell.intersection(mesh_union).area)
                if effective_area <= area_tolerance:
                    effective_area = 0.0
                    membership = "outside"
                elif math.isclose(
                    effective_area,
                    cell_area,
                    rel_tol=1.0e-12,
                    abs_tol=area_tolerance,
                ):
                    effective_area = cell_area
                    membership = "inside"
                else:
                    membership = "partial"
                effective_area = round(
                    effective_area,
                    PrecipitationApplicationArea.AREA_PRECISION_DECIMAL_PLACES,
                )
                receiving_area += effective_area
                membership_counts[membership] += 1
                cells.append(
                    {
                        "cell_id": row_index * columns + column_index,
                        "row_index": row_index,
                        "column_index": column_index,
                        "center": [
                            minimum_x + cell_size / 2.0,
                            minimum_y + cell_size / 2.0,
                        ],
                        "bounds": [
                            minimum_x,
                            minimum_y,
                            maximum_x,
                            maximum_y,
                        ],
                        "membership": membership,
                        "effective_area_square_meters": effective_area,
                    }
                )
        if receiving_area <= 0:
            raise ValueError("target grid has no positive-area overlap with the mesh")

        artifact: dict[str, Any] = {
            "schema": PrecipitationApplicationArea.SCHEMA,
            "method": PrecipitationApplicationArea.METHOD,
            "algorithm": PrecipitationApplicationArea.ALGORITHM,
            "model": model,
            "source_geometry_hdf": source_identity,
            "source_mesh_crs": source_mesh_crs,
            "source_mesh_cells_sha256": _canonical_sha256(
                {
                    "mesh_name": selected_name,
                    "crs": source_mesh_crs,
                    "cells": mesh_records,
                }
            ),
            "target_grid": target_grid,
            "spatial_reference": {
                "horizontal_crs": target_grid["crs"],
                "horizontal_datum": str(target_crs.datum.name),
                "horizontal_units": "meters",
                "vertical_datum": "not_applicable",
            },
            "boundary_predicate": "positive-area-intersection",
            "area_precision_decimal_places": (
                PrecipitationApplicationArea.AREA_PRECISION_DECIMAL_PLACES
            ),
            "compiler": {
                "shapely_version": str(shapely.__version__),
                "geos_version": str(shapely.geos_version_string),
            },
            "cells": cells,
            "metrics": {
                "mesh_cell_count": len(selected),
                "mesh_union_area_square_meters": round(
                    union_area,
                    PrecipitationApplicationArea.AREA_PRECISION_DECIMAL_PLACES,
                ),
                "target_cell_count": len(cells),
                "inside_target_cell_count": membership_counts["inside"],
                "partial_target_cell_count": membership_counts["partial"],
                "outside_target_cell_count": membership_counts["outside"],
                "receiving_area_square_meters": round(
                    receiving_area,
                    PrecipitationApplicationArea.AREA_PRECISION_DECIMAL_PLACES,
                ),
                "target_grid_area_square_meters": round(
                    len(cells) * cell_area,
                    PrecipitationApplicationArea.AREA_PRECISION_DECIMAL_PLACES,
                ),
            },
        }
        artifact["application_area_sha256"] = _canonical_sha256(artifact)
        return PrecipitationApplicationArea.validate(artifact)

    @staticmethod
    @log_call
    def validate(artifact: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and return a normalized application-area artifact.

        Args:
            artifact: Candidate application-area mapping.

        Returns:
            A JSON-normalized copy whose checksum, grid order, areas, counts,
            and method identity have been verified.

        Raises:
            ValueError: If the artifact is malformed, stale, or inconsistent.
        """
        normalized = json.loads(json.dumps(artifact, sort_keys=True, allow_nan=False))
        _required_mapping(
            normalized,
            {
                "schema",
                "method",
                "algorithm",
                "model",
                "source_geometry_hdf",
                "source_mesh_crs",
                "source_mesh_cells_sha256",
                "target_grid",
                "spatial_reference",
                "boundary_predicate",
                "area_precision_decimal_places",
                "compiler",
                "cells",
                "metrics",
                "application_area_sha256",
            },
            label="precipitation application area",
        )
        if normalized["schema"] != PrecipitationApplicationArea.SCHEMA:
            raise ValueError("precipitation application-area schema is unsupported")
        if normalized["method"] != PrecipitationApplicationArea.METHOD:
            raise ValueError("precipitation application-area method is unsupported")
        if normalized["algorithm"] != PrecipitationApplicationArea.ALGORITHM:
            raise ValueError("precipitation application-area algorithm is unsupported")
        if normalized["boundary_predicate"] != "positive-area-intersection":
            raise ValueError("precipitation application-area predicate is unsupported")
        if normalized["area_precision_decimal_places"] != (
            PrecipitationApplicationArea.AREA_PRECISION_DECIMAL_PLACES
        ):
            raise ValueError("precipitation application-area precision is unsupported")

        _required_mapping(
            normalized["model"],
            {"project_id", "plan_id", "geometry_id", "two_d_flow_area"},
            label="model",
        )
        for key, value in normalized["model"].items():
            _non_empty(value, label=f"model.{key}")
        normalized["source_geometry_hdf"] = _normalized_source_identity(
            normalized["source_geometry_hdf"]
        )
        source_mesh_crs = _non_empty(
            normalized["source_mesh_crs"],
            label="source_mesh_crs",
        )
        try:
            from pyproj import CRS

            if not CRS.from_user_input(source_mesh_crs).is_projected:
                raise ValueError("source_mesh_crs must be projected")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("source_mesh_crs is invalid") from exc
        if not _SHA256_PATTERN.fullmatch(normalized["source_mesh_cells_sha256"]):
            raise ValueError("source_mesh_cells_sha256 must be lowercase SHA-256")
        normalized_grid = _normalized_grid(normalized["target_grid"])
        if normalized_grid != normalized["target_grid"]:
            raise ValueError("target_grid is not in canonical form")

        _required_mapping(
            normalized["spatial_reference"],
            {
                "horizontal_crs",
                "horizontal_datum",
                "horizontal_units",
                "vertical_datum",
            },
            label="spatial_reference",
        )
        if (
            normalized["spatial_reference"]["horizontal_crs"] != normalized_grid["crs"]
            or normalized["spatial_reference"]["horizontal_units"] != "meters"
            or normalized["spatial_reference"]["vertical_datum"] != "not_applicable"
        ):
            raise ValueError("spatial_reference does not match the target grid")

        rows, columns = normalized_grid["shape"]
        cell_size = normalized_grid["cell_size_meters"]
        cell_area = normalized_grid["cell_area_square_meters"]
        origin_x, origin_y = normalized_grid["origin"]
        cells = normalized["cells"]
        if not isinstance(cells, list) or len(cells) != rows * columns:
            raise ValueError("cells do not cover the complete target grid")
        membership_counts = {"inside": 0, "partial": 0, "outside": 0}
        receiving_area = 0.0
        for expected_id, cell in enumerate(cells):
            _required_mapping(
                cell,
                {
                    "cell_id",
                    "row_index",
                    "column_index",
                    "center",
                    "bounds",
                    "membership",
                    "effective_area_square_meters",
                },
                label=f"cells[{expected_id}]",
            )
            expected_row, expected_column = divmod(expected_id, columns)
            if (
                cell["cell_id"] != expected_id
                or cell["row_index"] != expected_row
                or cell["column_index"] != expected_column
            ):
                raise ValueError("cells are not in canonical row-major order")
            minimum_x = origin_x + expected_column * cell_size
            minimum_y = origin_y + expected_row * cell_size
            expected_bounds = [
                minimum_x,
                minimum_y,
                minimum_x + cell_size,
                minimum_y + cell_size,
            ]
            expected_center = [
                minimum_x + cell_size / 2.0,
                minimum_y + cell_size / 2.0,
            ]
            if cell["bounds"] != expected_bounds or cell["center"] != expected_center:
                raise ValueError("cell coordinates do not match the target grid")
            membership = cell["membership"]
            if membership not in membership_counts:
                raise ValueError("cell membership is unsupported")
            area = float(cell["effective_area_square_meters"])
            if not math.isfinite(area) or area < 0 or area > cell_area:
                raise ValueError("cell effective area is outside valid bounds")
            if membership == "outside" and area != 0.0:
                raise ValueError("outside cell must have zero effective area")
            if membership == "inside" and area != cell_area:
                raise ValueError("inside cell must have its complete area")
            if membership == "partial" and not 0.0 < area < cell_area:
                raise ValueError("partial cell must have a partial effective area")
            membership_counts[membership] += 1
            receiving_area += area

        metrics = normalized["metrics"]
        _required_mapping(
            metrics,
            {
                "mesh_cell_count",
                "mesh_union_area_square_meters",
                "target_cell_count",
                "inside_target_cell_count",
                "partial_target_cell_count",
                "outside_target_cell_count",
                "receiving_area_square_meters",
                "target_grid_area_square_meters",
            },
            label="metrics",
        )
        expected_metrics = {
            "target_cell_count": len(cells),
            "inside_target_cell_count": membership_counts["inside"],
            "partial_target_cell_count": membership_counts["partial"],
            "outside_target_cell_count": membership_counts["outside"],
            "receiving_area_square_meters": round(
                receiving_area,
                PrecipitationApplicationArea.AREA_PRECISION_DECIMAL_PLACES,
            ),
            "target_grid_area_square_meters": round(
                len(cells) * cell_area,
                PrecipitationApplicationArea.AREA_PRECISION_DECIMAL_PLACES,
            ),
        }
        if any(metrics[key] != value for key, value in expected_metrics.items()):
            raise ValueError("metrics do not agree with the ordered target cells")
        if int(metrics["mesh_cell_count"]) <= 0:
            raise ValueError("metrics.mesh_cell_count must be positive")
        if float(metrics["mesh_union_area_square_meters"]) <= 0:
            raise ValueError("metrics.mesh_union_area_square_meters must be positive")
        if receiving_area <= 0:
            raise ValueError("metrics.receiving_area_square_meters must be positive")

        _required_mapping(
            normalized["compiler"],
            {"shapely_version", "geos_version"},
            label="compiler",
        )
        _non_empty(
            normalized["compiler"]["shapely_version"],
            label="compiler.shapely_version",
        )
        _non_empty(
            normalized["compiler"]["geos_version"],
            label="compiler.geos_version",
        )
        recorded_hash = normalized.pop("application_area_sha256")
        if recorded_hash != _canonical_sha256(normalized):
            raise ValueError("application_area_sha256 is missing or invalid")
        normalized["application_area_sha256"] = recorded_hash
        return normalized

    @staticmethod
    @log_call
    def write(
        artifact: Mapping[str, Any],
        output_path: Union[Path, str],
    ) -> Path:
        """Write an application-area artifact idempotently.

        Args:
            artifact: Valid application-area artifact.
            output_path: Destination JSON path.

        Returns:
            Resolved destination path.

        Raises:
            FileExistsError: If the destination contains different content.
            ValueError: If the artifact is invalid.
        """
        normalized = PrecipitationApplicationArea.validate(artifact)
        destination = Path(output_path).resolve()
        content = json.dumps(normalized, indent=2, sort_keys=True) + "\n"
        if destination.exists():
            if destination.read_text(encoding="utf-8") == content:
                return destination
            raise FileExistsError(
                f"Refusing to replace different artifact: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.stem}-",
            dir=destination.parent,
        ) as stage_name:
            staged = Path(stage_name) / destination.name
            staged.write_text(content, encoding="utf-8")
            staged.replace(destination)
        logger.info("Wrote precipitation application area %s", destination.name)
        logger.debug("Precipitation application area path: %s", destination)
        return destination
