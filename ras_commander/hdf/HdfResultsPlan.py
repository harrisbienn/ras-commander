"""
HdfResultsPlan: A module for extracting and analyzing HEC-RAS plan HDF file results.

Attribution:
    Substantial code sourced/derived from https://github.com/fema-ffrd/rashdf
    Copyright (c) 2024 fema-ffrd, MIT license

Description:
    Provides static methods for extracting both unsteady and steady flow results,
    volume accounting, and reference data from HEC-RAS plan HDF files.

Available Functions:
    Unsteady Flow:
        - get_unsteady_info: Extract unsteady attributes
        - get_unsteady_summary: Extract unsteady summary data
        - get_volume_accounting: Extract volume accounting data
        - get_volume_accounting_diagnostics: Read per-area balances and saved diagnostics
        - get_precipitation_diagnostics: Compare stored forcing, 1D depths and native accounting
        - get_precipitation_receiving_diagnostics: Integrate imported rain over stored receiving weights
        - get_precipitation_footprint_overlap: Measure stored 1D/2D footprint overlap and rain on each side
        - get_coupling_diagnostics: Read lateral segments and cumulative cross-section flow
        - get_runtime_data: Extract runtime and compute time data
        - get_reference_timeseries: Extract reference line/point timeseries
        - get_reference_summary: Extract reference line/point summary

    Steady Flow:
        - is_steady_plan: Check if HDF contains steady state results
        - get_steady_profile_names: Extract steady state profile names
        - get_steady_wse: Extract WSE data for steady state profiles
        - get_steady_info: Extract steady flow attributes and metadata

    Computation Messages:
        - get_compute_messages: Extract computation messages from HDF (with .txt fallback)

Note:
    All methods are static and designed to be used without class instantiation.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import h5py
import pandas as pd
import xarray as xr
from ..Decorators import standardize_input, log_call
from .HdfUtils import HdfUtils
from .HdfResultsXsec import HdfResultsXsec
from ..LoggingConfig import get_logger
import numpy as np
from datetime import datetime
from ..RasPrj import ras

logger = get_logger(__name__)


class HdfResultsPlan:
    """
    Handles extraction of results data from HEC-RAS plan HDF files.

    This class provides static methods for accessing and analyzing:
        - Unsteady flow results
        - Volume accounting data
        - Runtime statistics
        - Reference line/point time series outputs

    All methods use:
        - @standardize_input decorator for consistent file path handling
        - @log_call decorator for operation logging
        - HdfUtils class for common HDF operations

    Note:
        No instantiation required - all methods are static.
    """

    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def get_unsteady_info(hdf_path: Path) -> pd.DataFrame:
        """
        Get unsteady attributes from a HEC-RAS HDF plan file.

        Args:
            hdf_path (Path): Path to the HEC-RAS plan HDF file.
            ras_object (RasPrj, optional): Specific RAS object to use. If None, uses the global ras instance.

        Returns:
            pd.DataFrame: A DataFrame containing the decoded unsteady attributes.

        Raises:
            FileNotFoundError: If the specified HDF file is not found.
            KeyError: If the "Results/Unsteady" group is not found in the HDF file.
        """
        try:
            with h5py.File(hdf_path, 'r') as hdf_file:
                if "Results/Unsteady" not in hdf_file:
                    raise KeyError("Results/Unsteady group not found in the HDF file.")
                
                # Create dictionary from attributes and decode byte strings
                attrs_dict = {}
                for key, value in dict(hdf_file["Results/Unsteady"].attrs).items():
                    if isinstance(value, bytes):
                        attrs_dict[key] = value.decode('utf-8')
                    else:
                        attrs_dict[key] = value
                
                # Create DataFrame with a single row index
                return pd.DataFrame(attrs_dict, index=[0])
                
        except FileNotFoundError:
            raise FileNotFoundError(
                f"HDF file not found: {hdf_path}. "
                f"See: https://rascommander.info/ras/user-guide/hdf-data-extraction/"
            )
        except Exception as e:
            raise RuntimeError(f"Error reading unsteady attributes: {str(e)}")
        
    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def get_unsteady_summary(hdf_path: Path) -> pd.DataFrame:
        """
        Get results unsteady summary attributes from a HEC-RAS HDF plan file.

        Args:
            hdf_path (Path): Path to the HEC-RAS plan HDF file.
            ras_object (RasPrj, optional): Specific RAS object to use. If None, uses the global ras instance.

        Returns:
            pd.DataFrame: A DataFrame containing the decoded results unsteady summary attributes.

        Raises:
            FileNotFoundError: If the specified HDF file is not found.
            KeyError: If the "Results/Unsteady/Summary" group is not found in the HDF file.
        """
        try:           
            with h5py.File(hdf_path, 'r') as hdf_file:
                if "Results/Unsteady/Summary" not in hdf_file:
                    raise KeyError("Results/Unsteady/Summary group not found in the HDF file.")
                
                # Create dictionary from attributes and decode byte strings
                attrs_dict = {}
                for key, value in dict(hdf_file["Results/Unsteady/Summary"].attrs).items():
                    if isinstance(value, bytes):
                        attrs_dict[key] = value.decode('utf-8')
                    else:
                        attrs_dict[key] = value
                
                # Create DataFrame with a single row index
                return pd.DataFrame(attrs_dict, index=[0])
                
        except FileNotFoundError:
            raise FileNotFoundError(
                f"HDF file not found: {hdf_path}. "
                f"See: https://rascommander.info/ras/user-guide/hdf-data-extraction/"
            )
        except Exception as e:
            raise RuntimeError(f"Error reading unsteady summary attributes: {str(e)}")
        
    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def get_volume_accounting(hdf_path: Path) -> Optional[pd.DataFrame]:
        """
        Get volume accounting attributes from a HEC-RAS HDF plan file.

        Args:
            hdf_path (Path): Path to the HEC-RAS plan HDF file.
            ras_object (RasPrj, optional): Specific RAS object to use. If None, uses the global ras instance.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing the decoded volume accounting attributes,
                                  or None if the group is not found.

        Raises:
            FileNotFoundError: If the specified HDF file is not found.
        """
        try:
            with h5py.File(hdf_path, 'r') as hdf_file:
                if "Results/Unsteady/Summary/Volume Accounting" not in hdf_file:
                    return None
                
                # Get attributes and decode byte strings
                attrs_dict = {}
                for key, value in dict(hdf_file["Results/Unsteady/Summary/Volume Accounting"].attrs).items():
                    if isinstance(value, bytes):
                        attrs_dict[key] = value.decode('utf-8')
                    else:
                        attrs_dict[key] = value
                
                return pd.DataFrame(attrs_dict, index=[0])
                
        except FileNotFoundError:
            raise FileNotFoundError(
                f"HDF file not found: {hdf_path}. "
                f"See: https://rascommander.info/ras/user-guide/hdf-data-extraction/"
            )
        except Exception as e:
            raise RuntimeError(f"Error reading volume accounting attributes: {str(e)}")

    @staticmethod
    @log_call
    @standardize_input(file_type="plan_hdf")
    def get_volume_accounting_diagnostics(hdf_path: Path) -> Dict:
        """Read overall, 1D and per-area accounting without modifying results.

        Preserve native attributes and units alongside balance arithmetic.
        Positive residual means ending storage exceeds starting storage plus
        cumulative inflow minus outflow. Cumulative 2D inflow already includes
        precipitation; never add the separately reported precipitation again.
        Internal exchanges prevent summing area inflows into an external total.

        Args:
            hdf_path: Plan HDF path or supported plan selector.
            ras_object: Optional project context supplied by the decorator.

        Returns:
            Dictionary of raw accounting, reconstructed residuals and optional
            saved computation series. No hydraulic acceptance is assigned.
            Overall-minus-2D error is an unattributed remainder, not a measured
            1D solver error. Saved Volume Error samples need not be cumulative.

        Raises:
            KeyError: Required overall or area accounting is missing.
            ValueError: Accounting units differ or required values are invalid.
            OSError: The HDF cannot be read.
        """
        import math
        from numbers import Real
        from .HdfResultsMesh import HdfResultsMesh

        base = "Results/Unsteady/Summary/Volume Accounting"

        def number(attributes, key):
            value = attributes[key]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"Volume accounting {key!r} must be a finite number")
            return float(value)

        def residual(attributes, start, end, incoming, outgoing):
            return (
                number(attributes, end)
                - number(attributes, start)
                - number(attributes, incoming)
                + number(attributes, outgoing)
            )

        with h5py.File(hdf_path, "r") as source:
            if base not in source:
                raise KeyError(f"Missing HDF group: {base}")
            overall = HdfUtils.convert_hdf5_attrs_to_dict(source[base].attrs)
            units = overall.get("Vol Accounting in")
            if not isinstance(units, str) or not units.strip():
                raise ValueError("Overall volume accounting units are missing")
            overall_residual = residual(
                overall,
                "Volume Starting",
                "Volume Ending",
                "Total Boundary Flux of Water In",
                "Total Boundary Flux of Water Out",
            )
            reported_error = number(overall, "Error")
            one_d_path = f"{base}/Volume Accounting 1D"
            one_d = (
                HdfUtils.convert_hdf5_attrs_to_dict(source[one_d_path].attrs)
                if one_d_path in source
                else None
            )
            two_d_path = f"{base}/Volume Accounting 2D"
            areas = []
            series_available = {}
            for name in sorted(source[two_d_path]) if two_d_path in source else []:
                raw = HdfUtils.convert_hdf5_attrs_to_dict(source[f"{two_d_path}/{name}"].attrs)
                if raw.get("Vol Accounting in") != units:
                    raise ValueError(f"Volume accounting units differ for {name!r}")
                reconstructed = residual(raw, "Vol Starting", "Vol Ending", "Cum Inflow", "Cum Outflow")
                area_error = number(raw, "Error")
                areas.append(
                    {
                        "area": name,
                        "raw": raw,
                        "reconstructed_error": reconstructed,
                        "reconstruction_minus_reported": reconstructed - area_error,
                    }
                )
                series_base = (
                    "Results/Unsteady/Output/Output Blocks/Base Output/"
                    f"Unsteady Time Series/2D Flow Areas/{name}/Computations"
                )
                series_available[name] = [
                    var for var in ("Volume", "Volume Error") if f"{series_base}/{var}" in source
                ]

        saved_series = {}
        for name, variables in series_available.items():
            saved_series[name] = {}
            for variable in variables:
                data = HdfResultsMesh.get_mesh_timeseries(
                    hdf_path, name, f"Computations/{variable}", truncate=False
                )
                values = np.asarray(data.values)
                if values.ndim != 2 or values.shape[1] != 1:
                    raise ValueError(f"Expected one {variable} value per time for {name}")
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite {variable} samples for {name}")
                saved_series[name][variable] = {
                    "units": data.attrs.get("units", ""),
                    "times": [pd.Timestamp(t).isoformat() for t in data.time.values],
                    "values": values[:, 0].tolist(),
                }
        area_sum = math.fsum(number(area["raw"], "Error") for area in areas) if areas else None
        return {
            "units": units,
            "overall": overall,
            "one_d": one_d,
            "two_d": areas,
            "overall_reconstructed_error": overall_residual,
            "overall_reconstruction_minus_reported": overall_residual - reported_error,
            "sum_reported_two_d_errors": area_sum,
            "overall_minus_two_d_errors": reported_error - area_sum if area_sum is not None else None,
            "saved_computation_series": saved_series,
            "limitations": [
                "Do not add precipitation to cumulative 2D inflow a second time.",
                "Internal transfers mean per-area inflows/outflows are not external totals.",
                "Overall minus 2D errors is unattributed; 1D exchange terms require separate reconciliation.",
                "Saved Volume Error is not assumed cumulative or equivalent to final accounting Error.",
                "No engineering tolerance or hydraulic acceptance is assigned.",
            ],
        }

    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def get_precipitation_receiving_diagnostics(
        hdf_path: Path, *, value_semantics: str, source_grid_labels: Optional[List[Optional[str]]] = None,
    ) -> Dict:
        """Reconstruct prescribed receiving volumes from imported rain and weights.

        ``value_semantics`` must explicitly be ``interval_depth`` or
        ``cumulative_depth``. The caller must verify it against source interval
        totals: source ``per-cum`` metadata alone does not establish how native
        preprocessing stores Values. Cumulative mode assumes zero depth at the
        simulation start; full, regular interval-end coverage is required.

        Uses stored cell weights and physical mesh areas for 2D, and stored
        interpolation-surface weights with explicitly crosswalked ft^2 control
        volumes for 1D. Face weights are not additional rainfall receivers.
        Requires English-unit stages and metre-based source raster coordinates.
        No native rainfall summaries or cumulative output depths enter these
        calculations. This is prescribed rainfall, not proof of water admitted
        by the solver, infiltration loss, nonoverlapping footprints or acceptance.
        Raises on ambiguous identities, units, coverage or interpolation indexes.
        Optional source_grid_labels attribute volumes to caller-authenticated
        source regions in the native flattened Values order. None denotes an
        unassigned cell and must have zero rain. Labels do not alter forcing.
        """
        from pyproj import CRS
        from .HdfBase import HdfBase

        if value_semantics not in ("interval_depth", "cumulative_depth"):
            raise ValueError("Explicit imported rainfall value semantics are required")
        def decode(value):
            return value.decode().strip() if isinstance(value, bytes) else str(value).strip()

        base = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
        model_crs = CRS.from_user_input(HdfBase.get_projection(hdf_path))
        if (not model_crs.is_projected or len(model_crs.axis_info) != 2
                or any(axis.unit_name not in ("foot", "US survey foot") for axis in model_crs.axis_info)):
            raise ValueError("Receiving geometry coordinates must be feet")
        with h5py.File(hdf_path, "r") as source:
            met = source["Event Conditions/Meteorology/Precipitation"]
            metadata = HdfUtils.convert_hdf5_attrs_to_dict(met.attrs)
            if (metadata.get("Mode"), metadata.get("Data Type"), metadata.get("Units")) != (
                    "Gridded", "per-cum", "in"):
                raise ValueError("Receiving audit requires Gridded/per-cum/in forcing")
            if decode(source[f"{base}/Cross Sections/Water Surface"].attrs["Variable Units"]) != "Feet":
                raise ValueError("Receiving audit requires English geometry/stage units")
            crs = CRS.from_user_input(metadata["Projection"])
            if not crs.is_projected or any(axis.unit_name != "metre" for axis in crs.axis_info):
                raise ValueError("Source raster coordinates must be metres")
            rows, cols = int(metadata["Raster Rows"]), int(metadata["Raster Cols"])
            size = float(metadata["Raster Cellsize"])
            if rows <= 0 or cols <= 0 or not np.isfinite(size) or size <= 0:
                raise ValueError("Invalid source raster geometry")
            values = met["Values"][:].astype(float)
            timestamps = [datetime.strptime(decode(t), "%d%b%Y %H:%M:%S.%f") for t in met["Timestamp"][:]]
            saved_times = HdfBase.get_unsteady_timestamps(source)
            if (len(timestamps) < 2 or len(saved_times) < 2 or values.shape != (len(timestamps), rows * cols)
                    or not np.isfinite(values).all() or np.any(values < 0)):
                raise ValueError("Invalid imported rainfall values or dimensions")
            interval = timestamps[1] - timestamps[0]
            if (interval.total_seconds() <= 0 or timestamps[0] - interval != saved_times[0]
                    or timestamps[-1] != saved_times[-1]
                    or any(b - a != interval for a, b in zip(timestamps, timestamps[1:]))):
                raise ValueError("Imported rainfall requires full regular interval-end coverage")
            increments = (values if value_semantics == "interval_depth"
                          else np.diff(values, axis=0, prepend=np.zeros((1, rows * cols))))
            if np.any(increments < 0):
                raise ValueError("Cumulative rainfall decreases; verify stored value semantics")
            total_depth = increments.sum(axis=0)
            label_indexes = {}
            if source_grid_labels is not None:
                if len(source_grid_labels) != rows * cols or any(
                        label is not None and (not isinstance(label, str) or not label.strip())
                        for label in source_grid_labels):
                    raise ValueError("Source grid labels require one nonempty string or None per raster cell")
                if any(label is None and total_depth[i] != 0 for i, label in enumerate(source_grid_labels)):
                    raise ValueError("Unassigned source grid cells have rainfall; verify label alignment")
                label_indexes = {label: np.array([i for i, name in enumerate(source_grid_labels) if name == label])
                                 for label in sorted({v for v in source_grid_labels if v is not None})}

            def receivers(group, prefix, areas, identities):
                info = group[f"{prefix}Info"][:]
                indexes = group[f"{prefix}Indexes"][:]
                weights = group[f"{prefix}Weights"][:].astype(float)
                if (info.shape != (len(areas), 2) or not np.issubdtype(info.dtype, np.integer)
                        or indexes.ndim != 1 or not np.issubdtype(indexes.dtype, np.integer)
                        or weights.shape != indexes.shape or not np.isfinite(weights).all()
                        or np.any(weights < 0) or not np.isfinite(areas).all() or np.any(areas <= 0)):
                    raise ValueError("Invalid receiving areas or sparse weights")
                effective_areas = np.zeros(rows * cols)
                records, cursor = [], 0
                for i, (offset, count) in enumerate(info):
                    if offset != cursor or count < 0 or offset + count > len(indexes):
                        raise ValueError("Invalid or overlapping sparse weight slices")
                    cursor += count
                    selected, weight = indexes[offset:cursor], weights[offset:cursor]
                    if np.any(selected < 0) or np.any(selected >= rows * cols):
                        raise ValueError("Rainfall weight index outside source raster")
                    if count and not np.isclose(weight.sum(), 1, rtol=0, atol=1e-6):
                        raise ValueError("Receiving weights do not sum to one")
                    np.add.at(effective_areas, selected, areas[i] * weight)
                    depth = float(total_depth[selected] @ weight)
                    records.append({"identity": identities[i], "area_ft2": float(areas[i]),
                        "weight_count": int(count), "weight_sum": float(weight.sum()),
                        "depth_in": depth, "volume_af": depth * float(areas[i]) / 12 / 43560})
                if cursor != len(indexes):
                    raise ValueError("Unreferenced interpolation weights")
                series = increments @ effective_areas / 12 / 43560
                return {"receiver_count": len(areas), "area_ft2": float(areas.sum()),
                    "unmapped_receiver_count": sum(r["weight_count"] == 0 for r in records),
                    "total_volume_af": float(series.sum()), "interval_volume_af": series.tolist(),
                    "receivers": records,
                    "source_label_volume_af": {label: float(total_depth[idx] @ effective_areas[idx] / 12 / 43560)
                                               for label, idx in label_indexes.items()}}

            two_d = {}
            attrs = source["Geometry/2D Flow Areas/Attributes"][:]
            names = [decode(row["Name"]) for row in attrs]
            if len(set(names)) != len(names) or set(names) != set(met["2D Flow Areas"]):
                raise ValueError("Precipitation and geometry area identities differ")
            for row, name in zip(attrs, names):
                count = int(row["Cell Count"])
                areas = source[f"Geometry/2D Flow Areas/{name}/Cells Surface Area"][:].astype(float)
                if count <= 0 or len(areas) < count:
                    raise ValueError("Invalid physical cell count")
                two_d[name] = receivers(met[f"2D Flow Areas/{name}"], "Cell ", areas[:count], list(range(count)))
                two_d[name]["excluded_nonphysical_area_rows"] = len(areas) - count

            geom = source["Geometry/Cross Sections/Attributes"][:]
            section_ids = [tuple(decode(row[k]) for k in ("River", "Reach", "RS")) for row in geom]
            cv = source[f"{base}/Cross Sections Control Volume/XS CV Attributes"][:]
            cv_ids = [tuple(decode(row[k]) for k in ("River", "Reach", "Station US", "Station DS")) for row in cv]
            if len(set(cv_ids)) != len(cv_ids) or len(set(section_ids)) != len(section_ids):
                raise ValueError("Duplicate cross-section identities")
            by_id = dict(zip(cv_ids, cv["Surface Area (ft^2)"].astype(float)))
            pairs = source["Geometry/Cross Section Interpolation Surfaces/XSIDs"][:]
            if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
                raise ValueError("Invalid interpolation-surface section pairs")
            identities, areas = [], []
            for up, down in pairs:
                if min(up, down) < 0 or max(up, down) >= len(section_ids) or up == down:
                    raise ValueError("Invalid interpolation-surface section index")
                us, ds = section_ids[up], section_ids[down]
                identity = (*us, ds[2])
                if us[:2] != ds[:2] or identity not in by_id:
                    raise ValueError("Interpolation surface cannot resolve its control volume")
                identities.append(identity)
                areas.append(by_id[identity])
            if len(identities) != len(set(identities)) or set(identities) != set(cv_ids):
                raise ValueError("Incomplete or duplicate control-volume surface mapping")
            one_d = receivers(met["Cross Section Interpolation Surfaces"], "", np.asarray(areas), identities)
            # Standard acre-foot: 43,560 cubic feet, using 0.3048 metres per foot.
            grid_volume = increments.sum(axis=1) * size * size * 0.0254 / 1233.48183754752
        return {"value_semantics": value_semantics, "meteorology_metadata": metadata,
            "start": saved_times[0].isoformat(), "end": saved_times[-1].isoformat(),
            "interval_end": [t.isoformat() for t in timestamps], "interval_seconds": interval.total_seconds(),
            "grid_total_volume_af": float(grid_volume.sum()), "grid_interval_volume_af": grid_volume.tolist(),
            "grid_total_depth_in": total_depth.tolist(), "one_d": one_d, "two_d": two_d,
            "source_label_grid_volume_af": {label: float(total_depth[idx].sum() * size * size * 0.0254
                                                        / 1233.48183754752)
                                            for label, idx in label_indexes.items()},
            "total_two_d_volume_af": sum(area["total_volume_af"] for area in two_d.values()),
            "total_receiving_volume_af": one_d["total_volume_af"] + sum(
                area["total_volume_af"] for area in two_d.values()),
            "limitations": ["Prescribed rainfall from stored weights, not solver-admitted water or excess after losses.",
                "Summed 1D and 2D recipients do not establish geometrically disjoint footprints.",
                "Imported value semantics must be checked against independently authenticated source intervals.",
                "No hydraulic acceptance or production correction is assigned."]}

    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def get_precipitation_footprint_overlap(hdf_path: Path, *, value_semantics: str) -> Dict:
        """Measure stored 1D interpolation-surface / physical 2D cell overlap.

        Requires complete valid polygons and consistent IDs; never repairs or
        clips model geometry. Measures overlap in native foot coordinates.
        Rainfall-on-overlap uses each recipient's uniform prescribed depth,
        not a separate integration of the original rainfall field. Neither
        overlap nor these two side-specific volumes is a hydraulic correction.
        Stored polygons may differ from the physical computational footprint.
        """
        from shapely.ops import unary_union
        from .HdfMesh import HdfMesh
        from .HdfXsec import HdfXsec

        rainfall = HdfResultsPlan.get_precipitation_receiving_diagnostics(hdf_path, value_semantics=value_semantics)
        cells = HdfMesh.get_mesh_cell_polygons(hdf_path)
        surfaces = HdfXsec.get_xs_interpolation_surface(hdf_path)
        if cells.empty or surfaces.empty or cells.crs is None or cells.crs != surfaces.crs:
            raise ValueError("Complete receiving polygons with a common CRS are required")
        if (not cells.geometry.is_valid.all() or not surfaces.geometry.is_valid.all()
                or not (cells.geometry.area > 0).all() or not (surfaces.geometry.area > 0).all()):
            raise ValueError("Invalid receiving polygons; no automatic repair is permitted")
        expected_cells = {(name, rec["identity"]) for name, area in rainfall["two_d"].items()
                          for rec in area["receivers"]}
        actual_cells = list(zip(cells.mesh_name, cells.cell_id))
        if len(set(actual_cells)) != len(actual_cells) or set(actual_cells) != expected_cells:
            raise ValueError("Incomplete or duplicate physical cell polygons")
        with h5py.File(hdf_path, "r") as source:
            pairs = source["Geometry/Cross Section Interpolation Surfaces/XSIDs"][:]
        surface_ids = list(surfaces.surface_id)
        if len(set(surface_ids)) != len(surface_ids) or set(surface_ids) != set(range(len(pairs))):
            raise ValueError("Incomplete interpolation-surface polygons")
        for row in surfaces.itertuples():
            if (row.us_xs_id, row.ds_xs_id) != tuple(pairs[row.surface_id]):
                raise ValueError("Interpolation-surface identities differ")
        one_union, two_union = unary_union(surfaces.geometry), unary_union(cells.geometry)
        both = one_union.intersection(two_union)
        per_area = []
        two_d_rain = {(name, rec["identity"]): rec for name, area in rainfall["two_d"].items()
                     for rec in area["receivers"]}
        for name, group in cells.groupby("mesh_name", sort=True):
            union = unary_union(group.geometry)
            overlap = union.intersection(one_union)
            rain_volume = sum(row.geometry.intersection(one_union).area
                              * two_d_rain[(name, row.cell_id)]["depth_in"] / 12 / 43560
                              for row in group.itertuples())
            per_area.append({"area": name, "polygon_area_ft2": float(union.area),
                "overlap_with_one_d_ft2": float(overlap.area),
                "two_d_prescribed_rain_on_overlap_af": float(rain_volume)})
        one_d_on_overlap = sum(row.geometry.intersection(two_union).area
            * rainfall["one_d"]["receivers"][row.surface_id]["depth_in"] / 12 / 43560
            for row in surfaces.itertuples())
        return {"units": "square feet", "crs": str(cells.crs),
            "one_d_surface_count": len(surfaces), "two_d_cell_count": len(cells),
            "one_d_union_area_ft2": float(one_union.area), "two_d_union_area_ft2": float(two_union.area),
            "one_d_sum_minus_union_ft2": float(surfaces.geometry.area.sum() - one_union.area),
            "two_d_sum_minus_union_ft2": float(cells.geometry.area.sum() - two_union.area),
            "one_d_two_d_overlap_ft2": float(both.area),
            "one_d_prescribed_rain_on_overlap_af": float(one_d_on_overlap), "areas": per_area,
            "maximum_one_d_polygon_minus_accounting_area_ft2": max(abs(row.geometry.area
                - rainfall["one_d"]["receivers"][row.surface_id]["area_ft2"]) for row in surfaces.itertuples()),
            "maximum_two_d_polygon_minus_accounting_area_ft2": max(abs(row.geometry.area
                - two_d_rain[(row.mesh_name, row.cell_id)]["area_ft2"]) for row in cells.itertuples()),
            "limitations": ["Stored map polygons are not proof of solver computational footprint or duplicate water.",
                "Overlap rainfall uses each recipient's uniform prescribed depth; the two sides can differ.",
                "No overlap volume is automatically removable from forcing or hydraulic accounting."]}

    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def get_precipitation_diagnostics(hdf_path: Path) -> Dict:
        """Audit stored gridded forcing and 1D cumulative precipitation.

        Fingerprint every preprocessed meteorology dataset (including spatial
        weights) and reconstruct volume from saved cross-section control-volume
        depths and areas. This checks stored output, not solver source code or
        water actually admitted to the hydraulic equations. Currently requires
        gridded, period-cumulative precipitation in inches and areas in ft^2.
        Missing output, ambiguous units, invalid areas, and nonfinite depths
        raise instead of yielding a misleading zero. No acceptance is assigned.
        """
        import hashlib
        from .HdfBase import HdfBase

        met_path = "Event Conditions/Meteorology/Precipitation"
        base = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
        cv_path = f"{base}/Cross Sections Control Volume"
        with h5py.File(hdf_path, "r") as source:
            met = source[met_path]
            metadata = HdfUtils.convert_hdf5_attrs_to_dict(met.attrs)
            if (metadata.get("Mode"), metadata.get("Data Type"), metadata.get("Units")) != (
                "Gridded", "per-cum", "in"
            ):
                raise ValueError("Precipitation audit requires Gridded/per-cum/in metadata")
            fingerprints = {}

            def fingerprint(name, dataset):
                if isinstance(dataset, h5py.Dataset):
                    values = dataset[()]
                    if dataset.dtype.hasobject:
                        raise ValueError(f"Unsupported variable-length meteorology dataset: {name}")
                    fingerprints[name] = {
                        "shape": list(dataset.shape),
                        "dtype": str(dataset.dtype),
                        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                    }

            met.visititems(fingerprint)
            for required in ("Timestamp", "Values"):
                if required not in fingerprints:
                    raise KeyError(f"Missing precipitation dataset: {required}")
            forcing_times = [v.decode("utf-8").strip() for v in met["Timestamp"][:]]
            forcing_values = met["Values"][:]
            if forcing_values.ndim != 2 or forcing_values.shape[0] != len(forcing_times) or not forcing_times:
                raise ValueError("Gridded precipitation shape/time mismatch")
            if not np.isfinite(forcing_values).all():
                raise ValueError("Nonfinite gridded precipitation values")
            depth_ds = source[f"{cv_path}/Cumulative Precipitation Depth"]
            depth_metadata = HdfUtils.convert_hdf5_attrs_to_dict(depth_ds.attrs)
            if depth_metadata.get("Units") != "in":
                raise ValueError("Control-volume precipitation depths must be in inches")
            depths = depth_ds[:].astype(np.float64)
            attrs = source[f"{cv_path}/XS CV Attributes"][:]
            fields = {"River", "Reach", "Station US", "Station DS", "Surface Area (ft^2)"}
            if not fields.issubset(attrs.dtype.names or ()):
                raise ValueError("Missing cross-section control-volume identity or ft^2 area")
            areas = attrs["Surface Area (ft^2)"].astype(np.float64)
            times = HdfBase.get_unsteady_timestamps(source)
            if depths.shape != (len(times), len(attrs)) or len(times) < 2 or len(attrs) == 0:
                raise ValueError("Control-volume precipitation shape/time mismatch")
            if any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError("Precipitation output timestamps must strictly increase")
            if not np.isfinite(depths).all() or not np.isfinite(areas).all() or np.any(areas <= 0):
                raise ValueError("Invalid precipitation depths or control-volume areas")
            identities = [tuple(row[key].decode("utf-8").strip() for key in (
                "River", "Reach", "Station US", "Station DS"
            )) for row in attrs]
            if len(set(identities)) != len(identities):
                raise ValueError("Duplicate cross-section control-volume identities")
            volumes = depths @ areas / (12.0 * 43560.0)
            controls = []
            for index, identity in enumerate(identities):
                controls.append(dict(zip(("river", "reach", "station_us", "station_ds"), identity)) | {
                    "area_ft2": float(areas[index]),
                    "initial_depth_in": float(depths[0, index]),
                    "final_depth_in": float(depths[-1, index]),
                    "depth_series_sha256": hashlib.sha256(depths[:, index].tobytes()).hexdigest(),
                    "final_volume_af": float(depths[-1, index] * areas[index] / (12.0 * 43560.0)),
                })
            one_d = HdfUtils.convert_hdf5_attrs_to_dict(
                source["Results/Unsteady/Summary/Volume Accounting/Volume Accounting 1D"].attrs
            )
            native = one_d.get("Precip Excess (acre feet)")
            if one_d.get("Vol Accounting in") != "Acre Feet" or not isinstance(native, (int, float)):
                raise ValueError("Missing native 1D precipitation accounting in acre-feet")
            if not np.isfinite(native):
                raise ValueError("Nonfinite native 1D precipitation accounting")
        return {
            "meteorology_metadata": metadata,
            "meteorology_datasets": fingerprints,
            "forcing_timestamp_count": len(forcing_times),
            "forcing_first_timestamp": forcing_times[0],
            "forcing_last_timestamp": forcing_times[-1],
            "control_volumes": controls,
            "time": [t.isoformat() for t in times],
            "saved_cumulative_volume_af": volumes.tolist(),
            "saved_final_volume_af": float(volumes[-1]),
            "saved_window_volume_af": float(volumes[-1] - volumes[0]),
            "native_1d_precipitation_af": native,
            "saved_final_minus_native_af": float(volumes[-1] - native),
            "limitations": [
                "Saved cumulative depths may be precipitation, while native accounting labels excess precipitation.",
                "A mismatch does not establish actual hydraulic input loss or a unique engine defect.",
                "Dataset fingerprints cover stored meteorology, not all native execution internals.",
                "No engineering acceptance is assigned.",
            ],
        }

    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def get_exchange_diagnostics(
        hdf_path: Path, boundary_flow_signs: Optional[Dict[str, int]] = None
    ) -> Dict:
        """Crosswalk lateral/SA connections and integrate saved boundary flows.

        Structure flow is positive from the geometry's US side to its DS side.
        External BC signs must be supplied explicitly: +1 into the 2D area,
        -1 out. Unspecified nonzero BCs prevent an area reconciliation; zero
        inactive BCs are retained. Returns raw sampled hydrographs, topology,
        signed trapezoidal volumes in native cubic units, and per-area sums.
        These are sampled estimates, not the solver's every-step accounting.
        Missing structure results, ambiguous topology, incompatible units,
        nonfinite flows or incomplete/nonmonotonic time axes raise errors.
        No engineering acceptance threshold is applied.
        """
        from .HdfBase import HdfBase

        def decode(value):
            return value.decode('utf-8').strip() if isinstance(value, bytes) else str(value).strip()

        signs = boundary_flow_signs or {}
        if any(isinstance(sign, bool) or sign not in (-1, 1) for sign in signs.values()):
            raise ValueError('Boundary signs must be +1 (in) or -1 (out)')
        base = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series'
        records, boundaries = [], []
        with h5py.File(hdf_path, 'r') as hdf:
            times = HdfBase.get_unsteady_timestamps(hdf)
            seconds = np.array([(value - times[0]).total_seconds() for value in times])
            if len(times) < 2 or np.any(np.diff(seconds) <= 0):
                raise ValueError('Exchange integration requires increasing timestamps')
            area_names = [decode(row['Name']) for row in hdf['Geometry/2D Flow Areas/Attributes'][:]]
            if len(set(area_names)) != len(area_names):
                raise ValueError('Duplicate 2D area names')
            net = {area: 0.0 for area in area_names}
            unresolved = {area: [] for area in area_names}
            flow_units = set()
            mapped_structures = {'Lateral Structures': set(), 'SA 2D Area Conn': set()}

            def integrate(values, unit, label):
                values = np.asarray(values, dtype=float)
                if values.shape != seconds.shape or not np.isfinite(values).all():
                    raise ValueError(f'Invalid/incomplete flow series: {label}')
                if unit not in ('cfs', 'm3/s', 'm^3/s'):
                    raise ValueError(f'Unsupported flow unit {unit!r}: {label}')
                flow_units.add('cfs' if unit == 'cfs' else 'm3/s')
                return float(np.sum((values[1:] + values[:-1]) * 0.5 * np.diff(seconds)))

            geometry = hdf.get('Geometry/Structures/Attributes')
            for row in geometry[:] if geometry is not None else []:
                kind = decode(row['Type'])
                if kind not in ('Lateral', 'Connection'):
                    continue
                if 'LW Span Multiple' in row.dtype.names and row['LW Span Multiple']:
                    raise ValueError('Multi-reach lateral exchanges require segment-level reconciliation')
                if kind == 'Lateral':
                    name = ' '.join(decode(row[field]) for field in ('River', 'Reach', 'RS'))
                    group = 'Lateral Structures'
                else:
                    name = decode(row['Connection'])
                    group = 'SA 2D Area Conn'
                if name in mapped_structures[group]:
                    raise ValueError(f'Duplicate structure identity: {group}/{name}')
                mapped_structures[group].add(name)
                dataset = hdf[f'{base}/{group}/{name}/Structure Variables']
                columns = [(decode(pair[0]), decode(pair[1])) for pair in dataset.attrs['Variable_Unit']]
                indices = [i for i, (label, _) in enumerate(columns) if label == 'Total Flow']
                if len(indices) != 1:
                    raise ValueError(f'Missing/ambiguous Total Flow: {name}')
                index = indices[0]
                flow = dataset[:, index]
                volume = integrate(flow, columns[index][1], name)
                us, ds = decode(row['US SA/2D']), decode(row['DS SA/2D'])
                for side, area, multiplier in (('US', us, -1), ('DS', ds, 1)):
                    if decode(row[f'{side} Type']) == '2D':
                        if area not in net:
                            raise ValueError(f'Unknown {side} 2D area {area!r}: {name}')
                        net[area] += multiplier * volume
                records.append({'name': name, 'kind': kind, 'us_type': decode(row['US Type']),
                                'ds_type': decode(row['DS Type']), 'us_area': us, 'ds_area': ds,
                                'flow_units': columns[index][1], 'signed_volume': volume,
                                'flow': flow.astype(float).tolist()})
            for group, mapped in mapped_structures.items():
                stored = hdf.get(f'{base}/{group}')
                result_names = set(stored) if stored is not None else set()
                if result_names != mapped:
                    raise ValueError(f'Unmapped structure results in {group}: {sorted(result_names - mapped)}')
            bc_group = hdf.get(f'{base}/Boundary Conditions')
            if bc_group is not None:
                for name, dataset in bc_group.items():
                    if 'Columns' not in dataset.attrs:
                        continue  # Per-face arrays duplicate the aggregate BC flow.
                    columns = [decode(value) for value in dataset.attrs['Columns']]
                    if 'Flow' not in columns:
                        continue
                    area = decode(dataset.attrs['2D Area'])
                    if area not in net:
                        raise ValueError(f'Unknown boundary area {area!r}: {name}')
                    flow = dataset[:, columns.index('Flow')]
                    unit = decode(dataset.attrs['Flow'])
                    volume = integrate(flow, unit, name)
                    sign = signs.get(name)
                    if sign is not None:
                        net[area] += sign * volume
                    elif np.any(flow != 0):
                        unresolved[area].append(name)
                    boundaries.append({'name': name, 'area': area, 'flow_units': unit,
                                       'into_area_sign': sign, 'signed_volume': volume,
                                       'flow': flow.astype(float).tolist()})
            unknown = set(signs) - {row['name'] for row in boundaries}
            if unknown:
                raise ValueError(f'Unknown boundary flow sign keys: {sorted(unknown)}')
            if len(flow_units) != 1:
                raise ValueError(f'Incompatible or missing exchange units: {flow_units}')
        return {'time': [value.isoformat() for value in times],
                'volume_units': 'ft^3' if flow_units == {'cfs'} else 'm^3',
                'integration': 'trapezoid over all saved output samples',
                'structures': records, 'boundaries': boundaries,
                'areas': [{'area': area, 'net_nonprecipitation_volume': net[area]
                           if not unresolved[area] else None,
                           'unresolved_boundaries': unresolved[area]} for area in area_names],
                'limitations': ['Sampled transfers cannot reproduce every-step native accounting exactly.',
                                'Total Flow is shared exchange output, not independent audits of both solvers.',
                                'No engineering tolerance or hydraulic acceptance is assigned.']}

    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def get_coupling_diagnostics(
        hdf_path: Path, lateral_names: List[str], cross_sections: List[Tuple[str, str, str]]
    ) -> Dict:
        """Read saved lateral segments and cross-section cumulative flow.

        Lateral names are exact native result group names. Cross sections use
        (river, reach, station) identities. Returns JSON-compatible native
        labels, units and arrays on the common output time axis, not iteration
        histories. Tailwater cell labels are preserved without assuming their
        indexing convention. Segment endpoint arrays and interval cell labels
        intentionally have different lengths. This read-only diagnostic also
        supports failed complete runs and assigns no execution/QA acceptance.
        Currently requires English-unit lateral and cross-section output.
        """
        from .HdfBase import HdfBase

        def decode(value):
            return value.decode("utf-8").strip() if isinstance(value, bytes) else str(value).strip()

        if not lateral_names or len(set(lateral_names)) != len(lateral_names):
            raise ValueError("Require distinct lateral names")
        if any(not name or "/" in name for name in lateral_names):
            raise ValueError("Use exact lateral result group names")
        identities = [tuple(identity) for identity in cross_sections]
        if not identities or any(len(identity) != 3 for identity in identities) or len(set(identities)) != len(identities):
            raise ValueError("Require distinct (river, reach, station) cross sections")
        base = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
        with h5py.File(hdf_path, "r") as source:
            times = HdfBase.get_unsteady_timestamps(source)
            if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError("Coupling diagnostics require increasing timestamps")

            def numeric(dataset, shape):
                values = dataset[:].astype(float)
                if values.shape != shape or not np.isfinite(values).all():
                    raise ValueError(f"Invalid shape or nonfinite values: {dataset.name}")
                return values.tolist()

            laterals = []
            for name in lateral_names:
                group = source[f"{base}/Lateral Structures/{name}"]
                variables = group["Structure Variables"]
                columns = [(decode(pair[0]), decode(pair[1])) for pair in variables.attrs["Variable_Unit"]]
                if len({label for label, _ in columns}) != len(columns):
                    raise ValueError(f"Duplicate structure variables: {name}")
                if dict(columns).get("Total Flow") != "cfs" or any(
                    unit != "ft" for label, unit in columns if label.startswith("Stage ")
                ):
                    raise ValueError(f"Expected English structure units: {name}")
                values = np.asarray(numeric(variables, (len(times), len(columns))))
                if any(np.any(values[:, i] <= -9990) for i, (label, _) in enumerate(columns)
                       if label.startswith("Stage ")):
                    raise ValueError(f"Missing structure stage: {name}")
                segments = group["HW TW Segments"]
                stations = segments["HW TW Station"][:].astype(float)
                if stations.ndim != 1 or len(stations) < 2 or not np.isfinite(stations).all() or np.any(np.diff(stations) <= 0):
                    raise ValueError(f"Invalid segment stations: {name}")
                cells = [decode(value) for value in segments["Tailwater Cells"][:]]
                river_stations = [decode(value) for value in segments["Headwater River Stations"][:]]
                if len(cells) != len(stations) - 1 or len(river_stations) != len(stations):
                    raise ValueError(f"Invalid segment connectivity: {name}")
                series = {}
                for label, unit in (("Flow", "cfs"), ("Water Surface HW", "ft"), ("Water Surface TW", "ft")):
                    dataset = segments[label]
                    if decode(dataset.attrs["Units"]) != unit:
                        raise ValueError(f"Unexpected segment units: {name}/{label}")
                    series[label] = numeric(dataset, (len(times), len(stations)))
                    if label.startswith("Water Surface") and np.any(np.asarray(series[label]) <= -9990):
                        raise ValueError(f"Missing segment stage: {name}/{label}")
                laterals.append({
                    "name": name,
                    "variables": {label: {"units": unit, "values": values[:, i].tolist()}
                                  for i, (label, unit) in enumerate(columns)},
                    "segment_stations_ft": stations.tolist(), "tailwater_cell_labels": cells,
                    "headwater_river_stations": river_stations, "segments": series,
                })
            group = source[f"{base}/Cross Sections"]
            attrs = group["Cross Section Attributes"][:]
            available = [tuple(decode(row[key]) for key in ("River", "Reach", "Station")) for row in attrs]
            if len(set(available)) != len(available):
                raise ValueError("Duplicate cross-section identities")
            flow_ds, volume_ds = group["Flow"], group["Flow Volume Cumulative"]
            if decode(flow_ds.attrs["Variable Units"]) != "cfs" or decode(
                volume_ds.attrs["Cumulative Volumetric Flow"]
            ) != "Feet^3":
                raise ValueError("Expected cfs/Feet^3 cross-section flow units")
            flow = np.asarray(numeric(flow_ds, (len(times), len(attrs))))
            volume = np.asarray(numeric(volume_ds, (len(times), len(attrs))))
            sections = []
            for identity in identities:
                if identity not in available:
                    raise ValueError(f"Unknown cross section: {identity}")
                index = available.index(identity)
                sections.append({
                    "river": identity[0], "reach": identity[1], "station": identity[2],
                    "flow_cfs": flow[:, index].tolist(), "cumulative_flow_ft3": volume[:, index].tolist(),
                })
        return {"time": [value.isoformat() for value in times], "laterals": laterals, "cross_sections": sections,
                "limitations": ["Saved output does not contain every coupling iteration.",
                                "Cell labels retain the native indexing convention.",
                                "No execution or hydraulic acceptance is assigned."]}

    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def get_runtime_data(hdf_path: Path) -> Optional[pd.DataFrame]:
        """
        Extract detailed runtime and computational performance metrics from HDF file.

        Args:
            hdf_path (Path): Path to HEC-RAS plan HDF file
            ras_object (RasPrj, optional): Specific RAS object to use. If None, uses the global ras instance.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing runtime statistics or None if data cannot be extracted

        Notes:
            - Times are reported in multiple units (ms, s, hours)
            - Compute speeds are calculated as simulation-time/compute-time ratios
            - Process times include: geometry, preprocessing, event conditions, 
              and unsteady flow computations
        """
        try:
            if hdf_path is None:
                logger.error(f"Could not find HDF file for input")
                return None

            with h5py.File(hdf_path, 'r') as hdf_file:
                logger.debug(f"Extracting Plan Information from: {Path(hdf_file.filename).name}")
                plan_info = hdf_file.get('/Plan Data/Plan Information')
                if plan_info is None:
                    logger.debug(
                        "Runtime metadata group '/Plan Data/Plan Information' "
                        "not found in %s.",
                        Path(hdf_file.filename).name,
                    )
                    return None

                # Extract plan information
                plan_name = HdfUtils.convert_ras_string(plan_info.attrs.get('Plan Name', 'Unknown'))
                start_time_str = HdfUtils.convert_ras_string(plan_info.attrs.get('Simulation Start Time', 'Unknown'))
                end_time_str = HdfUtils.convert_ras_string(plan_info.attrs.get('Simulation End Time', 'Unknown'))

                try:
                    # Check if times are already datetime objects
                    if isinstance(start_time_str, datetime):
                        start_time = start_time_str
                    else:
                        start_time = datetime.strptime(start_time_str, "%d%b%Y %H:%M:%S")
                        
                    if isinstance(end_time_str, datetime):
                        end_time = end_time_str
                    else:
                        end_time = datetime.strptime(end_time_str, "%d%b%Y %H:%M:%S")
                        
                    simulation_duration = end_time - start_time
                    simulation_hours = simulation_duration.total_seconds() / 3600
                except ValueError as e:
                    logger.debug(
                        "Runtime metadata unavailable for %s: could not parse simulation times (%s)",
                        Path(hdf_file.filename).name,
                        e,
                    )
                    return None

                logger.debug(f"Plan Name: {plan_name}")
                logger.debug(f"Simulation Duration (hours): {simulation_hours}")

                # Extract compute processes data
                compute_processes = hdf_file.get('/Results/Summary/Compute Processes')
                if compute_processes is None:
                    logger.warning("Dataset '/Results/Summary/Compute Processes' not found.")
                    return None

                # Process compute times
                process_names = [HdfUtils.convert_ras_string(name) for name in compute_processes['Process'][:]]
                filenames = [HdfUtils.convert_ras_string(filename) for filename in compute_processes['Filename'][:]]
                completion_times = compute_processes['Compute Time (ms)'][:]

                compute_processes_df = pd.DataFrame({
                    'Process': process_names,
                    'Filename': filenames,
                    'Compute Time (ms)': completion_times,
                    'Compute Time (s)': completion_times / 1000,
                    'Compute Time (hours)': completion_times / (1000 * 3600)
                })

                # Create summary DataFrame
                compute_processes_summary = {
                    'Plan Name': [plan_name],
                    'File Name': [Path(hdf_file.filename).name],
                    'Simulation Start Time': [start_time_str],
                    'Simulation End Time': [end_time_str],
                    'Simulation Duration (s)': [simulation_duration.total_seconds()],
                    'Simulation Time (hr)': [simulation_hours]
                }

                # Add process-specific times
                process_types = {
                    'Completing Geometry': 'Completing Geometry (hr)',
                    'Preprocessing Geometry': 'Preprocessing Geometry (hr)',
                    'Completing Event Conditions': 'Completing Event Conditions (hr)',
                    'Unsteady Flow Computations': 'Unsteady Flow Computations (hr)'
                }

                for process, column in process_types.items():
                    time_value = compute_processes_df[
                        compute_processes_df['Process'] == process
                    ]['Compute Time (hours)'].values[0] if process in process_names else 'N/A'
                    compute_processes_summary[column] = [time_value]

                # Add total process time
                total_time = compute_processes_df['Compute Time (hours)'].sum()
                compute_processes_summary['Complete Process (hr)'] = [total_time]

                # Calculate speeds
                if compute_processes_summary['Unsteady Flow Computations (hr)'][0] != 'N/A':
                    compute_processes_summary['Unsteady Flow Speed (hr/hr)'] = [
                        simulation_hours / compute_processes_summary['Unsteady Flow Computations (hr)'][0]
                    ]
                else:
                    compute_processes_summary['Unsteady Flow Speed (hr/hr)'] = ['N/A']

                compute_processes_summary['Complete Process Speed (hr/hr)'] = [
                    simulation_hours / total_time
                ]

                return pd.DataFrame(compute_processes_summary)

        except Exception as e:
            logger.error(f"Error in get_runtime_data: {str(e)}")
            return None

    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def get_reference_timeseries(hdf_path: Path, reftype: str) -> pd.DataFrame:
        """
        Get reference line or point timeseries output from HDF file.

        Args:
            hdf_path (Path): Path to HEC-RAS plan HDF file
            reftype (str): Type of reference data ('lines' or 'points')
            ras_object (RasPrj, optional): Specific RAS object to use. If None, uses the global ras instance.

        Returns:
            pd.DataFrame: DataFrame containing reference timeseries data
        """
        try:
            with h5py.File(hdf_path, 'r') as hdf_file:
                base_path = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
                ref_path = f"{base_path}/Reference {reftype.capitalize()}"
                
                if ref_path not in hdf_file:
                    logger.debug(f"Reference {reftype} data not found in HDF file")
                    return pd.DataFrame()

                ref_group = hdf_file[ref_path]
                time_data = hdf_file[f"{base_path}/Time"][:]
                
                dfs = []
                for ref_name in ref_group.keys():
                    ref_data = ref_group[ref_name][:]
                    df = pd.DataFrame(ref_data, columns=[ref_name])
                    df['Time'] = time_data
                    dfs.append(df)

                if not dfs:
                    return pd.DataFrame()

                return pd.concat(dfs, axis=1)

        except Exception as e:
            logger.error(f"Error reading reference {reftype} timeseries: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def get_reference_summary(hdf_path: Path, reftype: str) -> pd.DataFrame:
        """
        Get reference line or point summary output from HDF file.

        Args:
            hdf_path (Path): Path to HEC-RAS plan HDF file
            reftype (str): Type of reference data ('lines' or 'points')
            ras_object (RasPrj, optional): Specific RAS object to use. If None, uses the global ras instance.

        Returns:
            pd.DataFrame: DataFrame containing reference summary data
        """
        try:
            with h5py.File(hdf_path, 'r') as hdf_file:
                base_path = "Results/Unsteady/Output/Output Blocks/Base Output/Summary Output"
                ref_path = f"{base_path}/Reference {reftype.capitalize()}"
                
                if ref_path not in hdf_file:
                    logger.debug(f"Reference {reftype} summary data not found in HDF file")
                    return pd.DataFrame()

                ref_group = hdf_file[ref_path]
                dfs = []
                
                for ref_name in ref_group.keys():
                    ref_data = ref_group[ref_name][:]
                    if ref_data.ndim == 2:
                        df = pd.DataFrame(ref_data.T, columns=['Value', 'Time'])
                    else:
                        df = pd.DataFrame({'Value': ref_data})
                    df['Reference'] = ref_name
                    dfs.append(df)

                if not dfs:
                    return pd.DataFrame()

                return pd.concat(dfs, ignore_index=True)

        except Exception as e:
            logger.error(f"Error reading reference {reftype} summary: {str(e)}")
            return pd.DataFrame()

    # ==================== STEADY STATE METHODS ====================

    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def is_steady_plan(hdf_path: Path) -> bool:
        """
        Check if HDF file contains steady state results.

        Args:
            hdf_path (Path): Path to HEC-RAS plan HDF file
            ras_object (RasPrj, optional): Specific RAS object to use. If None, uses the global ras instance.

        Returns:
            bool: True if the HDF contains steady state results, False otherwise

        Notes:
            - Checks for existence of Results/Steady group
            - Does not guarantee results are complete or valid
        """
        try:
            with h5py.File(hdf_path, 'r') as hdf_file:
                return "Results/Steady" in hdf_file
        except Exception as e:
            logger.debug(f"Error checking if plan is steady: {str(e)}")
            return False

    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def get_steady_profile_names(hdf_path: Path) -> List[str]:
        """
        Extract profile names from steady state results.

        Args:
            hdf_path (Path): Path to HEC-RAS plan HDF file
            ras_object (RasPrj, optional): Specific RAS object to use. If None, uses the global ras instance.

        Returns:
            List[str]: List of profile names (e.g., ['50Pct', '10Pct', '1Pct'])

        Raises:
            FileNotFoundError: If the specified HDF file is not found
            KeyError: If steady state results or profile names are not found
            ValueError: If the plan is not a steady state plan

        Example:
            >>> from ras_commander import HdfResultsPlan, init_ras_project
            >>> init_ras_project(Path('/path/to/project'), '6.6')
            >>> profiles = HdfResultsPlan.get_steady_profile_names('01')
            >>> print(profiles)
            ['50Pct', '20Pct', '10Pct', '4Pct', '2Pct', '1Pct', '0.2Pct']
        """
        try:
            with h5py.File(hdf_path, 'r') as hdf_file:
                # Check if this is a steady state plan
                if "Results/Steady" not in hdf_file:
                    raise ValueError(f"HDF file does not contain steady state results: {hdf_path.name}")

                # Path to profile names
                profile_names_path = "Results/Steady/Output/Output Blocks/Base Output/Steady Profiles/Profile Names"

                if profile_names_path not in hdf_file:
                    raise KeyError(f"Profile names not found at: {profile_names_path}")

                # Read profile names dataset
                profile_names_ds = hdf_file[profile_names_path]
                profile_names_raw = profile_names_ds[()]

                # Decode byte strings to regular strings
                profile_names = []
                for name in profile_names_raw:
                    if isinstance(name, bytes):
                        profile_names.append(name.decode('utf-8').strip())
                    else:
                        profile_names.append(str(name).strip())

                logger.debug(f"Found {len(profile_names)} steady state profiles: {profile_names}")
                return profile_names

        except FileNotFoundError:
            raise FileNotFoundError(
                f"HDF file not found: {hdf_path}. "
                f"See: https://rascommander.info/ras/user-guide/hdf-data-extraction/"
            )
        except KeyError as e:
            raise KeyError(f"Error accessing steady state profile names: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error reading steady state profile names: {str(e)}")

    @staticmethod
    def _get_steady_cross_section_attributes(
        hdf_file: h5py.File,
        expected_count: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, str]]:
        """Return steady cross-section identifiers across HEC-RAS HDF versions.

        Newer plan HDFs write identifiers beside steady output.  Several
        HEC-RAS 6.x files instead retain the equivalent table in geometry,
        where river station is named ``RS`` rather than ``Station``.
        """
        attribute_paths = (
            "Results/Steady/Output/Geometry Info/Cross Section Attributes",
            "Geometry/Cross Sections/Attributes",
        )
        for attribute_path in attribute_paths:
            if attribute_path not in hdf_file:
                continue

            attributes = hdf_file[attribute_path][()]
            field_names = attributes.dtype.names or ()
            field_lookup = {name.lower(): name for name in field_names}
            station_field = field_lookup.get("station") or field_lookup.get("rs")
            required_fields = {
                "river": field_lookup.get("river"),
                "reach": field_lookup.get("reach"),
                "station": station_field,
            }
            missing_fields = [name for name, field in required_fields.items() if field is None]
            if missing_fields:
                logger.debug(
                    "Ignoring steady cross-section attributes at %s because fields are missing: %s",
                    attribute_path,
                    ", ".join(missing_fields),
                )
                continue
            if expected_count is not None and len(attributes) != expected_count:
                raise ValueError(
                    "Steady cross-section identifier count does not match result values: "
                    f"{attribute_path} has {len(attributes)} rows; results have {expected_count}."
                )
            return attributes, required_fields

        paths = " or ".join(attribute_paths)
        raise KeyError(f"Cross section attributes not found at: {paths}")

    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def get_steady_wse(
        hdf_path: Path,
        profile_index: Optional[int] = None,
        profile_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract water surface elevation (WSE) data for steady state profiles.

        Args:
            hdf_path (Path): Path to HEC-RAS plan HDF file
            profile_index (int, optional): Index of profile to extract (0-based). If None, extracts all profiles.
            profile_name (str, optional): Name of profile to extract (e.g., '1Pct'). If specified, overrides profile_index.
            ras_object (RasPrj, optional): Specific RAS object to use. If None, uses the global ras instance.

        Returns:
            pd.DataFrame: DataFrame containing WSE data with columns:
                - River: River name
                - Reach: Reach name
                - Station: Cross section river station
                - Profile: Profile name (if multiple profiles)
                - WSE: Water surface elevation (ft)

        Raises:
            FileNotFoundError: If the specified HDF file is not found
            KeyError: If steady state results or WSE data are not found
            ValueError: If profile_index or profile_name is invalid

        Example:
            >>> # Extract single profile by index
            >>> wse_df = HdfResultsPlan.get_steady_wse('01', profile_index=5)  # 100-year profile

            >>> # Extract single profile by name
            >>> wse_df = HdfResultsPlan.get_steady_wse('01', profile_name='1Pct')

            >>> # Extract all profiles
            >>> wse_df = HdfResultsPlan.get_steady_wse('01')
        """
        try:
            with h5py.File(hdf_path, 'r') as hdf_file:
                # Check if this is a steady state plan
                if "Results/Steady" not in hdf_file:
                    raise ValueError(f"HDF file does not contain steady state results: {hdf_path.name}")

                # Paths to data
                wse_path = "Results/Steady/Output/Output Blocks/Base Output/Steady Profiles/Cross Sections/Water Surface"
                profile_names_path = "Results/Steady/Output/Output Blocks/Base Output/Steady Profiles/Profile Names"

                # Check required paths exist
                if wse_path not in hdf_file:
                    raise KeyError(f"WSE data not found at: {wse_path}")
                # Get WSE dataset (shape: num_profiles × num_cross_sections)
                wse_ds = hdf_file[wse_path]
                wse_data = wse_ds[()]
                num_profiles, num_xs = wse_data.shape

                # Get profile names
                if profile_names_path in hdf_file:
                    profile_names_raw = hdf_file[profile_names_path][()]
                    profile_names = [
                        name.decode('utf-8').strip() if isinstance(name, bytes) else str(name).strip()
                        for name in profile_names_raw
                    ]
                else:
                    # Fallback to numbered profiles
                    profile_names = [f"Profile_{i+1}" for i in range(num_profiles)]

                # Get cross section attributes
                xs_attrs, xs_fields = HdfResultsPlan._get_steady_cross_section_attributes(
                    hdf_file,
                    expected_count=num_xs,
                )

                # Determine which profiles to extract
                if profile_name is not None:
                    # Find profile by name
                    try:
                        profile_idx = profile_names.index(profile_name)
                    except ValueError:
                        raise ValueError(
                            f"Profile name '{profile_name}' not found. "
                            f"Available profiles: {profile_names}"
                        )
                    profiles_to_extract = [(profile_idx, profile_name)]

                elif profile_index is not None:
                    # Validate profile index
                    if profile_index < 0 or profile_index >= num_profiles:
                        raise ValueError(
                            f"Profile index {profile_index} out of range. "
                            f"Valid range: 0 to {num_profiles-1}"
                        )
                    profiles_to_extract = [(profile_index, profile_names[profile_index])]

                else:
                    # Extract all profiles
                    profiles_to_extract = list(enumerate(profile_names))

                # Build DataFrame
                rows = []
                for prof_idx, prof_name in profiles_to_extract:
                    wse_values = wse_data[prof_idx, :]

                    for xs_idx in range(num_xs):
                        river = xs_attrs[xs_idx][xs_fields['river']]
                        reach = xs_attrs[xs_idx][xs_fields['reach']]
                        station = xs_attrs[xs_idx][xs_fields['station']]

                        # Decode byte strings
                        river = river.decode('utf-8') if isinstance(river, bytes) else str(river)
                        reach = reach.decode('utf-8') if isinstance(reach, bytes) else str(reach)
                        station = station.decode('utf-8') if isinstance(station, bytes) else str(station)

                        row = {
                            'River': river.strip(),
                            'Reach': reach.strip(),
                            'Station': station.strip(),
                            'WSE': float(wse_values[xs_idx])
                        }

                        # Only add Profile column if extracting multiple profiles
                        if len(profiles_to_extract) > 1:
                            row['Profile'] = prof_name

                        rows.append(row)

                df = pd.DataFrame(rows)

                # Reorder columns
                if 'Profile' in df.columns:
                    df = df[['River', 'Reach', 'Station', 'Profile', 'WSE']]
                else:
                    df = df[['River', 'Reach', 'Station', 'WSE']]

                logger.debug(
                    f"Extracted WSE data for {len(profiles_to_extract)} profile(s), "
                    f"{num_xs} cross sections"
                )

                return df

        except FileNotFoundError:
            raise FileNotFoundError(
                f"HDF file not found: {hdf_path}. "
                f"See: https://rascommander.info/ras/user-guide/hdf-data-extraction/"
            )
        except KeyError as e:
            raise KeyError(f"Error accessing steady state WSE data: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error reading steady state WSE data: {str(e)}")

    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def get_steady_info(hdf_path: Path) -> pd.DataFrame:
        """
        Get steady flow attributes and metadata from HEC-RAS HDF plan file.

        Args:
            hdf_path (Path): Path to HEC-RAS plan HDF file
            ras_object (RasPrj, optional): Specific RAS object to use. If None, uses the global ras instance.

        Returns:
            pd.DataFrame: DataFrame containing steady flow attributes including:
                - Program Name
                - Program Version
                - Type of Run
                - Run Time Window
                - Solution status
                - And other metadata attributes

        Raises:
            FileNotFoundError: If the specified HDF file is not found
            KeyError: If steady state results are not found
            ValueError: If the plan is not a steady state plan

        Example:
            >>> info_df = HdfResultsPlan.get_steady_info('01')
            >>> print(info_df['Solution'].values[0])
            'Steady Finished Successfully'
        """
        try:
            with h5py.File(hdf_path, 'r') as hdf_file:
                # Check if this is a steady state plan
                if "Results/Steady" not in hdf_file:
                    raise ValueError(f"HDF file does not contain steady state results: {hdf_path.name}")

                attrs_dict = {}

                # Get attributes from Results/Steady/Output
                output_path = "Results/Steady/Output"
                if output_path in hdf_file:
                    output_group = hdf_file[output_path]
                    for key, value in output_group.attrs.items():
                        if isinstance(value, bytes):
                            attrs_dict[key] = value.decode('utf-8')
                        else:
                            attrs_dict[key] = value

                # Get attributes from Results/Steady/Summary
                summary_path = "Results/Steady/Summary"
                if summary_path in hdf_file:
                    summary_group = hdf_file[summary_path]
                    for key, value in summary_group.attrs.items():
                        if isinstance(value, bytes):
                            attrs_dict[key] = value.decode('utf-8')
                        else:
                            attrs_dict[key] = value

                # Add flow file information from Plan Data
                plan_info_path = "Plan Data/Plan Information"
                if plan_info_path in hdf_file:
                    plan_info = hdf_file[plan_info_path]
                    for key in ['Flow Filename', 'Flow Title']:
                        if key in plan_info.attrs:
                            value = plan_info.attrs[key]
                            if isinstance(value, bytes):
                                attrs_dict[key] = value.decode('utf-8')
                            else:
                                attrs_dict[key] = value

                if not attrs_dict:
                    logger.warning("No steady state attributes found in HDF file")
                    return pd.DataFrame()

                logger.debug(f"Extracted {len(attrs_dict)} steady state attributes")
                return pd.DataFrame(attrs_dict, index=[0])

        except FileNotFoundError:
            raise FileNotFoundError(
                f"HDF file not found: {hdf_path}. "
                f"See: https://rascommander.info/ras/user-guide/hdf-data-extraction/"
            )
        except KeyError as e:
            raise KeyError(f"Error accessing steady state info: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error reading steady state info: {str(e)}")

    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def get_compute_messages(hdf_path: Path) -> str:
        """
        Read computation messages from HDF file with fallback to .txt file.

        Extracts computation messages from the HDF Results/Summary structure.
        This includes detailed information about the computation process,
        warnings, errors, convergence information, and performance metrics.

        If HDF path not found, falls back to .txt file extraction using RasControl.

        Args:
            hdf_path: Path to plan HDF file (or plan number string if using
                     standardize_input decorator, which resolves to HDF path)

        Returns:
            String containing computation messages, or empty string if unavailable

        Example:
            >>> from ras_commander import init_ras_project, HdfResultsPlan
            >>> init_ras_project(r"/path/to/project", "6.5")
            >>> msgs = HdfResultsPlan.get_compute_messages("01")
            >>> print(msgs)

        Note:
            Modern HEC-RAS versions (6.x+) store computation messages in HDF:
            /Results/Summary/Compute Messages (text)

            Older versions (pre-6.x) use .txt files which are accessed via
            fallback to RasControl.get_comp_msgs()

            Function naming follows HDF structure conventions (get_compute_messages)
            vs RasControl legacy naming (get_comp_msgs) to reflect technological lineage.
        """
        try:
            with h5py.File(hdf_path, 'r') as hdf_file:
                # Define HDF path for compute messages
                compute_msgs_path = "Results/Summary/Compute Messages (text)"

                # Check if path exists in HDF
                if compute_msgs_path not in hdf_file:
                    logger.debug(
                        f"Compute Messages not found in HDF at '{compute_msgs_path}', "
                        f"falling back to .txt file extraction"
                    )

                    # Fallback to .txt file using RasControl
                    try:
                        # Late import to avoid circular dependency
                        from ..RasControl import RasControl

                        # Extract plan info from HDF path
                        # e.g., "C:/path/BaldEagle.p10.hdf" -> use path for RasControl
                        txt_contents = RasControl.get_comp_msgs(hdf_path)
                        if txt_contents:
                            logger.debug(f"Successfully retrieved {len(txt_contents)} characters from .txt file")
                            return txt_contents
                    except Exception as e:
                        logger.debug(f".txt file fallback failed: {e}")

                    # Both methods failed
                    logger.debug(
                        f"No computation messages found in HDF or .txt sources for {hdf_path.name}"
                    )
                    return ""

                # Read dataset from HDF
                logger.debug(f"Reading computation messages from HDF: {hdf_path.name}")
                dataset = hdf_file[compute_msgs_path]
                data = dataset[()]

                # Decode byte string to UTF-8
                if isinstance(data, bytes):
                    contents = data.decode('utf-8', errors='ignore')
                elif isinstance(data, np.ndarray) and len(data) > 0:
                    # Handle array of byte strings
                    if isinstance(data[0], bytes):
                        contents = data[0].decode('utf-8', errors='ignore')
                    else:
                        contents = str(data[0])
                else:
                    contents = str(data)

                logger.debug(f"Successfully extracted {len(contents)} characters from HDF")
                return contents

        except FileNotFoundError:
            logger.debug(f"HDF file not found: {hdf_path}")

            # Try .txt fallback
            try:
                from ..RasControl import RasControl
                txt_contents = RasControl.get_comp_msgs(hdf_path)
                if txt_contents:
                    logger.debug(
                        f"HDF file not found, successfully retrieved computation messages from .txt file"
                    )
                    return txt_contents
            except Exception as e:
                logger.debug(f".txt file fallback failed: {e}")

            logger.debug(f"No computation messages found for {hdf_path.name}")
            return ""

        except Exception as e:
            logger.debug(f"Error reading computation messages from HDF: {str(e)}")

            # Try .txt fallback on any HDF error
            try:
                from ..RasControl import RasControl
                txt_contents = RasControl.get_comp_msgs(hdf_path)
                if txt_contents:
                    logger.debug(
                        f"HDF extraction failed, successfully retrieved computation messages from .txt file"
                    )
                    return txt_contents
            except Exception as fallback_error:
                logger.debug(f".txt file fallback failed: {fallback_error}")

            logger.debug(f"No computation messages found for {hdf_path.name}")
            return ""

    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def get_compute_messages_hdf_only(hdf_path: Path) -> str:
        """
        Extract compute messages from HDF file or .txt files (no RasControl fallback).

        This method reads computation messages without using RasControl/COM interface,
        making it suitable for automated workflows where COM locking is problematic.

        Args:
            hdf_path: Path to plan HDF file (or plan number string if using
                     standardize_input decorator, which resolves to HDF path)

        Returns:
            str: Compute messages text, or empty string if unavailable

        Example:
            >>> from ras_commander import init_ras_project, HdfResultsPlan
            >>> init_ras_project(r"/path/to/project", "6.5")
            >>> msgs = HdfResultsPlan.get_compute_messages_hdf_only("01")
            >>> print(msgs)

        Note:
            Falls back to .txt files on disk but NEVER uses RasControl.
            Order of attempts:
            1. HDF Results/Summary/Compute Messages (text)
            2. {plan_file}.computeMsgs.txt (HEC-RAS 6.x+)
            3. {plan_file}.comp_msgs.txt (HEC-RAS 5.x)
        """
        # Attempt 1: Read from HDF file
        try:
            with h5py.File(hdf_path, 'r') as hdf_file:
                compute_msgs_path = "Results/Summary/Compute Messages (text)"

                if compute_msgs_path in hdf_file:
                    logger.debug(f"Reading computation messages from HDF: {hdf_path.name}")
                    dataset = hdf_file[compute_msgs_path]
                    data = dataset[()]

                    # Decode byte string to UTF-8
                    if isinstance(data, bytes):
                        contents = data.decode('utf-8', errors='ignore')
                    elif isinstance(data, np.ndarray) and len(data) > 0:
                        # Handle array of byte strings
                        if isinstance(data[0], bytes):
                            contents = data[0].decode('utf-8', errors='ignore')
                        else:
                            contents = str(data[0])
                    else:
                        contents = str(data)

                    logger.debug(f"Successfully extracted {len(contents)} characters from HDF")
                    return contents
                else:
                    logger.debug(
                        f"Compute Messages not found in HDF at '{compute_msgs_path}', "
                        f"trying .txt file fallbacks"
                    )
        except FileNotFoundError:
            logger.debug(f"HDF file not found: {hdf_path}")
        except Exception as e:
            logger.debug(f"Error reading computation messages from HDF: {str(e)}")

        # Attempt 2: Read .computeMsgs.txt file (HEC-RAS 6.x+)
        try:
            # Convert HDF path to .computeMsgs.txt path
            # e.g., "plan.p01.hdf" -> "plan.p01.computeMsgs.txt"
            txt_path_6x = Path(str(hdf_path).replace('.hdf', '.computeMsgs.txt'))
            if txt_path_6x.exists():
                contents = txt_path_6x.read_text(encoding='utf-8', errors='ignore')
                logger.debug(f"Successfully read {len(contents)} characters from {txt_path_6x.name}")
                return contents
            else:
                logger.debug(f".computeMsgs.txt file not found: {txt_path_6x}")
        except Exception as e:
            logger.debug(f"Error reading .computeMsgs.txt file: {str(e)}")

        # Attempt 3: Read .comp_msgs.txt file (HEC-RAS 5.x)
        try:
            # Convert HDF path to .comp_msgs.txt path
            # e.g., "plan.p01.hdf" -> "plan.p01.comp_msgs.txt"
            txt_path_5x = Path(str(hdf_path).replace('.hdf', '.comp_msgs.txt'))
            if txt_path_5x.exists():
                contents = txt_path_5x.read_text(encoding='utf-8', errors='ignore')
                logger.debug(f"Successfully read {len(contents)} characters from {txt_path_5x.name}")
                return contents
            else:
                logger.debug(f".comp_msgs.txt file not found: {txt_path_5x}")
        except Exception as e:
            logger.debug(f"Error reading .comp_msgs.txt file: {str(e)}")

        # All methods failed - return empty string (no RasControl fallback)
        logger.debug(f"No computation messages found for {hdf_path.name} (HDF-only mode)")
        return ""

    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def get_steady_results(hdf_path: Path) -> pd.DataFrame:
        """
        Extract steady state profile results from HEC-RAS HDF file.

        This function extracts all key hydraulic results for steady state
        profiles in a single call, matching the schema of RasControl.get_steady_results()
        for consistency between COM and HDF-based workflows.

        Parameters
        ----------
        hdf_path : Path
            Path to HEC-RAS plan HDF file (.p##.hdf)

        Returns
        -------
        pd.DataFrame
            Steady state results with one row per cross-section per profile.

            **Schema:**

            +----------------+----------+---------------------------------------+
            | Column         | Type     | Description                           |
            +================+==========+=======================================+
            | river          | str      | River name                            |
            +----------------+----------+---------------------------------------+
            | reach          | str      | Reach name                            |
            +----------------+----------+---------------------------------------+
            | node_id        | str      | Cross section river station           |
            +----------------+----------+---------------------------------------+
            | profile        | str      | Profile name (e.g., "PF 1", "1Pct")   |
            +----------------+----------+---------------------------------------+
            | wsel           | float    | Water surface elevation (ft or m)     |
            +----------------+----------+---------------------------------------+
            | velocity       | float    | Channel velocity (ft/s or m/s)        |
            +----------------+----------+---------------------------------------+
            | flow           | float    | Total flow (cfs or cms)               |
            +----------------+----------+---------------------------------------+
            | froude         | float    | Channel Froude number (dimensionless) |
            +----------------+----------+---------------------------------------+
            | energy         | float    | Energy grade elevation (ft or m)      |
            +----------------+----------+---------------------------------------+
            | max_depth      | float    | Maximum channel depth (ft or m)       |
            +----------------+----------+---------------------------------------+
            | min_ch_el      | float    | Minimum channel elevation (ft or m)   |
            +----------------+----------+---------------------------------------+
            | top_width      | float    | Total top width (ft or m)             |
            +----------------+----------+---------------------------------------+
            | area           | float    | Total flow area (sq ft or sq m)       |
            +----------------+----------+---------------------------------------+
            | eg_slope       | float    | Energy grade slope (ft/ft or m/m)     |
            +----------------+----------+---------------------------------------+
            | friction_slope | float    | Friction slope (ft/ft or m/m)         |
            +----------------+----------+---------------------------------------+

        Raises
        ------
        ValueError
            If the HDF file does not contain steady state results.

        Notes
        -----
        **Comparison with RasControl.get_steady_results():**

        This HDF-based method provides the same schema as the COM-based
        RasControl.get_steady_results(), plus additional hydraulic variables
        (top_width, area, eg_slope, friction_slope) that are readily
        available in the HDF file.

        **Performance:**

        This method is significantly faster than COM-based extraction
        since it reads directly from the HDF file without opening HEC-RAS.

        Examples
        --------
        Extract all steady results:

        >>> from ras_commander import init_ras_project, HdfResultsPlan
        >>> init_ras_project("/path/to/project", "7.0")
        >>> df = HdfResultsPlan.get_steady_results("01")
        >>> df.to_csv('steady_results.csv', index=False)

        Filter by profile:

        >>> profile_1 = df[df['profile'] == 'PF 1']

        Plot water surface profile:

        >>> import matplotlib.pyplot as plt
        >>> plt.plot(profile_1['node_id'].astype(float), profile_1['wsel'])
        >>> plt.xlabel('Station')
        >>> plt.ylabel('Water Surface Elevation (ft)')

        See Also
        --------
        RasControl.get_steady_results : COM-based steady results extraction
        is_steady_plan : Check if plan contains steady results
        get_steady_profile_names : Get list of profile names
        """
        try:
            with h5py.File(hdf_path, 'r') as hdf_file:
                if "Results/Steady" not in hdf_file:
                    raise ValueError(
                        f"HDF file does not contain steady state results: {hdf_path.name}\n"
                        "Ensure this is a steady flow plan that has been computed."
                    )

                # Paths
                base_path = "Results/Steady/Output/Output Blocks/Base Output/Steady Profiles"
                xs_path = f"{base_path}/Cross Sections"
                add_vars_path = f"{xs_path}/Additional Variables"
                profile_names_path = f"{base_path}/Profile Names"

                # Get profile names
                if profile_names_path in hdf_file:
                    profile_names_raw = hdf_file[profile_names_path][()]
                    profile_names = [
                        name.decode('utf-8').strip() if isinstance(name, bytes) else str(name).strip()
                        for name in profile_names_raw
                    ]
                else:
                    raise ValueError("Profile names not found in HDF file")

                # Get main variables (WSE, Flow)
                wse_data = hdf_file[f"{xs_path}/Water Surface"][()]
                flow_data = hdf_file[f"{xs_path}/Flow"][()]
                num_profiles, num_xs = wse_data.shape
                xs_attrs, xs_fields = HdfResultsPlan._get_steady_cross_section_attributes(
                    hdf_file,
                    expected_count=num_xs,
                )

                # Missing optional variables are returned as null values. Do
                # not infer a cross-section join from legacy packed arrays;
                # their axis ordering and row coverage vary by HEC-RAS version.
                def get_additional_var(var_name, default=np.nan):
                    path = f"{add_vars_path}/{var_name}"
                    if path in hdf_file:
                        return hdf_file[path][()]
                    return np.full((num_profiles, num_xs), default)

                velocity_data = get_additional_var('Velocity Channel')
                energy_data = get_additional_var('Energy Grade')
                froude_data = get_additional_var('Froude # Channel')
                max_depth_data = get_additional_var('Hydraulic Depth Channel')
                min_ch_el_data = get_additional_var('Min Ch El')
                top_width_data = get_additional_var('Top Width Total')
                area_data = get_additional_var('Area Flow Total')
                eg_slope_data = get_additional_var('EG Slope')
                friction_slope_data = get_additional_var('Friction Slope')

                # Build results DataFrame
                rows = []
                for prof_idx, prof_name in enumerate(profile_names):
                    for xs_idx in range(num_xs):
                        river = xs_attrs[xs_idx][xs_fields['river']]
                        reach = xs_attrs[xs_idx][xs_fields['reach']]
                        station = xs_attrs[xs_idx][xs_fields['station']]

                        # Decode byte strings
                        river = river.decode('utf-8').strip() if isinstance(river, bytes) else str(river).strip()
                        reach = reach.decode('utf-8').strip() if isinstance(reach, bytes) else str(reach).strip()
                        station = station.decode('utf-8').strip() if isinstance(station, bytes) else str(station).strip()

                        rows.append({
                            'river': river,
                            'reach': reach,
                            'node_id': station,
                            'profile': prof_name,
                            'wsel': float(wse_data[prof_idx, xs_idx]),
                            'velocity': float(velocity_data[prof_idx, xs_idx]),
                            'flow': float(flow_data[prof_idx, xs_idx]),
                            'froude': float(froude_data[prof_idx, xs_idx]),
                            'energy': float(energy_data[prof_idx, xs_idx]),
                            'max_depth': float(max_depth_data[prof_idx, xs_idx]),
                            'min_ch_el': float(min_ch_el_data[prof_idx, xs_idx]),
                            'top_width': float(top_width_data[prof_idx, xs_idx]),
                            'area': float(area_data[prof_idx, xs_idx]),
                            'eg_slope': float(eg_slope_data[prof_idx, xs_idx]),
                            'friction_slope': float(friction_slope_data[prof_idx, xs_idx]),
                        })

                df = pd.DataFrame(rows)
                logger.debug(f"Extracted steady results: {len(df)} rows "
                            f"({num_profiles} profiles x {num_xs} cross sections)")
                return df

        except Exception as e:
            logger.error(f"Error reading steady results: {str(e)}")
            raise

    @staticmethod
    @log_call
    @standardize_input(file_type='plan_hdf')
    def list_steady_variables(hdf_path: Path) -> Dict[str, List[str]]:
        """
        List all available steady state variables in the HDF file.

        This is a diagnostic function useful for exploring what data
        is available in a particular steady state results file.

        Parameters
        ----------
        hdf_path : Path
            Path to HEC-RAS plan HDF file

        Returns
        -------
        Dict[str, List[str]]
            Dictionary with keys:
            - 'cross_sections': Variables in Cross Sections group
            - 'additional': Variables in Additional Variables group
            - 'structures': Variables in Structures group (if present)

        Example
        -------
        >>> vars = HdfResultsPlan.list_steady_variables('01')
        >>> print(vars['additional'])
        ['Area Flow Channel', 'Velocity Channel', 'Top Width Total', ...]
        """
        try:
            result = {
                'cross_sections': [],
                'additional': [],
                'structures': []
            }

            with h5py.File(hdf_path, 'r') as hdf_file:
                if "Results/Steady" not in hdf_file:
                    logger.debug("No steady state results in this file")
                    return result

                base_path = "Results/Steady/Output/Output Blocks/Base Output/Steady Profiles"

                # Cross Sections variables
                xs_path = f"{base_path}/Cross Sections"
                if xs_path in hdf_file:
                    xs_group = hdf_file[xs_path]
                    for key in xs_group.keys():
                        if isinstance(xs_group[key], h5py.Dataset):
                            result['cross_sections'].append(key)

                # Additional Variables
                add_path = f"{xs_path}/Additional Variables"
                if add_path in hdf_file:
                    add_group = hdf_file[add_path]
                    for key in add_group.keys():
                        if isinstance(add_group[key], h5py.Dataset):
                            result['additional'].append(key)

                # Structures
                struct_path = f"{base_path}/Structures"
                if struct_path in hdf_file:
                    struct_group = hdf_file[struct_path]
                    for key in struct_group.keys():
                        if isinstance(struct_group[key], h5py.Dataset):
                            result['structures'].append(key)

                logger.debug(f"Found {len(result['cross_sections'])} XS vars, "
                            f"{len(result['additional'])} additional vars, "
                            f"{len(result['structures'])} structure vars")

                return result

        except Exception as e:
            logger.error(f"Error listing steady variables: {str(e)}")
            return {'cross_sections': [], 'additional': [], 'structures': []}
