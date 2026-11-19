"""Read-only admission of provisional static forecast initialization.

The orchestrator owns full package inventory authentication. This boundary
matches its pinned manifest declaration and authenticates the selected native
text files; it never infers observed state or converts a physical quantity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .RasPlan import RasPlan, _iter_plan_key_lines
from .RasPrj import RasPrj, init_ras_project
from .RasScenario import RasScenario, RasScenarioWorkspace
from .RasUnsteady import RasUnsteady
from .RasUtils import RasUtils

_HASH_FIELDS = {
    "package_sha256", "model_manifest_sha256", "plan_sha256", "geometry_sha256",
    "unsteady_sha256", "engine_sha256", "hydrology_sha256", "forcing_sha256",
}
_CONTROLS = {"Use Restart", "Restart Filename", "Prior WS Filename", "Prior WS Profile"}
_RECORDS = {"Initial Flow Loc": 4, "Initial Storage Elev": 2, "Initial RRR Elev": 4}
_POINT = "IC Point Elev"
_CORE_KEYS = {"UNET D1 Cores", "UNET D2 Cores", "PS Cores"}


def _validate_binding(binding: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    required = _HASH_FIELDS | {"method", "disposition", "anchor_time", "predecessor", "reference_policy"}
    if not isinstance(binding, dict) or set(binding) != required:
        raise ValueError("initialization requires exactly the version 1.3 binding fields")
    if binding["method"] != "static" or binding["disposition"] != "provisional-static":
        raise ValueError("Only explicit provisional-static initialization is supported")
    if binding["predecessor"] is not None:
        raise ValueError("Static initialization must have null predecessor; restart lineage is unsupported")
    if binding["reference_policy"] != "clear-in-clone":
        raise ValueError("Static initialization requires explicit reference_policy clear-in-clone")
    for field in _HASH_FIELDS:
        if not isinstance(binding[field], str) or not re.fullmatch(r"[0-9a-f]{64}", binding[field]):
            raise ValueError(f"initialization.{field} must be a lowercase SHA-256")
    if request["model_window"]["time_zone"] != "UTC":
        raise ValueError("Forecast initialization requires model_window.time_zone UTC")
    anchor = binding["anchor_time"]
    if not isinstance(anchor, str) or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", anchor):
        raise ValueError("initialization.anchor_time requires a UTC second-precision instant ending in Z")
    instant = datetime.fromisoformat(anchor.replace("Z", "+00:00"))
    model_start = datetime.fromisoformat(request["model_window"]["start"]).replace(tzinfo=timezone.utc)
    if instant != model_start:
        raise ValueError("initialization.anchor_time must equal the model-window start")
    for field, expected in {
        "package_sha256": request["source_model"]["package_sha256"],
        "model_manifest_sha256": request["source_model"]["model_manifest_sha256"],
        "hydrology_sha256": request["hydrology"]["sha256"],
        "forcing_sha256": request["forcing_excess"]["sha256"],
    }.items():
        if binding[field] != expected:
            raise ValueError(f"initialization.{field} does not match the independently selected input")
    return dict(binding)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lines(path: Path) -> list[str]:
    # The legacy reader replaces undecodable bytes. Admission must not hash a
    # substituted state or selector as though it were the original declaration.
    path.read_bytes().decode("utf-8", errors="strict")
    lines, _ = RasUtils._read_text_lines_preserving_newline(path)
    return lines


def _single_value(lines: list[str], key: str, *, required: bool = True) -> str | None:
    matching = [line.rstrip("\r\n").split("=", 1)[1].strip()
                for line in lines if line.startswith(key + "=")]
    if len(matching) > 1 or (required and not matching):
        raise ValueError(f"Expected one unambiguous {key} declaration")
    return matching[0] if matching else None


def _project_units(project_file: Path) -> str | None:
    declarations = set()
    for line in _lines(project_file):
        value = line.strip().casefold()
        if value == "english units":
            declarations.add("ft")
        elif value == "si units":
            declarations.add("m")
        elif value.startswith("si units="):
            flag = value.split("=", 1)[1].strip().casefold()
            if flag not in {"0", "1", "false", "true", "yes", "no", "on", "off"}:
                raise ValueError("Unknown native SI Units selector")
            declarations.add("m" if flag in {"1", "true", "yes", "on"} else "ft")
    if len(declarations) > 1:
        raise ValueError("Conflicting native project-unit declarations")
    return RasPrj.get_project_units(project_file)


def _plan_configuration(path: Path) -> str:
    # Scenario preparation owns labels, description, window and flow reference;
    # execution owns the requested core count. Preserve all remaining settings,
    # including warmup and numerical controls that affect the initial solution.
    changed_keys = _CORE_KEYS | {"Plan Title", "Short Identifier", "Simulation Date", "Flow File"}
    lines = _lines(path)
    changed_indices = {index for index, key, _ in _iter_plan_key_lines(lines) if key in changed_keys}
    values = []
    in_description = False
    for index, line in enumerate(lines):
        marker = line.strip().upper()
        if in_description:
            if marker.startswith("END DESCRIPTION"):
                in_description = False
            continue
        if marker.startswith("BEGIN DESCRIPTION"):
            in_description = "END DESCRIPTION" not in marker
            continue
        if index not in changed_indices and line.strip():
            values.append(line.rstrip("\r\n"))
    return hashlib.sha256(json.dumps(values, ensure_ascii=True).encode("utf-8")).hexdigest()


def _set_clone_cores(request: Mapping[str, Any], workspace: RasScenarioWorkspace) -> None:
    project = init_ras_project(
        workspace.project_file, ras_version=request["execution"]["ras_executable"],
        ras_object=RasPrj(), load_results_summary=False, load_hdf_metadata=False, hide_intro=True,
    )
    RasPlan.set_num_cores(workspace.plan_number, request["execution"]["cores"],
                          ras_object=project, refresh_dataframes=False)


def _selected_files(project_file: Path, plan_number: str, executable: str) -> dict[str, Path]:
    # Metadata loading resolves the selected plan through the same owner API
    # used by preparation, without HDF inspection or native engine startup.
    project = init_ras_project(
        project_file, ras_version=executable, ras_object=RasPrj(),
        load_results_summary=False, load_hdf_metadata=False, hide_intro=True,
    )
    number = RasUtils.normalize_ras_number(plan_number)
    selected = project.plan_df.loc[project.plan_df["plan_number"] == number]
    if len(selected) != 1:
        raise ValueError("Initialization requires one selected plan in project metadata")
    plan_path = RasPlan.get_plan_path(number, ras_object=project)
    if plan_path is None:
        raise ValueError("Selected initialization plan does not exist")
    plan_lines = _lines(plan_path)
    geometry_reference = _single_value(plan_lines, "Geom File")
    flow_reference = _single_value(plan_lines, "Flow File")
    if not re.fullmatch(r"g\d{2}", geometry_reference or ""):
        raise ValueError("Selected plan requires an explicit native geometry reference")
    if not re.fullmatch(r"u\d{2}", flow_reference or ""):
        raise ValueError("Selected plan requires an explicit unsteady-flow reference")
    geometry = RasPlan.get_geom_path(geometry_reference[1:], ras_object=project)
    unsteady = RasPlan.get_unsteady_path(flow_reference[1:], ras_object=project)
    files = {"project": project_file, "plan": plan_path, "geometry": geometry, "unsteady": unsteady}
    for label, path in files.items():
        if path is None or not path.is_file() or path.resolve().parent != project_file.parent.resolve():
            raise ValueError(f"Selected {label} must be an existing file within the project directory")
    # Reject ambiguous source current-plan declarations without asserting
    # that the source's current plan or historical window is the new forecast.
    _single_value(_lines(project_file), "Current Plan")
    return files


def _static_snapshot(unsteady: Path, geometry: Path, *, allow_references: bool = False) -> dict[str, Any]:
    lines = _lines(unsteady)
    selected_lines: list[str] = []
    control_lines: list[str] = []
    seen_records: set[tuple[str, ...]] = set()
    past_boundary = False
    known = _CONTROLS | set(_RECORDS) | {_POINT}
    for line in lines:
        key, separator, value = line.rstrip("\r\n").partition("=")
        stripped_key = key.strip()
        if stripped_key == "Boundary Location":
            past_boundary = True
        if stripped_key.casefold() not in {key.casefold() for key in known}:
            continue
        if not separator or stripped_key != key or key not in known:
            raise ValueError(f"Noncanonical initialization record: {stripped_key}")
        if past_boundary:
            raise ValueError(f"Initialization record occurs after a boundary: {key}")
        if key in _CONTROLS:
            control_lines.append(line)
        else:
            selected_lines.append(line)
        if key in _RECORDS:
            fields = [part.strip() for part in value.split(",")]
            if len(fields) != _RECORDS[key] or any(not part for part in fields):
                raise ValueError(f"Malformed {key} record")
            numeric = fields[-2:] if len(fields) == 4 else fields[-1:]
            if any(not math.isfinite(float(part)) for part in numeric):
                raise ValueError(f"Nonfinite {key} record")
            identity = (key, *(part.casefold() for part in fields[:-1]))
            if identity in seen_records:
                raise ValueError(f"Duplicate {key} selector")
            seen_records.add(identity)
    controls = {}
    for key in sorted(_CONTROLS):
        value = _single_value(lines, key, required=key == "Use Restart")
        controls[key] = value
        if key == "Use Restart":
            if value not in {"0", "false", "False"}:
                raise ValueError("Static admission requires explicit Use Restart=0 or False")
        elif value and not allow_references:
            raise ValueError(f"Static admission does not support populated {key}")
    method = RasUnsteady.get_initial_flow_method(unsteady)
    if not allow_references and method["method"] not in {"none", "initial_flow_distribution"}:
        raise ValueError("The selected unsteady file does not use static initial conditions")
    conditions = RasUnsteady.get_initial_conditions(unsteady)
    points = RasUnsteady.get_initial_point_elevations(unsteady)
    if len(conditions) != len(seen_records) or (conditions.empty and points.empty):
        raise ValueError("Static admission requires explicit, completely parsed initial conditions")
    geometry_lines = _lines(geometry)
    geometry_points = [line.rstrip("\r\n").split("=", 1)[1].strip().casefold()
                       for line in geometry_lines if line.startswith("IC Point Name=")]
    for name in points["point_name"]:
        if geometry_points.count(name.casefold()) != 1:
            raise ValueError(f"Initial point {name!r} must match exactly one selected geometry IC point")
    return {
        "sha256": hashlib.sha256("".join(selected_lines).encode("utf-8")).hexdigest(),
        "controls_sha256": hashlib.sha256("".join(control_lines).encode("utf-8")).hexdigest(),
        "controls": controls,
        "record_count": len(conditions) + len(points),
        "initial_condition_count": len(conditions),
        "initial_point_count": len(points),
        "reported_method": method["method"],
    }


def _verify_source(request: Mapping[str, Any]) -> dict[str, Any]:
    binding = request["initialization"]
    source = request["source_model"]
    manifest = json.loads(Path(source["model_manifest"]).read_text(encoding="utf-8"))
    summary = manifest.get("content_summary")
    if (manifest.get("contract_type") != "floodforecast/model-inventory/2.0.0"
            or not isinstance(summary, dict)
            or summary.get("content_tree_sha256") != source["package_sha256"]):
        raise ValueError("Pinned RAS model-inventory 2.0.0 manifest must declare the selected package digest")
    executable = Path(request["execution"]["ras_executable"])
    if _sha256(executable) != binding["engine_sha256"]:
        raise ValueError("Selected RAS executable does not match initialization.engine_sha256")
    project_file = RasScenario._resolve_project_file(Path(source["project"]))
    files = _selected_files(project_file, source["template_plan"], str(executable))
    selected = {}
    for label, path in files.items():
        digest = _sha256(path)
        expected = source["project_file_sha256"] if label == "project" else binding[label + "_sha256"]
        if digest != expected:
            raise ValueError(f"Selected {label} does not match the initialization binding")
        selected[label] = {"name": path.name, "sha256": digest}
    snapshot = _static_snapshot(files["unsteady"], files["geometry"], allow_references=True)
    return {
        "schema": "ras-commander/forecast-initialization-evidence/1.0",
        "status": "source-verified",
        "binding": dict(binding),
        "selected_files": selected,
        "plan_configuration_sha256": _plan_configuration(files["plan"]),
        "initialization": snapshot,
        "package_verification": "pinned-manifest-declaration-and-selected-files",
        "state_validation": "unreviewed-source-declaration",
        "forecast_eligible": False,
        "source_project_units": _project_units(project_file),
        "vertical_datum": "unreviewed",
        "quantity_conversion": "none",
    }


def _verify_clone(
    request: Mapping[str, Any], workspace: RasScenarioWorkspace, source_evidence: Mapping[str, Any],
    *, require_static: bool = True,
) -> dict[str, Any]:
    root = Path(request["workspace"])
    source_root = RasScenario._resolve_project_file(Path(request["source_model"]["project"])).parent.resolve()
    if root.is_relative_to(source_root) or source_root.is_relative_to(root):
        raise ValueError("Forecast initialization requires a disjoint run clone, never the canonical source")
    if workspace.project_folder.resolve() != root:
        raise ValueError("Prepared workspace directory differs from the request")
    if workspace.project_file.resolve().parent != root or workspace.scenario_id != request["scenario"]["scenario_id"]:
        raise ValueError("Prepared project does not belong to the selected scenario workspace")
    for path in (workspace.hydrology_file, workspace.forcing_excess_file):
        if path is None or not path.resolve().is_relative_to(root):
            raise ValueError("Prepared hydrology and forcing must be inside the scenario workspace")
    files = _selected_files(workspace.project_file, workspace.plan_number, request["execution"]["ras_executable"])
    if _project_units(files["project"]) != source_evidence["source_project_units"]:
        raise ValueError("Prepared project changed the source units or their unknown disposition")
    if (files["plan"].resolve() != workspace.plan_file.resolve()
            or files["unsteady"].resolve() != workspace.unsteady_file.resolve()):
        raise ValueError("Prepared workspace does not identify its selected plan and unsteady file")
    if _single_value(_lines(files["project"]), "Current Plan") != "p" + workspace.plan_number:
        raise ValueError("Prepared initialization plan must be the project Current Plan")
    plan_lines = _lines(files["plan"])
    _single_value(plan_lines, "Simulation Date")
    expected_window = tuple(datetime.fromisoformat(request["model_window"][key]) for key in ("start", "end"))
    if RasScenario._parse_simulation_window("".join(plan_lines)) != expected_window:
        raise ValueError("Prepared initialization plan window differs from the request")
    if _plan_configuration(files["plan"]) != source_evidence["plan_configuration_sha256"]:
        raise ValueError("Prepared plan changed source numerical/initialization settings")
    if require_static:
        cores = [(key, raw.strip()) for _, key, raw in _iter_plan_key_lines(plan_lines) if key in _CORE_KEYS]
        if {key for key, _ in cores} != _CORE_KEYS or any(
            value != str(request["execution"]["cores"]) for _, value in cores
        ):
            raise ValueError("Prepared plan core settings differ from the requested execution")
    if _sha256(files["geometry"]) != request["initialization"]["geometry_sha256"]:
        raise ValueError("Prepared geometry differs from the authenticated initialization geometry")
    snapshot = _static_snapshot(files["unsteady"], files["geometry"], allow_references=not require_static)
    source_snapshot = source_evidence["initialization"]
    if any(snapshot[key] != source_snapshot[key] for key in (
        "sha256", "record_count", "initial_condition_count", "initial_point_count",
    )):
        raise ValueError("Prepared clone changed the exact static initialization fingerprint")
    if not require_static and snapshot != source_snapshot:
        raise ValueError("Unprepared clone changed source initialization controls")
    return {"status": "preserved" if require_static else "source-controls-preserved", "initialization": snapshot,
            "selected_files": {label: {"name": path.name, "sha256": _sha256(path)}
                               for label, path in files.items()}}
