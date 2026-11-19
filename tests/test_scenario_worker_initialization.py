"""Offline admission of explicit provisional static RAS initialization."""

import importlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from ras_commander import RasScenario, RasScenarioWorker
from ras_commander._forecast_initialization import _set_clone_cores
from ras_commander.RasScenarioWorker import RasScenarioWorkerError
from test_scenario_worker import _install_success_fakes, _request, _sha256


FIXTURE = Path(__file__).parent / "fixtures" / "forecast_initialization"


def _forecast_request(tmp_path):
    request, request_path, result_path = _request(tmp_path)
    source = Path(request["source_model"]["project"]).parent
    for path in FIXTURE.glob("Model.*"):
        # Native RAS writers emit CRLF even on Linux. Use the same explicit
        # convention for the source project, including untouched geometry.
        (source / path.name).write_text(
            path.read_text(encoding="utf-8"), encoding="utf-8", newline="\r\n",
        )
    manifest = Path(request["source_model"]["model_manifest"])
    manifest.write_text(json.dumps({
        "stage": "ras", "contract_type": "floodforecast/model-inventory/2.0.0",
        "content_summary": {"content_tree_sha256": "2" * 64},
    }), encoding="utf-8")
    request["schema"] = RasScenarioWorker.FORECAST_REQUEST_SCHEMA
    request["source_model"].update(
        package_sha256="2" * 64,
        project_file_sha256=_sha256(source / "Model.prj"),
        model_manifest_sha256=_sha256(manifest),
    )
    request["model_window"]["time_zone"] = "UTC"
    request["products"]["crs_policy"] = "strict"
    request["initialization"] = {
        "method": "static", "disposition": "provisional-static", "predecessor": None,
        "reference_policy": "clear-in-clone",
        "anchor_time": request["model_window"]["start"] + "Z",
        "package_sha256": request["source_model"]["package_sha256"],
        "model_manifest_sha256": _sha256(manifest),
        "plan_sha256": _sha256(source / "Model.p01"),
        "geometry_sha256": _sha256(source / "Model.g01"),
        "unsteady_sha256": _sha256(source / "Model.u01"),
        "engine_sha256": _sha256(Path(request["execution"]["ras_executable"])),
        "hydrology_sha256": request["hydrology"]["sha256"],
        "forcing_sha256": request["forcing_excess"]["sha256"],
    }
    request_path.write_text(json.dumps(request), encoding="utf-8")
    return request, request_path, result_path


def _save(request, request_path):
    request_path.write_text(json.dumps(request), encoding="utf-8")


def _install_pipeline(monkeypatch, request, *, change_clone=None):
    calls = _install_success_fakes(monkeypatch, request)
    source = Path(request["source_model"]["project"]).parent
    prepare_fake = RasScenario.prepare_workspace

    def prepare(*args, **kwargs):
        workspace = prepare_fake(*args, **kwargs)
        for filename in ("Model.prj", "Model.p01", "Model.g01", "Model.u01"):
            shutil.copyfile(source / filename, workspace.project_folder / filename)
        workspace.project_file.write_text(
            (source / "Model.prj").read_text().replace("Current Plan=p01", "Current Plan=p02")
            + "Plan File=p02\nUnsteady File=u02\n", encoding="utf-8",
        )
        workspace.plan_file.write_text(
            (source / "Model.p01").read_text().replace("Flow File=u01", "Flow File=u02")
            .replace("01JAN2020,0000,02JAN2020,0000", "18SEP2019,1300,19SEP2019,1300"),
            encoding="utf-8",
        )
        shutil.copyfile(source / "Model.u01", workspace.unsteady_file)
        if change_clone:
            change_clone(workspace)
        return workspace

    monkeypatch.setattr(RasScenario, "prepare_workspace", prepare)
    monkeypatch.setattr(RasScenario, "inspect_product_crs", lambda *a, **k: {
        "crs_resolution": {"export_allowed": True},
    })
    return calls


def test_source_verification_is_read_only_and_provisional(tmp_path, monkeypatch):
    request, path, _ = _forecast_request(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(RasScenario, "execute", lambda *a, **k: pytest.fail("Engine started"))
    evidence = RasScenarioWorker.verify_forecast_initialization(str(path))
    assert evidence["forecast_eligible"] is False
    assert evidence["state_validation"] == "unreviewed-source-declaration"
    assert evidence["initialization"]["initial_point_count"] == 1
    assert evidence["status"] == "source-verified"
    assert evidence["initialization"]["controls"]["Prior WS Filename"] == "Model.p03.hdf"
    assert evidence["quantity_conversion"] == "none"
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert not Path(request["workspace"]).exists()


@pytest.mark.parametrize("field,value", [
    ("method", "restart"), ("method", "provider-derived"),
    ("disposition", "observed"), ("predecessor", "prior-run"),
    ("reference_policy", "infer-inactive"),
    ("anchor_time", "2019-09-18T12:00:00Z"),
    ("anchor_time", "2019-09-18T13:00:00"),
    ("anchor_time", "2019-09-18T13:00:00.1Z"),
    ("package_sha256", "3" * 64), ("model_manifest_sha256", "3" * 64),
    ("hydrology_sha256", "3" * 64), ("forcing_sha256", "3" * 64),
])
def test_invalid_binding_refused_before_prepare(tmp_path, monkeypatch, field, value):
    request, path, result = _forecast_request(tmp_path)
    request["initialization"][field] = value
    _save(request, path)
    monkeypatch.setattr(RasScenario, "prepare_workspace", lambda *a, **k: pytest.fail("Clone created"))
    assert RasScenarioWorker.run(path, result) == 2
    assert json.loads(result.read_text())["execution"]["status"] == "not_started"


@pytest.mark.parametrize("target", ["project", "plan", "geometry", "unsteady", "engine", "hydrology", "forcing"])
def test_changed_selected_input_is_refused(tmp_path, target):
    request, path, _ = _forecast_request(tmp_path)
    source = Path(request["source_model"]["project"]).parent
    selected = {
        "project": source / "Model.prj", "plan": source / "Model.p01",
        "geometry": source / "Model.g01", "unsteady": source / "Model.u01",
        "engine": Path(request["execution"]["ras_executable"]),
        "hydrology": Path(request["hydrology"]["dss"]),
        "forcing": Path(request["forcing_excess"]["dss"]),
    }[target]
    selected.write_bytes(selected.read_bytes() + b"changed\n")
    with pytest.raises(RasScenarioWorkerError):
        RasScenarioWorker.verify_forecast_initialization(path)


@pytest.mark.parametrize("old,new", [
    ("Use Restart= 0", "Use Restart= unknown"),
    ("Use Restart= 0", "Use Restart= 1"),
    ("Use Restart= 0", "Use Restart= 2"),
    ("Use Restart= 0", "Use Restart= 0\nuse restart=1"),
    ("Use Restart= 0", "Use Restart= 0\nUse Restart= 0"),
    ("Use Restart= 0\n", ""),
    ("Prior WS Profile=", "Prior WS Profile=\nPrior WS Profile="),
    ("Restart Filename=", "Restart Filename=\nRestart Filename="),
    ("Prior WS Filename=", "Prior WS Filename=\nPrior WS Filename="),
    ("IC Point Elev=ReferencePoint,10.25,", "IC Point Elev=ReferencePoint,nan,"),
    ("IC Point Elev=ReferencePoint,10.25,", "IC Point Elev=Missing,10.25,"),
    ("IC Point Elev=ReferencePoint,10.25,", "IC Point Elev=ReferencePoint,10.25,\nIC Point Elev=ReferencePoint,10.25,"),
    ("IC Point Elev=ReferencePoint,10.25,", "Initial Flow Loc=River,Reach"),
    ("IC Point Elev=ReferencePoint,10.25,", "Initial Storage Elev=Area,nan"),
])
def test_ambiguous_or_unsupported_state_refused(tmp_path, old, new):
    request, path, _ = _forecast_request(tmp_path)
    unsteady = Path(request["source_model"]["project"]).with_suffix(".u01")
    unsteady.write_text(unsteady.read_text().replace(old, new), encoding="utf-8")
    request["initialization"]["unsteady_sha256"] = _sha256(unsteady)
    _save(request, path)
    with pytest.raises(RasScenarioWorkerError, match="initialization verification failed"):
        RasScenarioWorker.verify_forecast_initialization(path)


def test_selected_plan_cannot_bind_other_geometry(tmp_path):
    request, path, _ = _forecast_request(tmp_path)
    source = Path(request["source_model"]["project"]).parent
    other = source / "Model.g02"
    other.write_text("Geom Title=Other\n", encoding="utf-8")
    request["initialization"]["geometry_sha256"] = _sha256(other)
    _save(request, path)
    with pytest.raises(RasScenarioWorkerError, match="Selected geometry"):
        RasScenarioWorker.verify_forecast_initialization(path)


def test_worker_preserves_state_and_reauthenticates_reuse(tmp_path, monkeypatch):
    request, path, result = _forecast_request(tmp_path)
    calls = _install_pipeline(monkeypatch, request)
    assert RasScenarioWorker.run(path, result) == 0
    retained = result.read_bytes()
    assert json.loads(retained)["preparation"]["forecast_initialization"]["clone"]["status"] == "preserved"
    assert RasScenarioWorker.run(path, result) == 0
    assert calls["execute"] == 1
    engine = Path(request["execution"]["ras_executable"])
    engine.write_bytes(b"replacement engine")
    assert RasScenarioWorker.run(path, result) == 3
    assert result.read_bytes() == retained
    assert calls["execute"] == 1


@pytest.mark.parametrize("mutation", ["clone", "receipt", "missing_evidence", "boolean_type"])
def test_completed_result_reauthenticates_clone_and_receipt(tmp_path, monkeypatch, mutation):
    request, path, result = _forecast_request(tmp_path)
    calls = _install_pipeline(monkeypatch, request)
    assert RasScenarioWorker.run(path, result) == 0
    if mutation == "clone":
        unsteady = Path(request["workspace"]) / "Model.u02"
        unsteady.write_text(unsteady.read_text().replace("10.25", "12.25"), encoding="utf-8")
    else:
        payload = json.loads(result.read_text())
        if mutation == "receipt":
            payload["preparation"]["forecast_initialization"]["binding"]["anchor_time"] = "2020-01-01T00:00:00Z"
        elif mutation == "boolean_type":
            payload["preparation"]["forecast_initialization"]["forecast_eligible"] = 0
        else:
            del payload["preparation"]["forecast_initialization"]
        result.write_text(json.dumps(payload), encoding="utf-8")
    retained = result.read_bytes()
    assert RasScenarioWorker.run(path, result) == 3
    assert result.read_bytes() == retained
    assert calls["execute"] == 1


@pytest.mark.parametrize("field,value", [("time_zone", "America/Chicago"),
    ("start", "2019-09-18T13:00:01"), ("end", "2019-09-19T13:00:00.1")])
def test_ambiguous_model_clock_is_refused(tmp_path, field, value):
    request, path, _ = _forecast_request(tmp_path)
    request["model_window"][field] = value
    _save(request, path)
    with pytest.raises(RasScenarioWorkerError):
        RasScenarioWorker.verify_forecast_initialization(path)


@pytest.mark.parametrize("change", ["digest", "schema"])
def test_manifest_declaration_must_match_selected_package(tmp_path, change):
    request, path, _ = _forecast_request(tmp_path)
    manifest = Path(request["source_model"]["model_manifest"])
    payload = json.loads(manifest.read_text())
    if change == "digest":
        payload["content_summary"]["content_tree_sha256"] = "5" * 64
    else:
        payload["contract_type"] = "unknown"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    request["source_model"]["model_manifest_sha256"] = _sha256(manifest)
    request["initialization"]["model_manifest_sha256"] = _sha256(manifest)
    _save(request, path)
    with pytest.raises(RasScenarioWorkerError, match="model-inventory"):
        RasScenarioWorker.verify_forecast_initialization(path)


def test_clone_initialization_change_never_executes(tmp_path, monkeypatch):
    request, path, result = _forecast_request(tmp_path)

    def change(workspace):
        workspace.unsteady_file.write_text(
            workspace.unsteady_file.read_text().replace("10.25", "11.25"), encoding="utf-8",
        )

    calls = _install_pipeline(monkeypatch, request, change_clone=change)
    assert RasScenarioWorker.run(path, result) == 3
    assert calls["execute"] == 0
    assert "fingerprint" in json.loads(result.read_text())["error"]["message"]


@pytest.mark.parametrize("replacement", ["SI Units", "English Units\nSI Units", ""])
def test_clone_unit_change_never_executes(tmp_path, monkeypatch, replacement):
    request, path, result = _forecast_request(tmp_path)

    def change(workspace):
        workspace.project_file.write_text(
            workspace.project_file.read_text().replace("English Units", replacement), encoding="utf-8",
        )

    calls = _install_pipeline(monkeypatch, request, change_clone=change)
    assert RasScenarioWorker.run(path, result) == 3
    assert calls["execute"] == 0


@pytest.mark.parametrize("newline", ["\r\n", "\n"], ids=["native-crlf", "lf-refused"])
def test_real_preparation_preserves_static_source_without_engine(tmp_path, monkeypatch, newline):
    request, path, result = _forecast_request(tmp_path)
    source = Path(request["source_model"]["project"]).parent
    if newline == "\n":
        for filename in ("Model.prj", "Model.p01", "Model.g01", "Model.u01"):
            native_file = source / filename
            native_file.write_bytes(native_file.read_bytes().replace(b"\r\n", b"\n"))
        request["source_model"]["project_file_sha256"] = _sha256(source / "Model.prj")
        for field, suffix in (("plan", ".p01"), ("geometry", ".g01"), ("unsteady", ".u01")):
            request["initialization"][field + "_sha256"] = _sha256(source / ("Model" + suffix))
        _save(request, path)
    original_prepare = RasScenario.prepare_workspace
    original_validate = RasScenario.validate_workspace
    original_inspect = RasScenario.inspect_workspace_evidence
    calls = _install_pipeline(monkeypatch, request)
    monkeypatch.setattr(RasScenario, "prepare_workspace", original_prepare)
    monkeypatch.setattr(RasScenario, "validate_workspace", original_validate)
    monkeypatch.setattr(RasScenario, "inspect_workspace_evidence", original_inspect)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    exit_code = RasScenarioWorker.run(path, result)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    if newline == "\n":
        # Some preparation writers retain LF and others emit CRLF. The existing
        # gate must refuse that combination instead of accepting a partial edit.
        assert exit_code == 4
        assert calls["execute"] == 0
        assert "one_newline_convention" in json.loads(result.read_text())["error"]["message"]
        return
    assert exit_code == 0
    assert calls["execute"] == 1
    evidence = json.loads(result.read_text())["preparation"]
    assert all(evidence["checks"].values())
    initialization = evidence["forecast_initialization"]
    assert initialization["clone"]["initialization"]["sha256"] == initialization["initialization"]["sha256"]
    assert initialization["status"] == "clone-prepared"
    assert initialization["clone"]["initialization"]["controls"]["Prior WS Filename"] is None


@pytest.mark.parametrize("mutation", ["mapping", "mapping_type", "inactive_type", "flow_type", "receipt"])
def test_real_preparation_reuses_completed_workspace_and_refuses_changed_evidence(
    tmp_path, monkeypatch, mutation,
):
    request, path, result = _forecast_request(tmp_path)
    source = Path(request["source_model"]["project"]).parent
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    original_prepare = RasScenario.prepare_workspace
    original_validate = RasScenario.validate_workspace
    original_inspect = RasScenario.inspect_workspace_evidence
    calls = _install_pipeline(monkeypatch, request)
    monkeypatch.setattr(RasScenario, "prepare_workspace", original_prepare)
    monkeypatch.setattr(RasScenario, "validate_workspace", original_validate)
    monkeypatch.setattr(RasScenario, "inspect_workspace_evidence", original_inspect)

    assert RasScenarioWorker.run(path, result) == 0
    completed = result.read_bytes()
    assert RasScenarioWorker.run(path, result) == 0
    assert result.read_bytes() == completed
    assert calls["execute"] == calls["export"] == 1
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before

    retained = json.loads(completed)
    workspace = retained["preparation"]["workspace"]
    if mutation == "mapping":
        workspace["boundary_mapping_ids"] = ["different-boundary"]
    elif mutation == "receipt":
        retained["preparation"]["forecast_initialization"]["forecast_eligible"] = 0
    else:
        field = {
            "mapping_type": "boundary_mapping_ids",
            "inactive_type": "inactive_inherited_boundaries",
            "flow_type": "boundary_flow_preparation",
        }[mutation]
        workspace[field] = {}
    result.write_text(json.dumps(retained), encoding="utf-8")
    changed = result.read_bytes()
    assert RasScenarioWorker.run(path, result) == 3
    assert result.read_bytes() == changed
    assert calls["execute"] == calls["export"] == 1
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


def test_request_schema_is_packaged_and_legacy_rejects_new_fields(tmp_path):
    request, _, _ = _forecast_request(tmp_path)
    contracts = Path(importlib.import_module("ras_commander").__file__).parent / "contracts"
    schema = json.loads((contracts / "scenario-worker-request-v1.3.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(request, schema)
    for old_schema in (RasScenarioWorker.LEGACY_REQUEST_SCHEMA, RasScenarioWorker.REQUEST_SCHEMA, RasScenarioWorker.CRS_REQUEST_SCHEMA):
        request["schema"] = old_schema
        with pytest.raises(RasScenarioWorkerError):
            RasScenarioWorker._validate_request(request)


@pytest.mark.parametrize("declaration", ["SI Units", "SI Units=1", "SI Units=unknown", "si units"])
def test_conflicting_or_unknown_native_units_refused(tmp_path, declaration):
    request, path, _ = _forecast_request(tmp_path)
    project = Path(request["source_model"]["project"])
    project.write_text(project.read_text() + declaration + "\n", encoding="utf-8")
    request["source_model"]["project_file_sha256"] = _sha256(project)
    _save(request, path)
    with pytest.raises(RasScenarioWorkerError, match="[Uu]nit"):
        RasScenarioWorker.verify_forecast_initialization(path)


def test_missing_units_remain_unreviewed_for_unchanged_state(tmp_path):
    request, path, _ = _forecast_request(tmp_path)
    project = Path(request["source_model"]["project"])
    project.write_text(project.read_text().replace("English Units\n", ""), encoding="utf-8")
    request["source_model"]["project_file_sha256"] = _sha256(project)
    _save(request, path)
    evidence = RasScenarioWorker.verify_forecast_initialization(path)
    assert evidence["source_project_units"] is None
    assert evidence["vertical_datum"] == "unreviewed"


@pytest.mark.parametrize("escape", ["canonical_workspace", "canonical_unsteady", "wrong_current_plan", "wrong_window"])
def test_public_preparation_refuses_unsafe_clone_before_edit(tmp_path, monkeypatch, escape):
    request, path, _ = _forecast_request(tmp_path)
    _install_pipeline(monkeypatch, request)
    workspace = RasScenario.prepare_workspace()
    source = Path(request["source_model"]["project"])
    if escape == "canonical_workspace":
        request["workspace"] = str(source.parent)
        workspace = replace(workspace, project_folder=source.parent, project_file=source,
                            plan_file=source.with_suffix(".p01"), unsteady_file=source.with_suffix(".u01"),
                            plan_number="01", unsteady_number="01")
    elif escape == "canonical_unsteady":
        workspace = replace(workspace, unsteady_file=source.with_suffix(".u01"))
    elif escape == "wrong_current_plan":
        workspace.project_file.write_text(workspace.project_file.read_text().replace("Current Plan=p02", "Current Plan=p01"))
    else:
        workspace.plan_file.write_text(workspace.plan_file.read_text().replace("18SEP2019,1300", "18SEP2019,1200"))
    _save(request, path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(RasScenarioWorkerError):
        RasScenarioWorker.prepare_forecast_initialization(path, workspace)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_public_prepare_then_read_only_verify(tmp_path, monkeypatch):
    request, path, _ = _forecast_request(tmp_path)
    _install_pipeline(monkeypatch, request)
    workspace = RasScenario.prepare_workspace()
    source = Path(request["source_model"]["project"]).with_suffix(".u01")
    before = source.read_bytes()
    prepared = RasScenarioWorker.prepare_forecast_initialization(path, workspace)
    assert prepared["status"] == "clone-prepared"
    assert source.read_bytes() == before
    assert RasScenarioWorker.verify_forecast_initialization(path, workspace=workspace) == prepared
    plan_bytes = workspace.plan_file.read_bytes()
    _set_clone_cores(RasScenarioWorker._validate_request(request), workspace)
    assert workspace.plan_file.read_bytes() == plan_bytes
    assert RasScenarioWorker.verify_forecast_initialization(path, workspace=workspace) == prepared


@pytest.mark.parametrize("change_kind", ["numerical", "bare_mode"])
def test_changed_clone_numerical_settings_refused(tmp_path, monkeypatch, change_kind):
    request, path, result = _forecast_request(tmp_path)

    def change(workspace):
        plan_text = workspace.plan_file.read_text()
        if change_kind == "numerical":
            plan_text += "UNET Warmup Time Steps=100\n"
        else:
            plan_text = plan_text.replace("Unsteady Flow", "Steady Flow")
        workspace.plan_file.write_text(plan_text)

    calls = _install_pipeline(monkeypatch, request, change_clone=change)
    assert RasScenarioWorker.run(path, result) == 3
    assert calls["execute"] == 0


def test_forecast_keeps_crs_preflight_before_execution(tmp_path, monkeypatch):
    request, path, result = _forecast_request(tmp_path)
    calls = _install_pipeline(monkeypatch, request)
    monkeypatch.setattr(RasScenario, "inspect_product_crs", lambda *a, **k: {
        "crs_resolution": {"export_allowed": False, "reason": "unresolved CRS"},
    })
    assert RasScenarioWorker.run(path, result) == 3
    assert calls["execute"] == 0
    assert json.loads(result.read_text())["error"]["classification"] == "product_crs_preflight_failed"


@pytest.mark.parametrize("change", [None, "initialization", "execution", "receipt_schema"])
def test_preparation_retry_binds_forecast_identity(tmp_path, change):
    request, path, result = _forecast_request(tmp_path)
    request["source_model"]["linked_asset_mode"] = "shared-cache"
    _save(request, path)
    worker_module = importlib.import_module("ras_commander.RasScenarioWorker")
    result.write_text(json.dumps({
        "schema": RasScenarioWorker.RESULT_SCHEMA, "status": "failed",
        "request": {"schema": request["schema"], "sha256": worker_module._json_sha256(request)},
        "preparation": {"status": "in_progress"}, "execution": {"status": "not_started"},
        "error": {"classification": "preparation_failed"},
    }), encoding="utf-8")
    Path(request["workspace"]).mkdir()
    request["workspace"] = str(tmp_path / "retry-workspace")
    request["products"]["directory"] = str(tmp_path / "retry-products")
    request["retry"] = {"prior_request": str(path), "prior_result": str(result)}
    if change == "initialization":
        request["initialization"]["plan_sha256"] = "8" * 64
    elif change == "execution":
        request["execution"]["cores"] = 1
    elif change == "receipt_schema":
        receipt = json.loads(result.read_text())
        receipt["request"]["schema"] = RasScenarioWorker.CRS_REQUEST_SCHEMA
        result.write_text(json.dumps(receipt), encoding="utf-8")
    normalized = RasScenarioWorker._validate_request(request)
    if change is None:
        assert RasScenarioWorker._verify_retry_evidence(normalized)["status"] == "authenticated"
    else:
        with pytest.raises(RasScenarioWorkerError) as raised:
            RasScenarioWorker._verify_retry_evidence(normalized)
        assert raised.value.classification == "retry_evidence"
