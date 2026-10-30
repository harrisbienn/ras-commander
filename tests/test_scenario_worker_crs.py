"""Versioned worker CRS policy: no solver dispatch is needed for these tests."""
import importlib
import json
from pathlib import Path

import h5py
import pytest
from pyproj import CRS

from ras_commander import HdfResultsProducts, RasScenarioWorker
from ras_commander.RasScenarioWorker import RasScenarioWorkerError
from test_scenario_worker import _install_success_fakes, _request, _write_prior_preparation_failure


def _crs_request(tmp_path, policy="strict"):
    request, request_path, result_path = _request(tmp_path)
    request["schema"] = "ras-commander/scenario-worker-request/1.2"
    request["products"]["crs_policy"] = policy
    request_path.write_text(json.dumps(request))
    return request, request_path, result_path


def _install_geometry(monkeypatch, request, *, terrain=3451, missing=False):
    module = importlib.import_module("ras_commander.RasScenarioWorker")
    calls = _install_success_fakes(monkeypatch, request)
    original = module.RasScenario.prepare_workspace
    original_export = HdfResultsProducts.export
    policies = []

    def prepare(*args, **kwargs):
        workspace = original(*args, **kwargs)
        folder = workspace.project_folder
        workspace.project_file.write_text("Proj Title=Fixture\nCurrent Plan=p02\nPlan File=p02\nGeom File=g01\n")
        workspace.plan_file.write_text("Plan Title=Fixture\nGeom File=g01\n")
        if not missing:
            with h5py.File(folder / "Model.g01.hdf", "w") as hdf:
                hdf.attrs["Projection"] = CRS.from_epsg(3451).to_wkt()
        (folder / "projection.prj").write_text(CRS.from_epsg(3451).to_wkt())
        with h5py.File(folder / "terrain.hdf", "w") as hdf:
            hdf.attrs["Projection"] = CRS.from_epsg(terrain).to_wkt()
        (folder / "Model.rasmap").write_text(
            '<RASMapper><RASProjectionFilename Filename="projection.prj" />'
            '<Terrains><Layer Filename="terrain.hdf" /></Terrains></RASMapper>'
        )
        return workspace

    def export(*args, **kwargs):
        policies.append(kwargs.get("crs_policy"))
        return original_export(*args, **kwargs)

    monkeypatch.setattr(module.RasScenario, "prepare_workspace", prepare)
    monkeypatch.setattr(HdfResultsProducts, "export", export)
    return calls, policies


@pytest.mark.parametrize("policy,terrain", [("strict", 3451), ("result_geometry", 6477)])
def test_worker_12_preflights_geometry_and_forwards_policy(tmp_path, monkeypatch, policy, terrain):
    request, request_path, result_path = _crs_request(tmp_path, policy)
    calls, policies = _install_geometry(monkeypatch, request, terrain=terrain)
    assert RasScenarioWorker.run(request_path, result_path) == 0
    result = json.loads(result_path.read_text())
    report = result["preparation"]["product_crs"]
    assert report["crs_resolution"]["policy"] == policy
    assert report["crs_resolution"]["export_allowed"] is True
    assert report["crs_resolution"]["terrain_conflict_retained"] == (terrain != 3451)
    assert len(report["geometry_hdf"]["sha256"]) == 64
    assert calls == {"prepare": 1, "execute": 1, "export": 1}
    assert policies == [policy]


@pytest.mark.parametrize("missing", [False, True])
def test_worker_12_blocks_compute_when_crs_is_unready(tmp_path, monkeypatch, missing):
    request, request_path, result_path = _crs_request(tmp_path)
    calls, policies = _install_geometry(monkeypatch, request, terrain=6477, missing=missing)
    assert RasScenarioWorker.run(request_path, result_path) == 3
    result = json.loads(result_path.read_text())
    assert result["error"]["classification"] == "product_crs_preflight_failed"
    assert result["error"]["retryable"] is False
    assert result["preparation"]["status"] == "failed"
    assert result["execution"]["status"] == "not_started"
    assert calls == {"prepare": 1, "execute": 0, "export": 0}
    assert policies == []
    assert not Path(request["products"]["directory"]).exists()
    if not missing:
        assert result["preparation"]["product_crs"]["crs_resolution"]["export_allowed"] is False


@pytest.mark.parametrize("policy", [None, "automatic", "EPSG:3451"])
def test_worker_12_requires_valid_explicit_policy(tmp_path, policy):
    request, _, _ = _crs_request(tmp_path)
    if policy is None:
        del request["products"]["crs_policy"]
    else:
        request["products"]["crs_policy"] = policy
    with pytest.raises(RasScenarioWorkerError, match="crs_policy"):
        RasScenarioWorker._validate_request(request)


def test_worker_11_remains_unchanged_and_rejects_12_option(tmp_path, monkeypatch):
    request, request_path, result_path = _request(tmp_path)
    calls = _install_success_fakes(monkeypatch, request)
    assert RasScenarioWorker.run(request_path, result_path) == 0
    assert "product_crs" not in json.loads(result_path.read_text())["preparation"]
    assert calls["execute"] == 1
    request["products"]["crs_policy"] = "result_geometry"
    with pytest.raises(RasScenarioWorkerError, match="unknown crs_policy"):
        RasScenarioWorker._validate_request(request)


def test_worker_12_schema_and_normalization(tmp_path):
    import importlib.resources
    import jsonschema

    request, _, _ = _crs_request(tmp_path, "result_geometry")
    path = importlib.resources.files("ras_commander") / "contracts/scenario-worker-request-v1.2.schema.json"
    schema = json.loads(path.read_text())
    jsonschema.validate(request, schema)
    normalized = RasScenarioWorker._validate_request(request)
    assert normalized["schema"] == schema["$id"]
    assert normalized["products"]["crs_policy"] == "result_geometry"
    request["forcing_excess"]["interpolation"] = "preserve-source"
    assert RasScenarioWorker._validate_request(request)["forcing_excess"]["interpolation"] == "preserve-source"


@pytest.mark.parametrize("change", ["policy", "schema", "resolution"])
def test_worker_12_retry_cannot_change_product_identity(tmp_path, change):
    request, request_path, result_path = _crs_request(tmp_path)
    request["source_model"]["linked_asset_mode"] = "shared-cache"
    _write_prior_preparation_failure(request, request_path, result_path)
    # The legacy fixture's result envelope must identify this retained request version.
    prior = json.loads(result_path.read_text())
    prior["request"]["schema"] = request["schema"]
    result_path.write_text(json.dumps(prior))
    request["retry"] = {"prior_request": str(request_path), "prior_result": str(result_path)}
    request["workspace"] = str(tmp_path / "retry-workspace")
    request["products"]["directory"] = str(tmp_path / "retry-products")
    if change == "policy":
        request["products"]["crs_policy"] = "result_geometry"
    elif change == "schema":
        request["schema"] = RasScenarioWorker.REQUEST_SCHEMA
        del request["products"]["crs_policy"]
    else:
        request["products"]["resolution"] = 10.0
    normalized = RasScenarioWorker._validate_request(request)
    with pytest.raises(RasScenarioWorkerError, match="Retry changed"):
        RasScenarioWorker._verify_retry_evidence(normalized)
