# Scenario Worker

## Upstream integration compatibility

Scenario preparation removes inactive inherited DSS boundaries through an
owned editing stage. The stage inventories execution assets but always records
`execution_readiness="not_ready"`; it cannot establish compute readiness.
Project, plan, geometry-text and flow-text references, copied bytes, source
identity and exact boundary checksums remain required. Selected removals use
one atomic transaction. The preparer verifies the original clone before
publishing the edited unsteady file and then runs workspace validation.

When inactive boundaries exist, allow disk space for one additional temporary
project copy. Successful preparation keeps `boundary-edit-evidence.json` with
file counts, sizes, checksums and mutation evidence. Failed operations can
retain an isolated `.boundary-edit-*` directory for diagnosis; they do not
authorize execution.

Materialized flow inputs retain exact-once QMult handling and newline
preservation. DSS grid copying accepts a zero-length tail for copy/translation
workflows. Long translations and hydraulic product exports retain Year 3000
timestamps. Existing hydraulic product keys and hydrograph columns remain
stable. Upstream square raster grids and physical-cell filtering can change
pixels relative to older exports; compare and accept derivatives before
replacing published products. Source HDF files remain read only.

Hydraulic product manifest version `2.0` restores full coupled model coverage:
`result-footprint.geojson` unions the 1D river, 2D flow-area and storage-area
polygons, while `raster-support.geojson` retains the per-mesh 2D polygons.
Both GeoJSON assets use WGS84; their `proj:bbox` and `proj:code` describe the
source projected coordinates. The full domain is not an inundation extent.
Missing or invalid declared 1D/storage geometry prevents product publication.

The manifest declares `spatial.support="coupled_model_domain"`, binds the
footprint and raster-support asset keys separately, and gives each raster a
`support_asset="raster-support"` reference. `spatial.bbox` covers the full
domain; `spatial.raster_bbox` covers the meshes. Raster sampling and masks remain
2D-only. Hydrograph coverage does not expand raster support. Consumers must
accept and validate version 2 before adopting this exporter. Historical version
1 packages remain immutable and must not be relabeled as version 2. Worker
request/result envelope versions are unchanged; the product receipt carries the
exporter's independent manifest version.

The historical direct-path `RasUnsteady.delete_boundary()` interface is rejected;
use an owned stage and exact inventory evidence. Land-cover edits require the
native writer's explicit HEC-RAS version or initialized project context. Legacy
study scripts that omit this context need migration before using the candidate.
Native Manning edits verify class identities and unrequested parameters after
saving; a mismatch restores the sidecar and native backup. This postcondition
does not reinstate the legacy raw-HDF writer or qualify a new installed runtime.
Model execution and engineering acceptance remain separate gates.

## Worker contract

`ras-scenario-worker` is the package-owned process boundary for one isolated
HEC-RAS scenario. It lets an orchestrator invoke RAS Commander from a dedicated
RAS environment without importing RAS Commander into the orchestrator.

The versioned request binds the scenario to:

- the source project checksum, full model-manifest identity, and exact template
  plan;
- the HMS result DSS and hydrologic product-manifest identities;
- every exact DSS boundary selector;
- the run-local gridded `EXCESS` DSS and provenance identity;
- the local model window, RAS executable, core count, and timeout; and
- new workspace and hydraulic-product destinations.

Run a request with either entry point:

```powershell
ras-scenario-worker --request request.json --result result.json

python -m ras_commander.RasScenarioWorker `
  --request request.json `
  --result result.json
```

The worker calls `RasScenario.prepare_workspace()`, `RasScenario.execute()`,
and `HdfResultsProducts.export()`. The original model is not mutated. A
successful result includes current-plan, simulation-window, newline,
geometry-crosswalk, DSS-link, HDF-completion, time-axis, product, numerical
QA/QC, and timing evidence.

Newline evidence covers authored project, plan, geometry, and flow text files;
generated `.b##`, `.c##`, `.o##`, `.r##`, and `.x##` compute artifacts are
excluded because they may be binary.

Scenario clones omit source-project generated outputs: the project-named DSS,
plan and temporary plan HDF files, and RAS Mapper `PostProcessing.hdf` caches.
Geometry HDF inputs remain in the clone. The preparation result records every
excluded path and byte count so the reduced clone is auditable.

Mechanical execution and hydraulic acceptance remain separate. Compute-message
findings such as ignored boundaries, precipitation coverage, maximum
iterations, water-surface error, and volume accounting are returned under
`conditional_findings`; their presence does not rewrite a mechanically
successful execution as a failure.

## Repeat and failure behavior

### Provisional forecast initialization (request 1.3)

Request `ras-commander/scenario-worker-request/1.3`, named by
`RasScenarioWorker.FORECAST_REQUEST_SCHEMA`, adds an explicit initialization
binding to the 1.2 CRS contract. Existing request 1.0/1.1/1.2 and result 1.0
schemas remain unchanged. The result's existing preparation object carries
`forecast_initialization` evidence; no historical result is relabeled.

The first supported method preserves source-declared static IC values in a
run clone. It requires these initialization fields:

| Field | Required value or identity |
| --- | --- |
| `method` | `static` |
| `disposition` | `provisional-static` |
| `reference_policy` | `clear-in-clone` |
| `anchor_time` | UTC instant ending in `Z`, equal to model-window start |
| `predecessor` | `null`; no prior forecast state is reused |
| `package_sha256` | Independently selected RAS model package digest |
| `model_manifest_sha256` | Pinned source model manifest digest |
| `plan_sha256`, `geometry_sha256`, `unsteady_sha256` | Actual selected source files |
| `engine_sha256` | Actual selected executable bytes |
| `hydrology_sha256`, `forcing_sha256` | Exact request DSS input digests |

Request 1.3 also requires `source_model.package_sha256`,
`model_window.time_zone="UTC"` and the 1.2 `products.crs_policy`.
Native model-window strings remain timezone-free but now represent UTC and
must be minute-aligned, matching the precision of native plan text. The anchor
declares applicability for this run; it does not establish an observed state
at that instant.

The pinned model manifest must have
`contract_type="floodforecast/model-inventory/2.0.0"`, `stage="ras"` and
`content_summary.content_tree_sha256` equal to the selected RAS package digest.
The worker matches this declaration and independently hashes the selected
project, linked plan/geometry/unsteady files, engine and forcing. The
orchestrator remains responsible for authenticating the complete package tree.
A manifest declaration is not evidence that the worker verified every asset.
Source verification alone does not establish clone-preparation readiness.
Current full preparation requires source text already normalized to native
CRLF line endings: some existing writers preserve source endings while others
emit CRLF. An LF source can therefore fail the unchanged newline gate during
cloning. Use an authenticated normalized source successor; this initialization
API does not silently normalize or rewrite the canonical source.

Source admission requires one explicit `Use Restart=0` or `False` selector,
unambiguous controls and completely parsed finite IC records. Duplicate or
unknown controls, malformed records and contradictory native unit declarations
refuse admission. Named IC elevations must match exactly one IC point in the
selected geometry. No point, flow or elevation is assigned or converted.
Missing scientific units/datum remain unreviewed for this unchanged-state
method; the clone must preserve the source's known or unknown native units.

Some delivered sources retain populated restart and prior-results references
alongside static values. The worker does not infer that those references are
inactive. The explicit `clear-in-clone` policy uses
`RasUnsteady.set_initial_flow_method(..., "initial_flow_distribution")` to
select static mode and remove those references only in the authenticated run
clone. It records the source controls and requested transformation and checks
the exact physical IC-record fingerprint afterward. Source files remain
unchanged. Active restart, prior-results initialization and provider-derived
values are unsupported in this version. The native interpretation of the
original inherited references remains unqualified. HEC documents static,
restart and prior-results methods separately in its
[6.6 initial-conditions manual](https://www.hec.usace.army.mil/confluence/rasdocs/r2dum/6.6/boundary-and-initial-conditions-for-2d-flow-areas/initial-conditions).

Two APIs support orchestration without compute:

```python
from ras_commander import RasScenarioWorker

# Authenticate only; creates no clone and changes no files.
source = RasScenarioWorker.verify_forecast_initialization("request.json")
assert source["status"] == "source-verified"

# `workspace` is the isolated result of RasScenario.prepare_workspace().
prepared = RasScenarioWorker.prepare_forecast_initialization("request.json", workspace)
assert prepared["status"] == "clone-prepared"

# Recheck a prepared clone without changing it.
verified = RasScenarioWorker.verify_forecast_initialization(
    "request.json", workspace=workspace,
)
```

Preparation refuses canonical-source overlap and paths outside the requested
clone. It retains CurrentPlan, window, boundary, newline and CRS gates, preserves
other plan settings, and sets the requested cores through `RasPlan` before
recording plan hashes. Reapplying the execution core setting leaves those bytes
stable. A second preparation call refuses changed source controls; use the
read-only verifier on an already prepared clone.

Evidence is labeled `unreviewed-source-declaration` with
`forecast_eligible=false`, whether source-verified or clone-prepared. Execution
success cannot promote that state. Completed-result reuse reauthenticates
inputs, the clone and the retained initialization evidence, including JSON
types, before returning success. A preparation retry must preserve initialization,
package, execution and CRS settings. Initialization failures return
`forecast_initialization_invalid`, exit 3, with execution not started;
malformed request bindings return `invalid_request`, exit 2.

Portable tests use synthetic native text and real clone preparation, with
execution/export replaced. Installed-engine qualification, native restart
layout support, observed/calibrated state and physical forecast validation
remain separate follow-up work.
Native qualification must also establish that execution retains the prepared
project/plan/unsteady text hashes used by completed-result reuse. Any legitimate
engine rewrite needs a separately reviewed normalization contract; this version
keeps exact retained-file checks and does not assume post-execution stability.

### Explicit product CRS policy (request 1.2)

Request `ras-commander/scenario-worker-request/1.2` requires
`products.crs_policy`, either `strict` or `result_geometry`. Other product
options retain their existing defaults. Versions 1.0 and 1.1 remain unchanged
and reject this new field; the compatibility constant `REQUEST_SCHEMA` remains
1.1, while `CRS_REQUEST_SCHEMA` names 1.2. The result envelope stays 1.0.

For 1.2, the worker calls `RasScenario.inspect_product_crs()` after workspace
validation and before `RasScenario.execute()`. The helper resolves the selected
plan's geometry through project metadata and records its HDF checksum plus
`HdfResultsProducts.inspect_export_crs()` evidence. Missing or unresolved
evidence produces `product_crs_preflight_failed`, exit 3, non-retryable, with
execution `not_started`. The clone and failed result remain available.

`strict` requires consistent project CRS evidence. `result_geometry` requires
matching projected geometry/project evidence and readable referenced CRSs;
terrain disagreement remains explicit. The worker passes the same policy to
product export, which independently checks the actual result HDF and associated
CRS evidence. Preflight readiness does not certify result completeness, datum
acceptance or hydraulic accuracy. Neither policy transforms the source model.

Bind this policy and other product options in the orchestrator's result-defining
specification. A preparation retry involving 1.2 must retain its schema and all
product options except the new output directory. A changed policy requires a
new specification; it must not relabel or rewrite an earlier attempt. Versions
1.0/1.1 retain their existing retry semantics and do not acquire the new geometry
preflight. This worker does not recover an already-computed failed attempt;
use the package exporter on the retained HDF in a separate recovery workflow.

### Existing destination and terminal checks

A verified successful result for the identical normalized request is returned
without rerunning RAS. The worker refuses an existing workspace, product
directory, conflicting result, or changed completed output. Timeout and failed
attempt workspaces are retained for diagnosis and GUI review, and failure
results are written atomically when possible.

The execution timeout is passed to `RasCmdr.compute_plan(max_runtime=...)`.
It covers engine execution and asynchronous solver completion; preparation,
cancellation and result finalization can take additional time. Scenario execution
waits for that package-owned cleanup and retains its `execution_details` in the
result. There is no detached scenario executor thread after a timeout.

A failed execution is retryable only when
`execution.execution_details.solver_quiescence_confirmed` is exactly `true`.
The cancellation record distinguishes stopped identities, surviving processes
and inspection uncertainty. A timeout exception without this evidence is
non-retryable. When quiescence is unconfirmed, the scenario also withholds HDF
inspection and product export. Preserve the failed workspace; use
`RasCmdr.inspect_plan_processes()` and `RasCmdr.cancel_plan_exact()` with the
failed clone's initialized project context to inspect or stop only that plan.
Retain the resulting evidence before an operator authorizes a fresh attempt.

Abrupt worker/coordinator termination is outside this in-process cleanup path.
The calling supervisor must contain owned descendants and fence retries after
interruption. Portable tests use harmless Python descendants and synthetic
process metadata; installed-engine interruption still requires qualification
on disposable model clones. Process quiescence does not establish hydraulic
acceptance.

## Shared linked-asset cache and preparation retry

Large immutable linked directories such as `Terrain`, `Land Cover`, and
`Hydrologic Input` do not need a private copy for every scenario. Set
`source_model.linked_asset_mode` to `shared-cache` and place scenario workspace
directories directly below one stable cache root. The first request copies or
adopts the linked directories beside the workspace, content-compares them with
their declared sources, records an identity manifest, and marks cached files
read-only. Later sibling workspaces verify the cache metadata and reuse it
without copying the large assets again.

```json
{
  "source_model": {
    "linked_asset_mode": "shared-cache",
    "linked_asset_directories": [
      "C:/models/release/Terrain",
      "C:/models/release/Land Cover"
    ]
  },
  "workspace": "C:/ras-cache/model-manifest-sha/scenario-001"
}
```

The cache key is the request's verified RAS model-manifest SHA-256. A different
key, source path, file population, size, or modification time fails closed.
Cache adoption may read both large trees once to establish their content
identity, but it does not allocate another copy.

A corrected retry after a preparation-only failure may reference the preserved
prior request and result:

```json
{
  "retry": {
    "prior_request": "C:/evidence/failed-request.json",
    "prior_result": "C:/evidence/failed-result.json"
  }
}
```

The worker authenticates the prior request/result binding, requires execution
to remain `not_started`, preserves the failed workspace, and requires the new
workspace to be a sibling under the same cache root. Scenario, source-model,
hydrology, forcing-excess, and model-window identities cannot change; corrected
boundary selectors and execution options may change.

The packaged contracts are:

- `ras-commander/scenario-worker-request/1.1` (current)
- `ras-commander/scenario-worker-request/1.0` (historical compatibility)
- `ras-commander/scenario-worker-result/1.0`

## Materialized flow inputs

Request 1.1 requires `flow_multiplier_policy` on every boundary link. Use
`preserve-source` when the DSS contains original values and RAS must apply its
authored multiplier. Use `materialized` when upstream processing has already
applied the required conversion or split. For example, a half-flow DSS uses
`materialized` so the cloned RAS QMult is unity, giving half the original flow
instead of one quarter. The scientific split remains in upstream provenance.

The default Python `RasBoundaryLink`/`set_boundary_dss_link()` behavior remains
`preserve-source`. Request 1.0 preserves historical behavior and rejects the new
field. Old requests/results are not relabeled. Request 1.1 accepts both policies
explicitly; the worker does not infer a policy from a pathname or multiplier.

Only the exact selected flow boundary is changed. An absent QMult already means
unity and remains absent. `RasUnsteady.inspect_boundary_flow()` reports the
effective multiplier, its explicit/implicit source and the DSS binding.
Materialized preparation rejects ambiguous or conflicting selectors, duplicate
or nonfinite multipliers and unsupported modifiers such as minimum-flow clamps,
fixed start times or critical-boundary overrides. It preserves energy-grade
slope and other boundaries. Mutations preserve LF/CRLF and reject mixed input.

Preparation evidence includes `boundary_flow_preparation` before/after records
and final `materialized_flow_boundaries` readback. The
`materialized_flow_bindings_match` check requires unity multipliers, exact DSS
bindings and no unsupported modifiers. Checks account for audited removal of
inactive inherited records; no scientific split is applied again in RAS.

Request 1.1 also permits `forcing_excess.interpolation = "preserve-source"` to
retain the source gridded interpolation, including an unset method. This uses
the existing precipitation API; it does not choose an alternative interpolation
for a delivered model. Explicit `Nearest`/`Bilinear` and the historical default
remain available. Blank and absent source interpolation both remain unset;
the precipitation writer may omit the blank text key. Source models are never
modified by scenario preparation.

## Configure the installed-engine test

The installed-engine test is opt-in through the `integration` pytest marker.
Configure its inputs with environment variables in the shell that launches
pytest:

| Variable | Required | Value |
| --- | --- | --- |
| `RAS_WORKER_TEST_SOURCE_PROJECT` | Yes | Source `.prj` file or directory containing one project |
| `RAS_WORKER_TEST_SOURCE_PLAN` | Yes | Template plan number, such as `01` |
| `RAS_WORKER_TEST_MODEL_MANIFEST` | Yes | JSON model manifest whose `stage` is `ras` |
| `RAS_WORKER_TEST_HYDROLOGY_DSS` | Yes | HMS result DSS used by the boundary links |
| `RAS_WORKER_TEST_HYDROLOGY_MANIFEST` | Yes | `hms-commander/hydrologic-product-manifest/1.0` JSON that identifies the HMS DSS checksum |
| `RAS_WORKER_TEST_BOUNDARY_LINKS_JSON` | Yes | JSON array of worker boundary-link objects |
| `RAS_WORKER_TEST_EXCESS_DSS` | Yes | Run-local gridded excess DSS |
| `RAS_WORKER_TEST_EXCESS_MANIFEST` | Yes | JSON provenance manifest for the excess DSS |
| `RAS_WORKER_TEST_EXCESS_PATHNAME` | Yes | Six-part DSS pathname family, ending with `/` |
| `RAS_WORKER_TEST_EXECUTABLE` | Yes | Installed `Ras.exe` path |
| `RAS_WORKER_TEST_START` / `RAS_WORKER_TEST_END` | No | Naive local model timestamps; defaults cover the DeLoutre canary window |
| `RAS_WORKER_TEST_TIME_ZONE` | No | IANA time-zone identity; default `America/Chicago` |
| `RAS_WORKER_TEST_TIMEOUT_SECONDS` | No | Positive worker timeout; default `7200` |
| `RAS_WORKER_TEST_CORES` | No | Positive RAS core count; default `2` |
| `RAS_WORKER_TEST_WORKSPACE` | No | New, nonexistent workspace path |
| `RAS_WORKER_TEST_LINKED_ASSETS_JSON` | No | JSON array of immutable linked-asset directories |
| `RAS_WORKER_TEST_LINKED_ASSET_MODE` | No | `copy` or `shared-cache`; default `copy` |
| `RAS_WORKER_TEST_RETRY_REQUEST` / `RAS_WORKER_TEST_RETRY_RESULT` | No | Authenticated preparation-only retry evidence; configure both or neither |

For example:

```powershell
$env:RAS_WORKER_TEST_SOURCE_PROJECT = 'C:\models\release\Model.prj'
$env:RAS_WORKER_TEST_SOURCE_PLAN = '01'
$env:RAS_WORKER_TEST_MODEL_MANIFEST = 'C:\evidence\ras-model.json'
$env:RAS_WORKER_TEST_HYDROLOGY_DSS = 'C:\inputs\hms-output.dss'
$env:RAS_WORKER_TEST_HYDROLOGY_MANIFEST = 'C:\inputs\hydrologic-products.json'
$env:RAS_WORKER_TEST_BOUNDARY_LINKS_JSON = Get-Content -Raw '.\boundary-links.json'
$env:RAS_WORKER_TEST_EXCESS_DSS = 'C:\inputs\ras-excess.dss'
$env:RAS_WORKER_TEST_EXCESS_MANIFEST = 'C:\inputs\ras-excess.json'
$env:RAS_WORKER_TEST_EXCESS_PATHNAME = '/SHG/BASIN/PRECIPITATION///EXCESS/'
$env:RAS_WORKER_TEST_EXECUTABLE = 'C:\Program Files (x86)\HEC\HEC-RAS\6.6\Ras.exe'
$env:RAS_WORKER_TEST_WORKSPACE = 'C:\ras-cache\model-hash\canary-001'
$env:RAS_WORKER_TEST_LINKED_ASSETS_JSON = '["C:/models/release/Terrain"]'
$env:RAS_WORKER_TEST_LINKED_ASSET_MODE = 'shared-cache'

python -m pytest tests\test_scenario_worker.py `
  -m integration `
  -k installed_engine_executes_one_scenario `
  --basetemp .\working\pytest-scenario-worker-installed
```

Use a fresh `--basetemp` and workspace for each new attempt. A shared cache
can adopt an existing directory only when its content matches the declared
canonical source; otherwise the worker fails closed before RAS execution.
