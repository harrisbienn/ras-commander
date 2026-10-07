"""Opt-in IC selector round trips on public HEC-RAS 6.6 examples; no compute."""

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from ras_commander import RasExamples, RasPrj, RasUnsteady, RasUtils, init_ras_project

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RAS_COMMANDER_RUN_HECRAS_INTEGRATION") != "1",
        reason="Set RAS_COMMANDER_RUN_HECRAS_INTEGRATION=1 for public-example preparation checks",
    ),
]


@pytest.mark.parametrize("example,project_file,fractional_stations", [
    ("Balde Eagle Creek", "BaldEagle.prj", {138154.4}),
    ("Storage Area Hydraulic Connection", "Beaver_LS_SAConn.prj", {5.99}),
    ("NavigationDam", "ROCK_TEST.prj", {301.18, 273.49}),
])
def test_public_example_all_ic_rows_roundtrip(tmp_path, example, project_file, fractional_stations):
    folder = RasExamples.extract_project(example, output_path=tmp_path / "example")
    project = init_ras_project(
        folder / project_file, ras_version="6.6", ras_object=RasPrj(),
        load_results_summary=False, hide_intro=True,
    )
    seen = set()
    for row in project.unsteady_df.itertuples():
        source = Path(row.full_path)
        original = source.read_bytes()
        before = RasUnsteady.get_initial_conditions(source)
        assert not before.empty
        seen.update(before.station.dropna())
        newline = RasUtils._detect_text_newline(source)
        copy = tmp_path / source.name
        shutil.copy2(source, copy)
        for target, context in ((copy, None), (row.unsteady_number, project)):
            RasUnsteady.set_initial_conditions(target, before, ras_object=context)
            prepared = copy if context is None else source
            after = RasUnsteady.get_initial_conditions(prepared)
            pd.testing.assert_frame_equal(before, after, check_exact=True)
            assert RasUtils._detect_text_newline(prepared) == newline
            assert prepared.read_bytes().split(b"Boundary Location=", 1)[1] == (
                original.split(b"Boundary Location=", 1)[1]
            )
            if context is None:
                assert source.read_bytes() == original
    assert fractional_stations <= seen
