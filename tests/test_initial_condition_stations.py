"""IC station selectors must survive flow/RRR serialization without rounding."""

import pytest

from ras_commander import RasUnsteady, RasUtils
from ras_commander.usgs.initial_conditions import InitialConditions


@pytest.mark.parametrize("ic_type", ["flow", "rrr"])
@pytest.mark.parametrize("station,text", [
    (5.99, "5.99"), (138154.4, "138154.4"), (301.18, "301.18"),
    (714.4, "714.4"), (-5.99, "-5.99"), (0.0000001, "0.0000001"),
    (1.2345678901234567, "1.2345678901234567"),
    (137520, "137520"), (137520.0, "137520"), (0, "0"),
])
def test_create_line_preserves_station_value_and_plain_decimal_text(ic_type, station, text):
    line = InitialConditions.create_ic_line(
        ic_type, river="Example River", reach="Example Reach", station=station, value=2000,
    )
    fields = line.split("=", 1)[1].split(",")
    assert fields[2] == text.ljust(8)
    assert float(fields[2]) == station
    assert fields[3] == "2000"


@pytest.mark.parametrize("ic_type", ["flow", "rrr"])
@pytest.mark.parametrize("station", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_station_is_rejected(ic_type, station):
    with pytest.raises(ValueError, match="station.*finite"):
        InitialConditions.create_ic_line(
            ic_type, river="Example River", reach="Example Reach", station=station, value=2000,
        )


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("wrapper", [False, True])
def test_all_row_roundtrip_preserves_fractional_selectors(tmp_path, newline, wrapper):
    # Actual record forms/values from the public HEC-RAS 6.6 examples.
    path = tmp_path / "Example.u01"
    path.write_bytes(newline.join([
        "Flow Title=Example", "Program Version=6.60", "Use Restart=0",
        "Initial Flow Loc=Beaver Creek    ,Kentwood        ,5.99    ,5000",
        "Initial Flow Loc=Bald Eagle      ,Loc Hav         ,138154.4,2000",
        "Initial RRR Elev=MISSISSIPPI     ,REACH # 17      ,301.18  ,459.3",
        "Initial RRR Elev=MISSISSIPPI     ,REACH # 17      ,273.49  ,448.5",
        "Initial Storage Elev=Bayou,206", "IC Point Elev=Reference Point,11,-1",
        "Boundary Location=Downstream", "Friction Slope=0.001", "",
    ]).encode())
    original = path.read_bytes()
    before = RasUnsteady.get_initial_conditions(path)
    if wrapper:
        RasUnsteady.set_initial_conditions(path, before)
    else:
        InitialConditions.write_initial_conditions(path, before.to_dict("records"))
    after = RasUnsteady.get_initial_conditions(path)
    assert before.equals(after)
    assert RasUtils._detect_text_newline(path) == newline
    assert path.read_bytes().split(b"Boundary Location=", 1)[1] == original.split(b"Boundary Location=", 1)[1]
    assert RasUnsteady.get_initial_point_elevations(path).elevation.tolist() == [11.0]


@pytest.mark.parametrize("wrapper", [False, True])
def test_invalid_station_leaves_original_file_unchanged(tmp_path, wrapper):
    path = tmp_path / "Example.u01"
    path.write_bytes(b"Flow Title=Example\r\nProgram Version=6.60\r\nUse Restart=0\r\n")
    original = path.read_bytes()
    entries = [
        {"type": "flow", "river": "River", "reach": "Reach", "station": 5.99, "value": 100},
        {"type": "rrr", "river": "River", "reach": "Reach", "station": float("nan"), "value": 50},
    ]
    writer = RasUnsteady.set_initial_conditions if wrapper else InitialConditions.write_initial_conditions
    with pytest.raises(ValueError, match="station.*finite"):
        writer(path, entries)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
