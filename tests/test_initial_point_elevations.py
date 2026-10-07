"""Named IC-point parsing is read-only and distinct from storage/flow ICs."""

import pytest

from ras_commander import RasUnsteady
from ras_commander.schemas import DATAFRAME_SCHEMAS


@pytest.mark.parametrize("newline", ["\r\n", "\n"])
def test_reads_named_points_without_altering_other_conditions(tmp_path, newline):
    path = tmp_path / "Example.u01"
    content = newline.join([
        "Flow Title=Example",
        "Initial Storage Elev=Floodplain,12",
        "IC Point Elev=Reference Point               ,10.25,-1",
        "IC Point Elev=Second Point,-2.5, raw field ,",
        "Boundary Location=River,Reach,100",
    ]).encode()
    path.write_bytes(content)
    points = RasUnsteady.get_initial_point_elevations(str(path))
    assert points.to_dict("records") == [
        {"point_name": "Reference Point", "elevation": 10.25, "trailing_fields": ("-1",)},
        {"point_name": "Second Point", "elevation": -2.5, "trailing_fields": (" raw field ", "")},
    ]
    assert RasUnsteady.get_initial_conditions(path)["type"].tolist() == ["storage"]
    assert path.read_bytes() == content
    assert len(list(tmp_path.iterdir())) == 1
    assert list(points.columns) == [c["name"] for c in DATAFRAME_SCHEMAS["initial_point_elevations"]["columns"]]


def test_absent_points_have_stable_columns_and_types(tmp_path):
    path = tmp_path / "Example.u01"
    path.write_bytes(b"Flow Title=Example\r\n")
    points = RasUnsteady.get_initial_point_elevations(path)
    assert points.empty
    assert list(points.columns) == ["point_name", "elevation", "trailing_fields"]
    assert str(points.elevation.dtype) == "float64"


@pytest.mark.parametrize("records", [
    "IC Point Elev=",
    "IC Point Elev=Point",
    "IC Point Elev= ,12,-1",
    "IC Point Elev=Point,,0",
    "IC Point Elev=Point,bad,-1",
    "IC Point Elev=Point,nan,-1",
    "IC Point Elev=Point,inf,-1",
    "IC Point Elev=Point,1,-1\nIC Point Elev=point,1,-1",
])
def test_invalid_or_ambiguous_points_fail_without_writes(tmp_path, records):
    path = tmp_path / "Example.u01"
    content = (records + "\n").encode()
    path.write_bytes(content)
    with pytest.raises(ValueError):
        RasUnsteady.get_initial_point_elevations(path)
    assert path.read_bytes() == content


def test_number_resolution_uses_explicit_project_context(tmp_path, monkeypatch):
    path = tmp_path / "Example.u03"
    path.write_bytes(b"IC Point Elev=Point,1,-1\r\n")
    context = object()

    def resolve(number, ras_object):
        assert number == "03"
        assert ras_object is context
        return path

    monkeypatch.setattr(RasUnsteady, "_resolve_unsteady_file_path", resolve)
    assert RasUnsteady.get_initial_point_elevations("03", ras_object=context).elevation.tolist() == [1.0]


def test_mixed_newlines_require_explicit_preparation(tmp_path):
    path = tmp_path / "Example.u01"
    path.write_bytes(b"Flow Title=Example\r\nIC Point Elev=Point,1,-1\n")
    with pytest.raises(ValueError, match="Mixed newline"):
        RasUnsteady.get_initial_point_elevations(path)
