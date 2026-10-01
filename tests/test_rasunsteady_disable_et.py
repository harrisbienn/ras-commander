"""Explicit ET disabling preserves rainfall, inactive definitions and text format."""

from pathlib import Path

import pytest

from ras_commander import RasUnsteady


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("modes", [[], ["Point Gage"], ["Constant", "Gridded"]])
def test_disable_et_preserves_other_inputs(tmp_path, newline, modes):
    path = tmp_path / "control.u01"
    other = ["Flow Title=Control", "Precipitation Mode=Enable",
             "Met BC=Precipitation|Mode=Gridded",
             "Met BC=Evapotranspiration|Constant Value=2",
             "Met BC=Evapotranspiration|Constant Units=mm/day"]
    lines = other[:2] + [f"Met BC=Evapotranspiration|Mode={m}" for m in modes] + other[2:]
    path.write_bytes((newline.join(lines) + newline).encode())
    RasUnsteady.disable_evapotranspiration(path)
    actual = path.read_bytes()
    assert actual.splitlines().count(b"Met BC=Evapotranspiration|Mode=None") == 1
    assert [x.decode() for x in actual.splitlines() if not x.startswith(b"Met BC=Evapotranspiration|Mode=")] == other
    assert actual.count(b"\r\n") == (len(other) + 1 if newline == "\r\n" else 0)
    RasUnsteady.disable_evapotranspiration(path)
    assert path.read_bytes() == actual


def test_disable_et_rejects_mixed_newlines_without_changing_file(tmp_path):
    path = tmp_path / "control.u01"
    original = b"Flow Title=Control\r\nPrecipitation Mode=Enable\n"
    path.write_bytes(original)
    with pytest.raises(ValueError, match="[Mm]ixed"):
        RasUnsteady.disable_evapotranspiration(path)
    assert path.read_bytes() == original


def test_disable_et_preserves_real_fixture(tmp_path):
    fixtures = Path(__file__).resolve().parents[1] / "research/fixtures/met_data"
    source = fixtures / "precip_00_disabled_official_davis.u01"
    path = tmp_path / "fixture.u01"
    path.write_bytes(source.read_bytes())
    before = path.read_bytes().splitlines()
    RasUnsteady.disable_evapotranspiration(path)
    def strip_mode(lines):
        return [x for x in lines if not x.startswith(b"Met BC=Evapotranspiration|Mode=")]

    assert strip_mode(path.read_bytes().splitlines()) == strip_mode(before)
