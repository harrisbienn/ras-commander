"""Byte-preserving edits of the native coupling setting, separate from 2D limits."""

import pytest

from ras_commander import RasPlan


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_only_coupling_limit_changes_and_repeated_call_is_idempotent(tmp_path, newline):
    path = tmp_path / "OL.p01"
    original = newline.join([b"Plan Title=Test", b"UNET D1D2 MaxIter= 0 ",
                             b"UNET D1D2 QTol=0.1", b"UNET D2 Max Iterations=20", b""])
    path.write_bytes(original)
    RasPlan.set_1d_2d_max_iterations(path, 3)
    expected = original.replace(b"MaxIter= 0 ", b"MaxIter= 3 ")
    assert path.read_bytes() == expected
    RasPlan.set_1d_2d_max_iterations(path, 3)
    assert path.read_bytes() == expected


@pytest.mark.parametrize("value", [-1, 21, 3.5, True, "3"])
def test_rejects_invalid_limits_without_writing(tmp_path, value):
    path = tmp_path / "OL.p01"
    path.write_bytes(b"UNET D1D2 MaxIter=0\n")
    with pytest.raises(ValueError, match="integer from 0 to 20"):
        RasPlan.set_1d_2d_max_iterations(path, value)
    assert path.read_bytes() == b"UNET D1D2 MaxIter=0\n"


@pytest.mark.parametrize("original", [
    b"Plan Title=Missing\n", b"UNET D1D2 MaxIter=0\nUNET D1D2 MaxIter=1\n",
    b"UNET D1D2 MaxIter=oops\n", b"Plan Title=Mixed\r\nUNET D1D2 MaxIter=0\n",
])
def test_rejects_ambiguous_or_malformed_plan_without_writing(tmp_path, original):
    path = tmp_path / "OL.p01"
    path.write_bytes(original)
    with pytest.raises(ValueError):
        RasPlan.set_1d_2d_max_iterations(path, 3)
    assert path.read_bytes() == original
