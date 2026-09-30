"""HEC-DSS major-version creation coverage."""

import numpy as np
from datetime import datetime
import hashlib
import subprocess
import sys
import pandas as pd
import pytest

from ras_commander import RasDss


def _configure_dss_or_skip():
    try:
        RasDss._configure_jvm()
    except Exception as exc:
        pytest.skip(f"HEC-DSS Java bridge unavailable: {exc}")
    return RasDss


def test_write_timeseries_can_create_version_6_file(tmp_path):
    dss = _configure_dss_or_skip()
    output = tmp_path / "boundary-v6.dss"
    pathname = "/BASIN/UPSTREAM/FLOW//5MIN/QUALIFICATION/"
    times = pd.date_range("2019-09-18 13:00", periods=5, freq="5min")
    values = np.array([100.0, 110.0, 125.0, 120.0, 115.0])

    dss.write_timeseries(
        output,
        pathname,
        times,
        values,
        units="CFS",
        data_type="INST-VAL",
        dss_version=6,
    )

    assert dss.get_file_version(output) == 6
    reread = dss.read_timeseries(output, pathname)
    assert reread.index.equals(times.rename("datetime"))
    np.testing.assert_array_equal(reread["value"].to_numpy(), values)


@pytest.mark.parametrize("version", [0, 5, 8])
def test_write_timeseries_rejects_unsupported_version(tmp_path, version):
    dss = _configure_dss_or_skip()

    with pytest.raises(ValueError, match="dss_version"):
        dss.write_timeseries(
            tmp_path / "invalid.dss",
            "/BASIN/UPSTREAM/FLOW//5MIN/QUALIFICATION/",
            pd.date_range("2019-09-18", periods=2, freq="5min"),
            [1.0, 2.0],
            dss_version=version,
        )


def test_native_grid_conversion_preserves_year_3000_values_and_mask(tmp_path):
    dss = _configure_dss_or_skip()
    source = tmp_path / "grid7.dss"
    output = tmp_path / "grid6.dss"
    values = np.array([[[0, 1.25], [np.nan, 35.5]]], dtype=np.float32)
    script = """
from datetime import datetime
import sys
import numpy as np
from ras_commander import RasDss
RasDss.write_grid_timeseries(
    sys.argv[1], "/SHG/TEST/PRECIP///SYNTHETIC/",
    np.array([[[0, 1.25], [np.nan, 35.5]]], dtype=np.float32),
    [datetime(3000, 1, 2), datetime(3000, 1, 2, 1)],
    {"crs": "SHG", "cellsize": 1000, "origin": (1000, 2000),
     "units": "MM", "data_type": "PER-CUM"},
)
"""
    subprocess.run([sys.executable, "-I", "-c", script, str(source)], check=True)
    paths = ["/SHG/TEST/PRECIP/02JAN3000:0000/02JAN3000:0100/SYNTHETIC/"]
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    result = dss.convert_file_version(source, output, 6)
    assert result["record_count"] == 1 and result["output_version"] == 6
    observed = dss.read_grid(output, paths[0])
    np.testing.assert_array_equal(observed["data"], values[0])
    assert observed["start_time"].to_pydatetime() == datetime(3000, 1, 2)
    assert observed["end_time"].to_pydatetime() == datetime(3000, 1, 2, 1)
    assert observed["units"] == "MM" and observed["data_type"] == "PER-CUM"
    assert observed["cell_size"] == 1000
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with pytest.raises(FileExistsError):
        dss.convert_file_version(source, output, 6)
    with pytest.raises(ValueError, match="must differ"):
        dss.convert_file_version(source, source, 6)
    # Exercise the reverse conversion and same-version immutable copy too.
    roundtrip = tmp_path / "roundtrip7.dss"
    dss.convert_file_version(output, roundtrip, 7)
    np.testing.assert_array_equal(dss.read_grid(roundtrip, paths[0])["data"], values[0])
    same = tmp_path / "copy6.dss"
    dss.convert_file_version(output, same, 6)
    assert same.read_bytes() == output.read_bytes()
