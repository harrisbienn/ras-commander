"""Exercise diagnostic cell scaling against native DSS, including row order."""

import hashlib
import subprocess
import sys

import numpy as np
import pytest
from ras_commander import RasDss


@pytest.mark.parametrize("factors", [[], [1, 2], [[-1]], [[np.nan]], [[np.inf]]])
def test_invalid_cell_factors_refuse_before_creating_output(tmp_path, factors):
    source = tmp_path / "source.dss"
    source.write_bytes(b"unopened fixture")
    output = tmp_path / "output.dss"
    with pytest.raises(ValueError, match="cell_factors"):
        RasDss.copy_grid_with_zero_tail(source, output, "/SHG/T/PRECIP///F/", 0,
                                        cell_factors=factors)
    assert not output.exists()


@pytest.mark.integration
def test_scaled_grid_preserves_time_space_nodata_and_source(tmp_path):
    # Separate JVM lifetimes release Windows source-file locks before hashing.
    source = tmp_path / "source.dss"
    output = tmp_path / "output.dss"
    create = """
from datetime import datetime, timedelta
import numpy as np
import sys
from ras_commander import RasDss
RasDss.write_grid_timeseries(sys.argv[1], '/SHG/T/PRECIP///F/',
    np.array([[[1., np.nan], [2., 3.]], [[4., np.nan], [5., 0.]]]),
    [datetime(3000, 1, 1) + timedelta(minutes=5*i) for i in range(3)],
    {'cell_size':500, 'origin':(352500,1061500), 'crs':'SHG',
     'units':'IN', 'data_type':'PER-CUM', 'compression':'ZLIB'})
"""
    subprocess.run([sys.executable, "-I", "-c", create, str(source)],
                   check=True, capture_output=True, text=True)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    scale = """
import sys
import numpy as np
from ras_commander import RasDss
factors = np.array([[1.0382734028640963, 1.], [0., 1.0634803527552934]])
r = RasDss.copy_grid_with_zero_tail(sys.argv[1], sys.argv[2],
    '/SHG/T/PRECIP///F/', 1, cell_factors=factors)
assert r['cell_scaling']['readback_verified']
assert r['source_record_count'] == 2 and r['appended_record_count'] == 1
assert r['source_start'] == '3000-01-01T00:00:00'
assert r['padded_end'] == '3000-01-01T00:15:00'
assert r['output_lower_left_cell'] == (705, 2123)
first = RasDss.read_grid(sys.argv[2], r['shifted_pathnames'][0])
np.testing.assert_array_equal(first['data'],
    (np.array([[1., np.nan], [2., 3.]]) * factors).astype(np.float32))
last = RasDss.read_grid(sys.argv[2], r['appended_pathnames'][0])
np.testing.assert_array_equal(last['data'], [[0., np.nan], [0., 0.]])
assert first['units'] == 'IN' and first['cell_size'] == 500
assert len(RasDss.get_catalog(sys.argv[2])) == 3
for bad, suffix in [(np.ones((3,2)), 'shape'), (np.full((2,2), 1e300), 'overflow')]:
    try:
        RasDss.copy_grid_with_zero_tail(sys.argv[1], sys.argv[2]+suffix,
            '/SHG/T/PRECIP///F/', 0, cell_factors=bad)
    except ValueError as exc:
        assert suffix in str(exc), str(exc)
    else:
        raise AssertionError('invalid scaling accepted')
"""
    subprocess.run([sys.executable, "-I", "-c", scale, str(source), str(output)],
                   check=True, capture_output=True, text=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
