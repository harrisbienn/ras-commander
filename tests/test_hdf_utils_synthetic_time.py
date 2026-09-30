"""Regression coverage for RAS summary times beyond nanosecond date bounds."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ras_commander import HdfUtils


@pytest.mark.parametrize("year", [2016, 2263, 3000])
@pytest.mark.parametrize("unit,seconds", [("days", 86400), ("hours", 3600)])
def test_summary_times_preserve_subseconds_and_missing_values(year, unit, seconds):
    start = datetime(year, 1, 2)
    offsets = np.asarray([0, 0.16 / seconds, 3 * 86400 / seconds, np.nan])
    result = HdfUtils.convert_timesteps_to_datetimes(offsets, start, time_unit=unit)
    assert list(result[:3].to_pydatetime()) == [
        start, start + timedelta(milliseconds=200), start + timedelta(days=3)
    ]
    assert pd.isna(result[3])


def test_summary_times_reject_unknown_units():
    with pytest.raises(ValueError, match="Unsupported time unit"):
        HdfUtils.convert_timesteps_to_datetimes(np.asarray([0]), datetime(3000, 1, 2), time_unit="fortnights")
