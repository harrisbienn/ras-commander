import pytest

from ras_commander.RasBco import BcoMonitor


@pytest.mark.parametrize(('start', 'end', 'duration'), [('-2.80000', '-2.79167', 2.8),
                                                     ('-1.40000', '-1.39583', 1.4), ('0', '0.00417', 0.)])
def test_reads_actual_warmup_from_native_time_window(tmp_path, start, end, duration):
    monitor = BcoMonitor(project_path=tmp_path, project_name='OL', plan_number='01')
    monitor.bco_file.write_text(f'Header\n Solving for Time Window = {start} to {end} Hours\n')
    result = monitor.get_initial_time_window()
    assert result['warmup_duration_hours'] == duration
    assert result['start_hours'] == float(start)


@pytest.mark.parametrize('text', ['No computation window', 'Solving for Time Window = 0 to 0 Hours',
                                  'Solving for Time Window = 1 to 2 Hours'])
def test_missing_or_invalid_window_is_not_zero_warmup(tmp_path, text):
    monitor = BcoMonitor(project_path=tmp_path, project_name='OL', plan_number='01')
    monitor.bco_file.write_text(text)
    with pytest.raises(ValueError):
        monitor.get_initial_time_window()
