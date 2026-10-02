"""Small HEC-RAS-layout edge cases supplement real DeLoutre validation."""
import hashlib

import h5py
import numpy as np
import pytest

from ras_commander import HdfResultsPlan

BASE = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series'


@pytest.fixture
def exchange_hdf(tmp_path):
    path = tmp_path / 'exchange.p01.hdf'
    with h5py.File(path, 'w') as hdf:
        hdf.create_dataset(f'{BASE}/Time Date Stamp (ms)', data=np.array(
            ['02JAN3000 00:00:00:000', '02JAN3000 00:05:00:000', '02JAN3000 00:10:00:000'], dtype='S24'))
        hdf.create_dataset('Geometry/2D Flow Areas/Attributes', data=np.array(
            [('A',), ('B',)], dtype=[('Name', 'S16')]))
        fields = ['Type', 'River', 'Reach', 'RS', 'Connection', 'US SA/2D', 'DS SA/2D', 'US Type', 'DS Type']
        hdf.create_dataset('Geometry/Structures/Attributes', data=np.array([
            ('Lateral', 'River', 'Reach', '10', '', '', 'A', 'XS', '2D'),
            ('Connection', '', '', '', 'AtoB', 'A', 'B', '2D', '2D'),
        ], dtype=[(field, 'S16') for field in fields]))
        for group, name, flows in [('Lateral Structures', 'River Reach 10', [2., 2., -2.]),
                                    ('SA 2D Area Conn', 'AtoB', [1., 1., 1.])]:
            ds = hdf.create_dataset(f'{BASE}/{group}/{name}/Structure Variables', data=np.array(flows)[:, None])
            ds.attrs['Variable_Unit'] = np.array([['Total Flow', 'cfs']], dtype='S16')
        ds = hdf.create_dataset(f'{BASE}/Boundary Conditions/outlet', data=np.array([[0., 0.5]] * 3))
        ds.attrs.update({'Columns': np.array(['Stage', 'Flow'], dtype='S16'), '2D Area': np.bytes_('B'),
                         'Flow': np.bytes_('cfs')})
    return path


def test_signed_transfer_cancels_between_areas_and_source_unchanged(exchange_hdf):
    before = hashlib.sha256(exchange_hdf.read_bytes()).hexdigest()
    result = HdfResultsPlan.get_exchange_diagnostics(exchange_hdf, {'outlet': -1})
    assert result['time'][0] == '3000-01-02T00:00:00'
    assert result['volume_units'] == 'ft^3'
    assert [s['signed_volume'] for s in result['structures']] == [600., 600.]
    assert [a['net_nonprecipitation_volume'] for a in result['areas']] == [0., 300.]
    assert hashlib.sha256(exchange_hdf.read_bytes()).hexdigest() == before


def test_unknown_nonzero_boundary_prevents_reconciliation(exchange_hdf):
    result = HdfResultsPlan.get_exchange_diagnostics(exchange_hdf)
    assert result['areas'][1]['net_nonprecipitation_volume'] is None
    assert result['areas'][1]['unresolved_boundaries'] == ['outlet']


@pytest.mark.parametrize('signs', [{'outlet': 0}, {'wrong': 1}])
def test_rejects_invalid_boundary_directions(exchange_hdf, signs):
    with pytest.raises(ValueError):
        HdfResultsPlan.get_exchange_diagnostics(exchange_hdf, signs)


@pytest.mark.parametrize('fault', ['missing', 'nonfinite', 'mixed_units', 'nonmonotonic', 'unknown_area', 'unmapped'])
def test_rejects_incomplete_or_ambiguous_exchange_data(exchange_hdf, fault):
    with h5py.File(exchange_hdf, 'a') as hdf:
        target = f'{BASE}/Lateral Structures/River Reach 10/Structure Variables'
        if fault == 'missing':
            del hdf[target]
        elif fault == 'nonfinite':
            hdf[target][1, 0] = np.nan
        elif fault == 'mixed_units':
            hdf[target].attrs['Variable_Unit'] = np.array([['Total Flow', 'm3/s']], dtype='S16')
        elif fault == 'nonmonotonic':
            hdf[f'{BASE}/Time Date Stamp (ms)'][1] = b'02JAN3000 00:00:00:000'
        elif fault == 'unknown_area':
            hdf[f'{BASE}/Boundary Conditions/outlet'].attrs['2D Area'] = np.bytes_('Missing')
        else:
            data = hdf['Geometry/Structures/Attributes'][1:]
            del hdf['Geometry/Structures/Attributes']
            hdf.create_dataset('Geometry/Structures/Attributes', data=data)
    with pytest.raises((KeyError, ValueError)):
        HdfResultsPlan.get_exchange_diagnostics(exchange_hdf, {'outlet': -1})
