"""Native Manning edits must retain class identity and unrequested parameters."""

import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

import ras_commander._landcover_native as native
from ras_commander import HdfLandCover


class _Row(dict):
    def IsNull(self, column):
        return self[column] is None


class _Columns(list):
    @property
    def Count(self):
        return len(self)


def _table(rows):
    return SimpleNamespace(
        Rows=[_Row(row) for row in rows],
        Columns=_Columns(SimpleNamespace(ColumnName=name) for name in rows[0]),
    )


@pytest.mark.parametrize("damage", [
    None, "truncate_class_name", "remove_class", "change_id", "change_other",
    "change_unrequested_manning", "drop_column", "duplicate_id",
])
@pytest.mark.parametrize("requested_class", ["Water", "Emergent Herbaceous Wetlands"])
def test_native_manning_postconditions_restore_unrequested_damage(tmp_path, monkeypatch, damage, requested_class):
    sidecar = tmp_path / "LandCover.hdf"
    sidecar.write_bytes(b"original native sidecar")
    backup = tmp_path / "LandCover.backup.hdf"
    backup.write_bytes(b"original native backup")
    long_name = "Emergent Herbaceous Wetlands"
    original = [
        {"ID": 1, "Name": "Water", "ManningsN": 0.12, "Other": 7.0},
        {"ID": 2, "Name": long_name, "ManningsN": 0.12, "Other": 8.0},
    ]
    saved = deepcopy(original)
    requested_index = next(index for index, row in enumerate(saved) if row["Name"] == requested_class)
    unrequested_index = 1 - requested_index
    saved[requested_index]["ManningsN"] = 0.06
    if damage == "truncate_class_name":
        saved[1]["Name"] = long_name[:-1]
    elif damage == "remove_class":
        saved.pop()
    elif damage == "change_id":
        saved[1]["ID"] = 3
    elif damage == "change_other":
        saved[1]["Other"] = 0.0
    elif damage == "change_unrequested_manning":
        saved[unrequested_index]["ManningsN"] = 0.06
    elif damage == "drop_column":
        for row in saved:
            del row["Other"]
    elif damage == "duplicate_id":
        saved[1]["ID"] = 1
    tables = {"original": _table(original), "saved": _table(saved)}

    class Layer:
        Parameters = object()

        def __init__(self, identity):
            self.Classification = identity

        def TryAssigningNewParamtersUsingTable(self, table, apply):
            assert apply is True
            assert table.Rows[requested_index]["ManningsN"] == 0.06
            sidecar.write_bytes(b"native save")
            backup.write_bytes(b"native generated backup")
            return True

    layers = iter([Layer("original"), Layer("saved")])
    api = SimpleNamespace(
        LandCoverType=SimpleNamespace(LandCover=object()),
        TryLoadLayer=lambda *_: (True, next(layers), ""),
        GetClassificationVariablesAsDataTable=lambda identity, _: tables[identity],
    )
    monkeypatch.setattr(native, "find_hecras_install", lambda _: tmp_path)
    monkeypatch.setattr(native, "load_clr", lambda _: None)
    monkeypatch.setitem(sys.modules, "RasMapperLib", SimpleNamespace(LandCoverLayer=api))

    if damage is None:
        result = HdfLandCover.set_landcover_mannings_n(sidecar, {requested_class: 0.06}, hecras_version="6.6")
        assert result["changed"] == 1
        assert result["not_requested"] == 1
        assert result["recompute_required"] is True
        assert sidecar.read_bytes() == b"native save"
    else:
        expected_error = (
            "did not persist Manning values"
            if requested_index == 1 and damage in {"truncate_class_name", "remove_class"}
            else "class|parameter"
        )
        with pytest.raises(RuntimeError, match=expected_error):
            HdfLandCover.set_landcover_mannings_n(sidecar, {requested_class: 0.06}, hecras_version="6.6")
        assert sidecar.read_bytes() == b"original native sidecar"
        assert backup.read_bytes() == b"original native backup"
    snapshots = list(tmp_path.glob("LandCover.native_parameters.*.backup.hdf"))
    assert len(snapshots) == 1
    assert snapshots[0].read_bytes() == b"original native sidecar"
