import hashlib
from pathlib import Path

import pytest

from ras_commander import RasCmdr, RasUtils


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (1, "01"),
        ("1", "01"),
        ("01", "01"),
        ("001", "01"),
        ("p01", "01"),
        ("P01", "01"),
        (".p01", "01"),
        ("g02", "02"),
        ("project.p03", "03"),
        (Path("project.p04"), "04"),
    ],
)
def test_normalize_ras_number_accepts_prefixed_and_path_forms(raw, expected):
    assert RasUtils.normalize_ras_number(raw) == expected


@pytest.mark.parametrize("raw", ["p00", "p100", "plan01", "foo", "p"])
def test_normalize_ras_number_rejects_invalid_prefixed_forms(raw):
    with pytest.raises(ValueError):
        RasUtils.normalize_ras_number(raw)


def test_compute_hdf_path_uses_normalized_plan_number(tmp_path):
    ras_obj = type(
        "FakeRas",
        (),
        {"project_folder": tmp_path, "project_name": "Demo"},
    )()

    assert RasCmdr._get_hdf_path("p01", ras_obj) == tmp_path / "Demo.p01.hdf"


@pytest.mark.parametrize("newline", ["\r\n", "\n"])
def test_explicit_newline_preparation_preserves_bytes_and_backup(tmp_path, newline):
    path = tmp_path / "Example.g01"
    original = b"Geom Title=Test\r\nProgram Version=6.60\nDescription=\xe9  \r\nLast"
    path.write_bytes(original)
    with pytest.raises(ValueError, match="Mixed newline"):
        RasUtils._read_text_lines_preserving_newline(path)
    evidence = RasUtils.normalize_text_newlines(str(path), newline=newline)
    expected = original.replace(b"\r\n", b"\n").replace(b"\n", newline.encode())
    assert path.read_bytes() == expected
    assert Path(evidence["backup"]).read_bytes() == original
    assert evidence["before_sha256"] == hashlib.sha256(original).hexdigest()
    assert evidence["after_sha256"] == hashlib.sha256(expected).hexdigest()
    assert evidence["original_crlf_count"] == 2
    assert evidence["original_bare_lf_count"] == 1
    assert RasUtils._detect_text_newline(path) == newline
    assert not RasUtils.normalize_text_newlines(path, newline=newline)["changed"]
    assert Path(evidence["backup"]).read_bytes() == original


@pytest.mark.parametrize(
    "content,suffix",
    [(b"x\x00\n", ".g01"), (b"x\ry\n", ".p01"), (b"x\n", ".hdf")],
)
def test_newline_preparation_rejects_unsupported_content(tmp_path, content, suffix):
    path = tmp_path / ("Example" + suffix)
    path.write_bytes(content)
    with pytest.raises(ValueError):
        RasUtils.normalize_text_newlines(path, newline="\r\n")
    assert path.read_bytes() == content
    assert len(list(tmp_path.iterdir())) == 1


def test_newline_preparation_refuses_existing_backup(tmp_path):
    path = tmp_path / "Example.prj"
    path.write_bytes(b"Proj Title=Test\n")
    backup = tmp_path / "Example.prj.newline.bak"
    backup.write_bytes(b"historical evidence")
    with pytest.raises(FileExistsError):
        RasUtils.normalize_text_newlines(path, newline="\r\n")
    assert path.read_bytes() == b"Proj Title=Test\n"
    assert backup.read_bytes() == b"historical evidence"


def test_newline_preparation_validates_target_before_writing(tmp_path):
    path = tmp_path / "Example.prj"
    path.write_bytes(b"Proj Title=Test\n")
    with pytest.raises(ValueError, match="CRLF or LF"):
        RasUtils.normalize_text_newlines(path, newline="\r")
    assert len(list(tmp_path.iterdir())) == 1


def test_newline_preparation_failed_replace_preserves_original(tmp_path, monkeypatch):
    path = tmp_path / "Example.g01"
    original = b"Geom Title=Example\r\nProgram Version=6.60\n"
    path.write_bytes(original)

    def deny_replace(source, destination):
        raise PermissionError("file in use")

    monkeypatch.setattr("os.replace", deny_replace)
    with pytest.raises(PermissionError, match="file in use"):
        RasUtils.normalize_text_newlines(path, newline="\r\n")
    assert path.read_bytes() == original
    assert (tmp_path / "Example.g01.newline.bak").read_bytes() == original
    assert len(list(tmp_path.iterdir())) == 2
