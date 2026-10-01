"""Warmup changes preserve native plan formatting and reject ambiguity."""
import pytest

from ras_commander import RasPlan


@pytest.mark.parametrize('newline', [b'\n', b'\r\n'])
def test_warmup_change_preserves_every_other_byte(tmp_path, newline):
    path = tmp_path / 'test.p01'
    original = newline.join([b'Plan Title=Test', b'UNET MaxInSteps= 336 ', b'UNET DtIC= 0.5 ', b''])
    path.write_bytes(original)
    RasPlan.set_warmup_steps(path, 672)
    assert path.read_bytes() == original.replace(b'336', b'672')
    RasPlan.set_warmup_steps(path, 672)
    assert path.read_bytes() == original.replace(b'336', b'672')


@pytest.mark.parametrize('steps', [-1, 1.5, True, '672'])
def test_invalid_step_counts_do_not_write(tmp_path, steps):
    path = tmp_path / 'test.p01'
    path.write_bytes(b'UNET MaxInSteps=336\n')
    with pytest.raises(ValueError):
        RasPlan.set_warmup_steps(path, steps)
    assert path.read_bytes() == b'UNET MaxInSteps=336\n'


@pytest.mark.parametrize('content', [b'Plan Title=Test\n', b'UNET MaxInSteps=1\nUNET MaxInSteps=2\n',
                                   b'UNET MaxInSteps=bad\n', b'Plan Title=Test\r\nUNET MaxInSteps=336\n'])
def test_invalid_plan_does_not_write(tmp_path, content):
    path = tmp_path / 'test.p01'
    path.write_bytes(content)
    with pytest.raises(ValueError):
        RasPlan.set_warmup_steps(path, 672)
    assert path.read_bytes() == content
