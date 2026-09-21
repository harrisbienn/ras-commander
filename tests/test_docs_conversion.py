"""Documentation conversion must fail even when stale rendered files exist."""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestDocumentationConversion(unittest.TestCase):
    def test_failed_conversion_cannot_reuse_stale_output(self):
        script = (
            Path(__file__).resolve().parents[1]
            / ".claude/scripts/prepare_notebooks_for_docs.py"
        )
        spec = importlib.util.spec_from_file_location("prepare_docs", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "examples", root / "rendered"
            source.mkdir()
            output.mkdir()
            (source / "example.ipynb").write_text("{}")
            (output / "example.md").write_text("stale output")
            failure = subprocess.CompletedProcess(
                ["nbconvert"], 1, "", "conversion failed"
            )
            with patch.object(module.subprocess, "run", return_value=failure):
                with self.assertRaises(subprocess.CalledProcessError):
                    module.convert_notebooks(source, output)
