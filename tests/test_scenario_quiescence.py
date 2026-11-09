"""Harmless OS process fixtures for the scenario cancellation boundary.

Only the inventory's RAS command metadata is synthetic. PID/create-time
identities, descendant discovery, signalling and exit verification use psutil
against Python helpers; no HEC executable or model is used.
"""

import importlib
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from ras_commander import RasCmdr, RasProcessInventory, RasProcessRecord
from ras_commander._process_inspection import _ProcessScan


@pytest.mark.skipif(os.name != "nt", reason="Exact-plan cancellation supports Windows")
@pytest.mark.parametrize("worker_died", [False, True])
def test_exact_cancellation_quiesces_real_descendants_only(tmp_path, monkeypatch, worker_died):
    helper = tmp_path / "helper.py"
    ready = tmp_path / "ready.json"
    helper.write_text(
        "import json, os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "if sys.argv[1] == 'root':\n"
        "    subprocess.Popen([sys.executable, __file__, 'child', sys.argv[2]])\n"
        "elif sys.argv[1] == 'child':\n"
        "    leaf = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "    Path(sys.argv[2]).write_text(json.dumps([os.getpid(), leaf.pid]))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    owned = []
    launcher = subprocess.Popen([sys.executable, str(helper), "root", str(ready)])
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        owned.append(psutil.Process(launcher.pid))
        deadline = time.monotonic() + 10
        while not ready.exists():
            assert time.monotonic() < deadline, "Python descendant fixture did not start"
            time.sleep(0.02)
        child_pid, leaf_pid = json.loads(ready.read_text(encoding="utf-8"))
        owned.extend([psutil.Process(child_pid), psutil.Process(leaf_pid)])
        if worker_died:
            launcher.kill()
            launcher.wait(timeout=5)
        target = owned[1] if worker_died else owned[0]
        project = tmp_path / "Fixture.prj"
        plan = tmp_path / "Fixture.p01"
        temporary_hdf = tmp_path / "Fixture.p01.tmp.hdf"
        record = RasProcessRecord(
            pid=target.pid,
            create_time=target.create_time(),
            name="RasUnsteady.exe" if worker_died else "Ras.exe",
            command_line=("RasUnsteady.exe", str(temporary_hdf)) if worker_died else (
                "Ras.exe", str(project), str(plan),
            ),
            working_directory=str(tmp_path),
        )

        def scan(**_kwargs):
            records = (record,) if target.is_running() else ()
            return _ProcessScan(
                inventory=RasProcessInventory(observed_at=time.time(), processes=records),
                handles={record.identity: target} if records else {},
            )

        inspection = importlib.import_module("ras_commander._process_inspection")
        monkeypatch.setattr(inspection, "_scan_ras_process_handles", scan)
        monkeypatch.setattr(
            RasCmdr, "_resolve_plan_process_paths",
            lambda *_args: ("01", project, plan, temporary_hdf),
        )
        result = RasCmdr.cancel_plan_exact(
            "01", ras_object=SimpleNamespace(check_initialized=lambda: None), timeout_seconds=2,
        )
        assert result.quiescence_confirmed is True
        assert {child_pid, leaf_pid}.issubset({item.pid for item in result.stopped})
        assert all(not process.is_running() for process in owned)
        assert unrelated.poll() is None
    finally:
        for process in reversed(owned):
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(owned, timeout=5)
        if launcher.poll() is None:
            launcher.kill()
        launcher.wait(timeout=5)
        unrelated.kill()
        unrelated.wait(timeout=5)
