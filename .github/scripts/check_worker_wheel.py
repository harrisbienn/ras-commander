"""Check installed wheel resources and CLI startup without running HEC-RAS."""

import importlib.metadata
import importlib.resources
import json
from pathlib import Path
import subprocess
import sysconfig

import ras_commander


def main() -> None:
    distribution = importlib.metadata.distribution("ras-commander")
    installed_root = Path(distribution.locate_file("")).resolve()
    source_root = Path(__file__).resolve().parents[2] / "ras_commander"
    imported = Path(ras_commander.__file__).resolve()
    assert imported.is_relative_to(installed_root), imported
    assert not imported.is_relative_to(source_root), (
        "Source import masks wheel contents"
    )

    contracts = importlib.resources.files("ras_commander").joinpath("contracts")
    required = {
        "scenario-worker-request-v1.2.schema.json",
        "scenario-worker-result-v1.0.schema.json",
    }
    schemas = {
        entry.name: entry
        for entry in contracts.iterdir()
        if entry.name.endswith(".schema.json")
    }
    assert required <= schemas.keys(), required - schemas.keys()
    for resource in schemas.values():
        assert isinstance(json.loads(resource.read_text(encoding="utf-8")), dict)

    entry_points = {
        entry.name: entry
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    }
    assert callable(entry_points["ras-scenario-worker"].load())
    suffix = ".exe" if sysconfig.get_platform().startswith("win") else ""
    name = "ras-scenario-worker" + suffix
    executable = Path(sysconfig.get_path("scripts")) / name
    completed = subprocess.run(
        [str(executable), "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "--request" in completed.stdout and "--result" in completed.stdout
    print(
        f"ras-commander {distribution.version}: "
        f"{len(schemas)} schemas and worker entry point verified"
    )


if __name__ == "__main__":
    main()
