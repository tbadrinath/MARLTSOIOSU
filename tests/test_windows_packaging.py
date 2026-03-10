from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_windows_workflow_exists():
    workflow = REPO_ROOT / ".github" / "workflows" / "windows-exe.yml"
    assert workflow.exists()
    content = workflow.read_text(encoding="utf-8")
    assert "Build Windows EXE" in content
    assert "IUTMS-Setup.exe" in content
    assert "softprops/action-gh-release@v2" in content


def test_server_package_can_build_windows_exe():
    package_json = REPO_ROOT / "web" / "server" / "package.json"
    data = json.loads(package_json.read_text(encoding="utf-8"))

    assert data["bin"] == "server.js"
    assert "build:exe" in data["scripts"]
    assert "pkg" in data
    assert "../client/build/**/*" in data["pkg"]["assets"]


def test_windows_readme_link_targets_latest_release_asset():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "IUTMS-Setup.exe" in readme
    assert "releases/latest/download/IUTMS-Setup.exe" in readme


def test_pyinstaller_spec_bundles_maps():
    spec = REPO_ROOT / "packaging" / "windows" / "iutms_gui.spec"
    assert spec.exists()
    content = spec.read_text(encoding="utf-8")
    assert '"maps"' in content
    assert 'simulation/run_gui.py' in content
