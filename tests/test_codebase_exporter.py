from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from simulation.codebase_exporter import create_codebase_zip


def test_create_codebase_zip_includes_project_sources_and_excludes_generated_dirs(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    (repo_root / "README.md").write_text("# demo\n")
    (repo_root / "requirements.txt").write_text("pytest\n")
    (repo_root / "simulation").mkdir()
    (repo_root / "simulation" / "__init__.py").write_text("")
    (repo_root / "simulation" / "trainer.py").write_text("print('train')\n")
    (repo_root / "web").mkdir()
    (repo_root / "web" / "server").mkdir()
    (repo_root / "web" / "server" / "server.js").write_text("console.log('ok');\n")
    (repo_root / "web" / "client").mkdir()
    (repo_root / "web" / "client" / "build").mkdir(parents=True)
    (repo_root / "web" / "client" / "build" / "asset.js").write_text("ignored build\n")
    (repo_root / "tests").mkdir()
    (repo_root / "tests" / "test_basic.py").write_text("def test_ok():\n    assert True\n")

    (repo_root / ".git").mkdir()
    (repo_root / ".git" / "config").write_text("[core]\n")
    (repo_root / "web" / "client" / "node_modules").mkdir(parents=True)
    (repo_root / "web" / "client" / "node_modules" / "ignored.js").write_text("ignored\n")
    (repo_root / "__pycache__").mkdir()
    (repo_root / "__pycache__" / "ignored.pyc").write_bytes(b"x")
    (repo_root / ".pytest_cache").mkdir()
    (repo_root / ".pytest_cache" / "state").write_text("ignored\n")

    archive_path = create_codebase_zip(tmp_path / "exports" / "project.zip", repo_root=repo_root)

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())

    assert "README.md" in names
    assert "requirements.txt" in names
    assert "simulation/trainer.py" in names
    assert "web/server/server.js" in names
    assert "tests/test_basic.py" in names
    assert ".git/config" not in names
    assert "web/client/build/asset.js" not in names
    assert "web/client/node_modules/ignored.js" not in names
    assert "__pycache__/ignored.pyc" not in names
    assert ".pytest_cache/state" not in names


def test_create_codebase_zip_omits_output_archive_from_zip(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("# demo\n")

    output_path = repo_root / "exports" / "project.zip"
    archive_path = create_codebase_zip(output_path, repo_root=repo_root)

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())

    assert "README.md" in names
    assert "exports/project.zip" not in names
