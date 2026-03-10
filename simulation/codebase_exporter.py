"""
Create a portable zip archive of the IUTMS codebase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


EXCLUDED_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    ".venv",
    "venv",
    "env",
}

EXCLUDED_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
}

EXCLUDED_FILE_NAMES = {
    ".DS_Store",
}


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _should_exclude(path: Path, repo_root: Path, output_path: Path | None = None) -> bool:
    relative = path.resolve().relative_to(repo_root)
    if any(part in EXCLUDED_DIR_NAMES for part in relative.parts):
        return True
    if path.name in EXCLUDED_FILE_NAMES:
        return True
    if path.suffix in EXCLUDED_FILE_SUFFIXES:
        return True
    if output_path is not None:
        try:
            if path.resolve() == output_path.resolve():
                return True
        except FileNotFoundError:
            pass
    return False


def create_codebase_zip(output_path: str | Path, repo_root: str | Path | None = None) -> Path:
    """
    Package the repository source into a zip archive.

    Parameters
    ----------
    output_path:
        Destination `.zip` file.
    repo_root:
        Repository root to archive. Defaults to the project root.
    """
    repo_root_path = Path(repo_root) if repo_root is not None else _default_repo_root()
    repo_root_path = repo_root_path.resolve()

    archive_path = Path(output_path).resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()

    files_added = 0
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for file_path in sorted(repo_root_path.rglob("*")):
            if not file_path.is_file():
                continue
            if _should_exclude(file_path, repo_root_path, archive_path):
                continue

            archive.write(file_path, arcname=file_path.relative_to(repo_root_path))
            files_added += 1

    if files_added == 0:
        raise ValueError(f"No files were added to archive {archive_path}")

    return archive_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the full IUTMS codebase as a zip archive.")
    parser.add_argument(
        "--output",
        required=True,
        help="Absolute or relative path for the generated zip archive.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(_default_repo_root()),
        help="Repository root to archive. Defaults to the current project root.",
    )
    args = parser.parse_args()

    archive_path = create_codebase_zip(output_path=args.output, repo_root=args.repo_root)
    print(json.dumps({"archive_path": str(archive_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
