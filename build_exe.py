#!/usr/bin/env python3
"""
build_exe.py
------------
Cross-platform build script for creating the IUTMS Desktop executable.

Usage:
    python build_exe.py

This script:
  1. Installs required Python packages (if missing).
  2. Optionally builds the React dashboard (if Node.js is available).
  3. Runs PyInstaller with the iutms.spec file.
  4. Reports the output location.

Output:
    dist/IUTMS/IUTMS.exe   (Windows)
    dist/IUTMS/IUTMS        (Linux/macOS)
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def run(cmd: list, cwd: str | None = None, check: bool = True) -> int:
    """Run a command, printing it first."""
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or str(REPO_ROOT))
    if check and result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
    return result.returncode


def install_dependencies() -> None:
    """Install Python dependencies from requirements.txt."""
    print("\n[1/4] Installing Python dependencies...")
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=False)
    # Ensure PyInstaller is installed
    run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=False)


def build_react_dashboard() -> None:
    """Build the React dashboard if Node.js is available."""
    client_dir = REPO_ROOT / "web" / "client"
    if not (client_dir / "package.json").exists():
        print("\n[2/4] No React client found, skipping dashboard build.")
        return

    npm = shutil.which("npm")
    if not npm:
        print("\n[2/4] Node.js/npm not found, skipping React dashboard build.")
        print("       The desktop app will work without it.")
        return

    print("\n[2/4] Building React dashboard...")
    run([npm, "install"], cwd=str(client_dir), check=False)
    run([npm, "run", "build"], cwd=str(client_dir), check=False)


def build_exe() -> None:
    """Run PyInstaller to create the executable."""
    spec_file = REPO_ROOT / "iutms.spec"
    if not spec_file.exists():
        print(f"ERROR: {spec_file} not found!")
        sys.exit(1)

    print("\n[3/4] Building executable with PyInstaller...")
    ret = run([
        sys.executable, "-m", "PyInstaller",
        str(spec_file),
        "--noconfirm",
    ])
    if ret != 0:
        print("ERROR: PyInstaller build failed.")
        sys.exit(1)


def report() -> None:
    """Report the build output location."""
    ext = ".exe" if platform.system() == "Windows" else ""
    exe_path = REPO_ROOT / "dist" / "IUTMS" / f"IUTMS{ext}"

    print("\n" + "=" * 50)
    print("  IUTMS Desktop - Build Complete!")
    print("=" * 50)

    if exe_path.exists():
        print(f"\n  Output: {exe_path}")
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"  Size:   {size_mb:.1f} MB")
    else:
        dist_dir = REPO_ROOT / "dist" / "IUTMS"
        if dist_dir.exists():
            print(f"\n  Output directory: {dist_dir}")
        else:
            print("\n  WARNING: Output not found. Check PyInstaller logs.")

    print("\nNote: SUMO must be installed separately for real simulations.")
    print("Download SUMO: https://sumo.dlr.de/docs/Downloads.php")
    print()


def main() -> None:
    print("=" * 50)
    print("  IUTMS Desktop - Executable Builder")
    print("=" * 50)

    install_dependencies()
    build_react_dashboard()
    build_exe()
    report()


if __name__ == "__main__":
    main()
