# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


repo_root = Path.cwd()

datas = [
    (str(repo_root / "maps"), "maps"),
]

client_build = repo_root / "web" / "client" / "build"
if client_build.exists():
    datas.append((str(client_build), "web/client/build"))

hiddenimports = [
    "simulation.osm_importer",
]


a = Analysis(
    ["simulation/run_gui.py"],
    pathex=[str(repo_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="IUTMS-GUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
