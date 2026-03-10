# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for IUTMS Desktop.

Build command (run from the repo root on Windows):
    pyinstaller iutms.spec

This produces a single-folder distribution in dist/IUTMS/ that contains
the .exe and all bundled dependencies.
"""

import os
import sys
from pathlib import Path

block_cipher = None

# Repo root is the directory containing this spec file
REPO_ROOT = os.path.dirname(os.path.abspath(SPECPATH))

# Collect data files that must be included in the bundle
datas = [
    # SUMO map files
    (os.path.join(REPO_ROOT, 'maps', 'grid.net.xml'), 'maps'),
    (os.path.join(REPO_ROOT, 'maps', 'grid.rou.xml'), 'maps'),
    (os.path.join(REPO_ROOT, 'maps', 'simulation.sumocfg'), 'maps'),
    # Simulation package (Python source needed for subprocess calls)
    (os.path.join(REPO_ROOT, 'simulation'), 'simulation'),
    # Desktop package
    (os.path.join(REPO_ROOT, 'desktop'), 'desktop'),
]

# Include Hyderabad OSM file if it exists
hyd_osm = os.path.join(REPO_ROOT, 'maps', 'hyderabad_hitec.osm')
if os.path.isfile(hyd_osm):
    datas.append((hyd_osm, 'maps'))

# Include pre-built React dashboard if it exists
react_build = os.path.join(REPO_ROOT, 'web', 'client', 'build')
if os.path.isdir(react_build):
    datas.append((react_build, os.path.join('web', 'client', 'build')))

a = Analysis(
    [os.path.join(REPO_ROOT, 'desktop', 'app.py')],
    pathex=[REPO_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=[
        # Core ML
        'torch',
        'torch.nn',
        'torch.optim',
        'torch.distributions',
        'numpy',
        # Web server
        'flask',
        'flask_socketio',
        'engineio',
        'socketio',
        # HTTP
        'requests',
        'urllib3',
        'certifi',
        'charset_normalizer',
        'idna',
        # Plotting
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        'matplotlib.figure',
        # Simulation modules
        'simulation',
        'simulation.agent',
        'simulation.ppo_agent',
        'simulation.env_wrapper',
        'simulation.trainer',
        'simulation.osm_importer',
        # Desktop modules
        'desktop',
        'desktop.app',
        'desktop.telemetry_server',
        # Tkinter
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy packages not needed at runtime
        'pytest',
        'IPython',
        'notebook',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='IUTMS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,       # No console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='assets/icon.ico',  # Uncomment if you add an icon file
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='IUTMS',
)
