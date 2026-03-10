# Building IUTMS as a Windows Executable

This guide explains how to build the IUTMS Desktop application into a
standalone Windows `.exe` that bundles all Python dependencies. Users can
run the application without installing Python, pip packages, or Node.js.

---

## Quick Start (Windows)

```batch
# 1. Install Python 3.9+ from https://python.org (check "Add to PATH")

# 2. Clone the repository
git clone https://github.com/tbadrinath/MARLTSOIOSU.git
cd MARLTSOIOSU

# 3. Run the build script
build_exe.bat
```

The executable will be created at `dist\IUTMS\IUTMS.exe`.

---

## Detailed Steps

### Prerequisites

| Tool      | Required | Notes                                         |
|-----------|----------|-----------------------------------------------|
| Python    | 3.9+     | Must be on PATH                               |
| pip       | latest   | Comes with Python                             |
| Node.js   | 18+      | Optional - only for building React dashboard  |
| SUMO      | 1.15+    | Optional - for real traffic simulations       |

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
pip install pyinstaller
```

### Step 2: (Optional) Build the React Dashboard

If you have Node.js installed, you can build the web dashboard to bundle
it with the executable:

```bash
cd web/client
npm install
npm run build
cd ../..
```

The desktop app works without this step — it has its own built-in
matplotlib-based dashboard.

### Step 3: Build the Executable

**Option A — Using the batch script (Windows):**
```batch
build_exe.bat
```

**Option B — Using the Python build script (cross-platform):**
```bash
python build_exe.py
```

**Option C — Using PyInstaller directly:**
```bash
pyinstaller iutms.spec --noconfirm
```

### Step 4: Run the Application

Navigate to `dist/IUTMS/` and run `IUTMS.exe` (Windows) or `./IUTMS` (Linux/macOS).

---

## What's Included in the Bundle

The built executable includes:

- **Python runtime** — no need to install Python
- **PyTorch** — for DQN and PPO neural networks
- **NumPy** — numerical computation
- **Matplotlib** — embedded live charts in the GUI
- **Flask + Flask-SocketIO** — built-in telemetry server (replaces Node.js)
- **Requests** — HTTP client for OSM import
- **Simulation code** — all MARL training and environment modules
- **Map files** — bundled 3x3 grid network and Hyderabad OSM
- **React dashboard** — pre-built static files (if built before bundling)

---

## What's NOT Included

- **SUMO** — the traffic simulator must be installed separately.
  Download from: https://sumo.dlr.de/docs/Downloads.php
  The desktop app detects SUMO automatically and provides a setup guide
  in the "SUMO Setup" tab. The app works in **demo mode** without SUMO.

---

## Running Without Building

You can also run the desktop app directly from source without building
an exe:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the desktop app
python -m desktop
```

---

## Desktop App Features

The GUI provides these tabs:

| Tab             | Description                                              |
|-----------------|----------------------------------------------------------|
| **Dashboard**   | Live matplotlib charts (reward, speed, congestion, CO2)  |
| **Training**    | Configure and launch DQN/PPO training with all params    |
| **OSM Import**  | Search cities, import real-world maps from OpenStreetMap  |
| **SUMO Setup**  | Detect SUMO installation, set SUMO_HOME path             |
| **About**       | Project information and algorithm details                |

### Demo Mode

Click "Start Demo" on the Dashboard tab to see simulated training
metrics without SUMO. This generates realistic data showing how RL
agents improve over training episodes.

### Telemetry Server

The app starts an embedded Python telemetry server on port 3001 that
is compatible with the original Node.js server. The React web dashboard
(if built) can be accessed at http://localhost:3001.

---

## Troubleshooting

### "SUMO not found"
Install SUMO and either:
- Set the `SUMO_HOME` environment variable, or
- Use the "SUMO Setup" tab in the app to browse for the installation

### "flask not installed" (server error)
Run: `pip install flask flask-socketio`

### "matplotlib not installed" (no charts)
Run: `pip install matplotlib`
The app will show a message and you can still use the web dashboard.

### PyInstaller build fails
- Ensure you have enough disk space (PyTorch bundles are large)
- Try: `pip install --upgrade pyinstaller`
- On Windows, run as Administrator if permission errors occur

### Antivirus blocks the .exe
Some antivirus software flags PyInstaller executables. Add an exception
for the `dist/IUTMS/` directory.
