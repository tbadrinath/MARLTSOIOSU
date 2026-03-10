"""
run_gui.py
----------
Standalone launcher that builds a complete SUMO environment from an
OpenStreetMap location (or the bundled 3×3 grid) and opens it in
``sumo-gui`` for live visual inspection.

Usage examples
--------------
# Run with the built-in 3×3 grid (no network required):
    python -m simulation.run_gui

# Import an OSM map and open it in sumo-gui:
    python -m simulation.run_gui --location "Bangalore, India"

# OSM import, headless sumo (no GUI), 600-second simulation:
    python -m simulation.run_gui --location "Berlin Mitte" --no-gui --end 600

# Run the existing grid without SUMO-GUI (headless, useful for CI):
    python -m simulation.run_gui --no-gui --end 300

Environment variables
---------------------
SUMO_HOME   Path to the SUMO installation root (e.g. ``/usr/share/sumo``).
            When set, SUMO tools are located inside ``$SUMO_HOME/tools``.
DISPLAY     If unset and sumo-gui is requested, the launcher automatically
            starts an Xvfb virtual framebuffer so that the GUI can render
            off-screen in headless CI environments.

References
----------
- LucasAlegre/sumo-rl    – TraCI + OSM setup patterns
- AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control – SUMO config
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).resolve().parent.parent
MAPS_DIR    = REPO_ROOT / "maps"
DEFAULT_NET = str(MAPS_DIR / "grid.net.xml")
DEFAULT_ROU = str(MAPS_DIR / "grid.rou.xml")
DEFAULT_CFG = str(MAPS_DIR / "simulation.sumocfg")

# Xvfb display number used when $DISPLAY is not available
XVFB_DISPLAY = ":99"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_sumo_home() -> None:
    """Set SUMO_HOME to the system-installed SUMO if not already set."""
    if os.environ.get("SUMO_HOME"):
        return
    candidates = ["/usr/share/sumo", "/opt/sumo", "/usr/local/share/sumo"]
    for path in candidates:
        if Path(path).is_dir():
            os.environ["SUMO_HOME"] = path
            logger.info("SUMO_HOME auto-set to %s", path)
            return
    logger.warning(
        "SUMO_HOME is not set and could not be auto-detected. "
        "Network validation will be disabled."
    )


def _start_xvfb() -> "subprocess.Popen | None":
    """
    Start an Xvfb virtual framebuffer so sumo-gui can render off-screen.

    Returns the Popen handle on success, or ``None`` if Xvfb is not
    available (in that case sumo-gui should be skipped).
    """
    xvfb = shutil.which("Xvfb")
    if xvfb is None:
        logger.warning("Xvfb not found – cannot start virtual display.")
        return None

    # Remove any stale lock file from a previous run (OS temp dir is portable)
    lock = Path(tempfile.gettempdir()) / f".X{XVFB_DISPLAY[1:]}-lock"
    if lock.exists():
        lock.unlink(missing_ok=True)

    proc = subprocess.Popen(
        [xvfb, XVFB_DISPLAY, "-screen", "0", "1280x720x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = XVFB_DISPLAY
    time.sleep(1.0)  # give Xvfb a moment to initialise
    logger.info("Xvfb started on display %s (PID %d)", XVFB_DISPLAY, proc.pid)
    return proc


def _build_grid_config(cfg_path: str, end: int, step_length: float) -> None:
    """
    Generate a ``.sumocfg`` for the built-in 3×3 grid maps.

    This is used when no ``--location`` is supplied so the launcher can
    start immediately without an internet connection.
    """
    from simulation.osm_importer import generate_sumo_config  # noqa: PLC0415

    generate_sumo_config(
        net_file    = DEFAULT_NET,
        route_file  = DEFAULT_ROU,
        config_file = cfg_path,
        begin       = 0,
        end         = end,
        step_length = step_length,
    )
    logger.info("Grid simulation config written to %s", cfg_path)


def _import_osm_map(location: str, output_dir: str, num_vehicles: int, seed: int,
                    end: int, step_length: float) -> str:
    """
    Run the full OSM → SUMO pipeline for *location* and return the
    path to the generated ``.sumocfg``.
    """
    from simulation.osm_importer import import_map  # noqa: PLC0415

    logger.info("Importing OSM map for '%s' into %s …", location, output_dir)
    result = import_map(location, output_dir, num_vehicles=num_vehicles, seed=seed)

    cfg_path = result["config_file"]
    logger.info("OSM import complete.")
    logger.info("  Network:     %s", result["net_file"])
    logger.info("  Routes:      %s", result["route_file"])
    logger.info("  SUMO config: %s", cfg_path)
    logger.info("  Bbox:        %s", result["bbox"])
    return cfg_path


def run_simulation(
    config_file: str,
    use_gui: bool = True,
    end: int = 3600,
    xvfb_proc: "subprocess.Popen | None" = None,
) -> int:
    """
    Launch ``sumo`` or ``sumo-gui`` with *config_file* and wait for it to
    finish.  Returns the process exit code.
    """
    binary = "sumo-gui" if use_gui else "sumo"
    if shutil.which(binary) is None:
        logger.error(
            "'%s' not found.  Install SUMO and ensure it is on PATH.", binary
        )
        return 1

    cmd = [binary, "-c", config_file, "--end", str(end)]
    # Auto-start the simulation when running with sumo-gui so that it does not
    # wait for a manual "play" button press (useful for CI/scripted demos).
    if use_gui:
        cmd += ["--start", "true"]
    logger.info("Launching: %s", " ".join(cmd))

    try:
        proc = subprocess.run(cmd)
        return proc.returncode
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
        return 0
    finally:
        if xvfb_proc is not None:
            xvfb_proc.terminate()
            logger.info("Xvfb stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Set up a SUMO environment from OSM (or the built-in grid) and "
            "open it in sumo-gui for live simulation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--location",
        default=None,
        metavar="PLACE",
        help=(
            "OSM location to simulate, e.g. 'Bangalore, India'.  "
            "When omitted the bundled 3×3 grid network is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(MAPS_DIR / "osm_import"),
        metavar="DIR",
        help="Directory where OSM-imported files are written.",
    )
    parser.add_argument(
        "--num-vehicles",
        type=int,
        default=400,
        help="Number of vehicles to generate (OSM import only).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=3600,
        help="Simulation end time (seconds).",
    )
    parser.add_argument(
        "--step-length",
        type=float,
        default=1.0,
        help="Duration of each simulation step (seconds).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for route generation and SUMO.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headless sumo instead of sumo-gui.",
    )
    parser.add_argument(
        "--config-file",
        default=DEFAULT_CFG,
        metavar="FILE",
        help="Path to write/use the .sumocfg file (grid mode only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_sumo_home()

    use_gui = not args.no_gui

    # ------------------------------------------------------------------
    # Step 1 – prepare the SUMO config file
    # ------------------------------------------------------------------
    if args.location:
        # Full OSM pipeline
        config_file = _import_osm_map(
            location     = args.location,
            output_dir   = args.output_dir,
            num_vehicles = args.num_vehicles,
            seed         = args.seed,
            end          = args.end,
            step_length  = args.step_length,
        )
    else:
        # Use the bundled 3×3 grid network
        logger.info("No --location given; using bundled 3×3 grid network.")
        _build_grid_config(args.config_file, end=args.end, step_length=args.step_length)
        config_file = args.config_file

    # ------------------------------------------------------------------
    # Step 2 – handle display for sumo-gui
    # ------------------------------------------------------------------
    xvfb_proc = None
    if use_gui:
        display = os.environ.get("DISPLAY", "")
        if not display:
            logger.info(
                "No $DISPLAY set – starting Xvfb virtual framebuffer for sumo-gui."
            )
            xvfb_proc = _start_xvfb()
            if xvfb_proc is None:
                logger.warning(
                    "Xvfb unavailable; falling back to headless sumo."
                )
                use_gui = False

    # ------------------------------------------------------------------
    # Step 3 – launch the simulation
    # ------------------------------------------------------------------
    exit_code = run_simulation(
        config_file = config_file,
        use_gui     = use_gui,
        end         = args.end,
        xvfb_proc   = xvfb_proc,
    )

    if exit_code != 0:
        logger.error("SUMO exited with code %d", exit_code)
        sys.exit(exit_code)

    logger.info("Simulation finished successfully.")


if __name__ == "__main__":
    main()
