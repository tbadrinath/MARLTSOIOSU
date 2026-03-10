"""
osm_importer.py
---------------
OpenStreetMap → SUMO pipeline for the Intelligent Urban Traffic Management
System (IUTMS).

Workflow
--------
1. Search for a location by name using the Nominatim geocoding API to obtain
   a bounding box.
2. Download OSM map data for that bounding box via the Overpass API.
3. Convert the downloaded ``.osm`` file to a SUMO ``.net.xml`` network using
   SUMO's ``netconvert`` tool.
4. Generate random vehicle routes on the network using SUMO's
   ``randomTrips.py`` script (bundled with every SUMO installation) or, as a
   fallback, a direct call to ``duarouter``.

All heavy I/O is isolated here so that the training loop (trainer.py) only
needs to swap ``net_file`` / ``route_file`` paths.

External dependencies
---------------------
- ``requests`` (already in requirements.txt)
- SUMO installation (provides ``netconvert``, ``randomTrips.py``)
  These are optional at import time — the module loads fine without them.
  Errors are raised only when the relevant functions are actually called.

References
----------
- LucasAlegre/sumo-rl  — OSM conversion pattern
- AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control — SUMO setup
- cts198859/deeprl_signal_control — randomTrips usage
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nominatim / Overpass constants
# ---------------------------------------------------------------------------

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL  = "https://overpass-api.de/api/interpreter"

# User-Agent required by Nominatim usage policy.  Can be overridden via the
# IUTMS_USER_AGENT environment variable for custom deployments.
_USER_AGENT = os.environ.get(
    "IUTMS_USER_AGENT",
    "IUTMS-TrafficSim/1.0 (https://github.com/tbadrinath/MARLTSOIOSU)",
)

# Margin (in degrees) added around a place's bounding box
BBOX_MARGIN = 0.005

# Maximum side length (degrees) of a bounding box.  Overpass will refuse
# very large queries, and netconvert gets slow on huge networks.
MAX_BBOX_SIDE = 0.08

# Default number of random trip vehicles to generate
DEFAULT_NUM_VEHICLES = 400

# Default simulation period for randomTrips
DEFAULT_ROUTE_PERIOD = 1.0  # vehicles inserted per second on average


# ---------------------------------------------------------------------------
# Internal request helper
# ---------------------------------------------------------------------------

def _get(url: str, params: dict, timeout: float = 30.0):
    """Issue a GET request, returning the parsed JSON response."""
    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The 'requests' package is required for OSM import. "
            "Install it with:  pip install requests"
        ) from exc

    headers = {"User-Agent": _USER_AGENT}
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_location(query: str, limit: int = 5) -> List[Dict]:
    """
    Search for a place by name using the Nominatim geocoding API.

    Parameters
    ----------
    query : str
        Free-form location string, e.g. ``"Manhattan, New York"`` or
        ``"Bangalore, India"``.
    limit : int
        Maximum number of results to return (1–10).

    Returns
    -------
    list[dict]
        Each dict contains:
        ``display_name``, ``lat``, ``lon``,
        ``boundingbox`` (list of 4 str: [min_lat, max_lat, min_lon, max_lon]).

    Raises
    ------
    RuntimeError
        If the HTTP request fails or returns no results.
    """
    params = {
        "q":      query,
        "format": "json",
        "limit":  str(max(1, min(limit, 10))),
    }
    resp = _get(NOMINATIM_URL, params, timeout=15.0)
    results = resp.json()
    if not results:
        raise RuntimeError(f"No results found for location: {query!r}")
    return [
        {
            "display_name": r.get("display_name", ""),
            "lat":          float(r.get("lat", 0)),
            "lon":          float(r.get("lon", 0)),
            "boundingbox":  r.get("boundingbox", []),
            "osm_type":     r.get("osm_type", ""),
            "osm_id":       r.get("osm_id", ""),
        }
        for r in results
    ]


def _clamp_bbox(
    min_lat: float, max_lat: float, min_lon: float, max_lon: float
) -> Tuple[float, float, float, float]:
    """Apply margin and clamp bbox to MAX_BBOX_SIDE."""
    min_lat -= BBOX_MARGIN
    max_lat += BBOX_MARGIN
    min_lon -= BBOX_MARGIN
    max_lon += BBOX_MARGIN

    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon

    if lat_span > MAX_BBOX_SIDE:
        mid = (min_lat + max_lat) / 2.0
        min_lat = mid - MAX_BBOX_SIDE / 2.0
        max_lat = mid + MAX_BBOX_SIDE / 2.0

    if lon_span > MAX_BBOX_SIDE:
        mid = (min_lon + max_lon) / 2.0
        min_lon = mid - MAX_BBOX_SIDE / 2.0
        max_lon = mid + MAX_BBOX_SIDE / 2.0

    return min_lat, max_lat, min_lon, max_lon


def download_osm(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    output_file: str,
) -> str:
    """
    Download OSM road network data for a bounding box using the Overpass API
    and save it to *output_file*.

    The Overpass query fetches all ``highway`` ways and their nodes, which is
    exactly what SUMO's ``netconvert`` needs to build a road network.

    Parameters
    ----------
    min_lat, max_lat, min_lon, max_lon : float
        Geographic bounding box in decimal degrees (WGS-84).
    output_file : str
        Destination path for the downloaded ``.osm`` file.

    Returns
    -------
    str
        The resolved absolute path of *output_file*.
    """
    min_lat, max_lat, min_lon, max_lon = _clamp_bbox(
        min_lat, max_lat, min_lon, max_lon
    )

    bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    query = (
        f"[out:xml][timeout:60];"
        f"("
        f"  way[\"highway\"]({bbox_str});"
        f"  >;  "      # fetch all nodes referenced by the ways
        f");"
        f"out body;"
    )

    logger.info("Downloading OSM data for bbox %s …", bbox_str)
    resp = _get(OVERPASS_URL, {"data": query}, timeout=90.0)

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(resp.content)
    logger.info("OSM data saved to %s (%d bytes)", out, out.stat().st_size)
    return str(out.resolve())


def _find_sumo_tool(tool_name: str) -> Optional[str]:
    """
    Locate a SUMO tool binary or Python helper.

    Search order:
      1. ``SUMO_HOME`` environment variable (if set) + ``/bin`` or ``/tools``
      2. System PATH
    """
    sumo_home = os.environ.get("SUMO_HOME", "")

    candidates: List[str] = []
    if sumo_home:
        candidates.extend([
            os.path.join(sumo_home, "bin",   tool_name),
            os.path.join(sumo_home, "tools", tool_name),
            os.path.join(sumo_home, "bin",   tool_name + ".exe"),
        ])

    # Fall back to whatever is on PATH
    path_tool = shutil.which(tool_name)
    if path_tool:
        candidates.append(path_tool)

    for c in candidates:
        if os.path.isfile(c):
            return c

    return None


def convert_to_sumo(
    osm_file: str,
    net_file: str,
    extra_netconvert_args: Optional[List[str]] = None,
) -> str:
    """
    Convert an ``.osm`` file to a SUMO ``.net.xml`` using ``netconvert``.

    Parameters
    ----------
    osm_file : str
        Path to the downloaded OSM file.
    net_file : str
        Destination path for the SUMO network file.
    extra_netconvert_args : list[str] | None
        Additional CLI arguments forwarded to ``netconvert``.

    Returns
    -------
    str
        Resolved path of *net_file*.

    Raises
    ------
    RuntimeError
        If ``netconvert`` is not found or exits with an error.
    """
    netconvert = _find_sumo_tool("netconvert")
    if netconvert is None:
        raise RuntimeError(
            "SUMO's 'netconvert' tool was not found.  "
            "Install SUMO and ensure it is on PATH or set SUMO_HOME."
        )

    Path(net_file).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        netconvert,
        "--osm-files",      osm_file,
        "--output-file",    net_file,
        "--geometry.remove", "true",
        "--roundabouts.guess", "true",
        "--ramps.guess",    "true",
        "--junctions.join", "true",
        "--tls.guess-signals", "true",
        "--tls.discard-loaded", "true",
        "--no-internal-links", "false",
        "--keep-edges.by-vclass", "passenger",
        "--remove-edges.isolated", "true",
        "--no-warnings",    "true",
    ]
    if extra_netconvert_args:
        cmd.extend(extra_netconvert_args)

    logger.info("Running netconvert: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"netconvert failed (exit {result.returncode}):\n"
            + (result.stderr or result.stdout)
        )

    logger.info("SUMO network written to %s", net_file)
    return str(Path(net_file).resolve())


def generate_routes(
    net_file: str,
    route_file: str,
    num_vehicles: int = DEFAULT_NUM_VEHICLES,
    period: float = DEFAULT_ROUTE_PERIOD,
    seed: int = 42,
) -> str:
    """
    Generate random vehicle routes for the SUMO network using
    ``randomTrips.py`` (preferred) or ``duarouter`` (fallback).

    Parameters
    ----------
    net_file : str
        Path to the SUMO ``.net.xml`` network file.
    route_file : str
        Destination path for the ``.rou.xml`` route file.
    num_vehicles : int
        Approximate number of vehicles to insert.
    period : float
        Average time gap (seconds) between vehicle insertions.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    str
        Resolved path of *route_file*.

    Raises
    ------
    RuntimeError
        If neither ``randomTrips.py`` nor ``duarouter`` is available.
    """
    Path(route_file).parent.mkdir(parents=True, exist_ok=True)
    trips_file = route_file.replace(".rou.xml", ".trips.xml")

    # ------------------------------------------------------------------
    # Strategy 1 – randomTrips.py (bundled with SUMO)
    # ------------------------------------------------------------------
    random_trips = _find_sumo_tool("randomTrips.py")
    if random_trips:
        python_exe = sys.executable
        cmd_trips = [
            python_exe, random_trips,
            "-n",   net_file,
            "-o",   trips_file,
            "-e",   str(num_vehicles),
            "-p",   str(period),
            "--seed", str(seed),
            "--fringe-factor", "5",
            "--min-distance", "100",
            "--validate",
        ]
        logger.info("Generating random trips …")
        res = subprocess.run(cmd_trips, capture_output=True, text=True)
        if res.returncode != 0:
            logger.warning("randomTrips.py failed: %s", res.stderr)
        else:
            # Route the trips with duarouter
            duarouter = _find_sumo_tool("duarouter")
            if duarouter:
                cmd_route = [
                    duarouter,
                    "--net-file",     net_file,
                    "--route-files",  trips_file,
                    "--output-file",  route_file,
                    "--ignore-errors", "true",
                    "--no-warnings",   "true",
                    "--seed",          str(seed),
                ]
                logger.info("Routing trips with duarouter …")
                res2 = subprocess.run(cmd_route, capture_output=True, text=True)
                if res2.returncode == 0 and Path(route_file).exists():
                    logger.info("Routes written to %s", route_file)
                    return str(Path(route_file).resolve())
                else:
                    logger.warning("duarouter failed: %s", res2.stderr)

    # ------------------------------------------------------------------
    # Strategy 2 – write a minimal synthetic route file so that the
    # simulation can still be started even without SUMO tools.
    # This is a last-resort fallback; real simulations need real routes.
    # ------------------------------------------------------------------
    logger.warning(
        "SUMO route generation tools not found.  "
        "Writing a minimal synthetic route file to %s",
        route_file,
    )
    _write_synthetic_routes(route_file, num_vehicles)
    return str(Path(route_file).resolve())


def _write_synthetic_routes(route_file: str, num_vehicles: int) -> None:
    """Write a minimal `.rou.xml` with uniform departure intervals."""
    end_time = num_vehicles  # one vehicle per second
    header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:noNamespaceSchemaLocation='
        '"http://sumo.dlr.de/xsd/routes_file.xsd">\n'
        '  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" maxSpeed="13.89"/>\n'
    )
    footer = "</routes>\n"
    with open(route_file, "w") as fh:
        fh.write(header)
        for i in range(num_vehicles):
            fh.write(
                f'  <vehicle id="v{i}" type="car" depart="{i}"/>\n'
            )
        fh.write(footer)


def import_map(
    location: str,
    output_dir: str,
    num_vehicles: int = DEFAULT_NUM_VEHICLES,
    seed: int = 42,
) -> Dict[str, str]:
    """
    High-level pipeline: search → download → convert → generate routes.

    Parameters
    ----------
    location : str
        Human-readable location name, e.g. ``"Downtown Toronto, Canada"``.
    output_dir : str
        Directory where all generated files are written.
    num_vehicles : int
        Number of vehicles for route generation.
    seed : int
        Random seed for route generation.

    Returns
    -------
    dict with keys:
        ``"display_name"`` – resolved Nominatim display name,
        ``"osm_file"``     – path to downloaded .osm,
        ``"net_file"``     – path to generated .net.xml,
        ``"route_file"``   – path to generated .rou.xml,
        ``"bbox"``         – [min_lat, max_lat, min_lon, max_lon].

    Raises
    ------
    RuntimeError
        If any step of the pipeline fails.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1 – geocode
    logger.info("Searching for location: %s", location)
    results = search_location(location, limit=1)
    place = results[0]
    display_name = place["display_name"]

    # Parse bounding box returned by Nominatim
    bb = place["boundingbox"]  # [min_lat, max_lat, min_lon, max_lon]
    min_lat, max_lat = float(bb[0]), float(bb[1])
    min_lon, max_lon = float(bb[2]), float(bb[3])

    logger.info("Location resolved: %s  bbox=%s", display_name, bb)

    # Step 2 – download OSM
    osm_file = str(out / "map.osm")
    download_osm(min_lat, max_lat, min_lon, max_lon, osm_file)

    # Step 3 – convert to SUMO network
    net_file = str(out / "map.net.xml")
    convert_to_sumo(osm_file, net_file)

    # Step 4 – generate routes
    route_file = str(out / "map.rou.xml")
    generate_routes(net_file, route_file, num_vehicles=num_vehicles, seed=seed)

    # Step 5 – write SUMO config file
    config_file = str(out / "simulation.sumocfg")
    generate_sumo_config(net_file, route_file, config_file)

    return {
        "display_name": display_name,
        "osm_file":     osm_file,
        "net_file":     net_file,
        "route_file":   route_file,
        "config_file":  config_file,
        "bbox":         [min_lat, max_lat, min_lon, max_lon],
    }


def generate_sumo_config(
    net_file: str,
    route_file: str,
    config_file: str,
    begin: int = 0,
    end: int = 3600,
    step_length: float = 1.0,
    use_gui: bool = False,
    quit_on_end: bool = True,
    seed: int = 42,
) -> str:
    """
    Write a SUMO simulation configuration (``.sumocfg``) file.

    A ``.sumocfg`` is the standard way to bundle all SUMO input files and
    simulation parameters into a single XML document that can be passed
    directly to ``sumo`` or ``sumo-gui``:

    .. code-block:: bash

        sumo-gui -c simulation.sumocfg

    Parameters
    ----------
    net_file : str
        Path to the SUMO ``.net.xml`` network file (absolute or relative
        to the config file's directory).
    route_file : str
        Path to the ``.rou.xml`` route file.
    config_file : str
        Destination path for the generated ``.sumocfg``.
    begin : int
        Simulation start time in seconds (default 0).
    end : int
        Simulation end time in seconds (default 3600).
    step_length : float
        Duration of each simulation step in seconds (default 1.0).
    use_gui : bool
        When ``True`` the ``gui-settings-file`` attribute is added so that
        ``sumo-gui`` opens with a sensible default view.  Not required for
        headless ``sumo``.
    quit_on_end : bool
        When ``True``, SUMO exits automatically when the simulation finishes
        (equivalent to ``--quit-on-end true``).
    seed : int
        Random seed written to the ``<random_number>`` section (default 42).

    Returns
    -------
    str
        Resolved absolute path of the written ``.sumocfg`` file.
    """
    cfg_path = Path(config_file).resolve()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    # Store paths relative to the config file so the config is portable
    net_rel   = os.path.relpath(net_file,   start=str(cfg_path.parent))
    route_rel = os.path.relpath(route_file, start=str(cfg_path.parent))

    quit_flag = "true" if quit_on_end else "false"

    content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">\n'
        '\n'
        '    <input>\n'
        f'        <net-file value="{net_rel}"/>\n'
        f'        <route-files value="{route_rel}"/>\n'
        '    </input>\n'
        '\n'
        '    <time>\n'
        f'        <begin value="{begin}"/>\n'
        f'        <end value="{end}"/>\n'
        f'        <step-length value="{step_length}"/>\n'
        '    </time>\n'
        '\n'
        '    <processing>\n'
        f'        <time-to-teleport value="-1"/>\n'
        '        <waiting-time-memory value="1000"/>\n'
        '    </processing>\n'
        '\n'
        '    <report>\n'
        '        <no-step-log value="true"/>\n'
        '        <no-warnings value="true"/>\n'
        '    </report>\n'
        '\n'
        '    <random_number>\n'
        f'        <seed value="{seed}"/>\n'
        '    </random_number>\n'
        '\n'
        '    <output>\n'
        '        <quit-on-end value="{quit}"/>\n'.format(quit=quit_flag) +
        '    </output>\n'
        '\n'
        '</configuration>\n'
    )

    cfg_path.write_text(content, encoding="utf-8")
    logger.info("SUMO config written to %s", cfg_path)
    return str(cfg_path)
