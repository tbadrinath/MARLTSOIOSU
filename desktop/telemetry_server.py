"""
telemetry_server.py
-------------------
Pure-Python telemetry server that replaces the Node.js Express + Socket.io
server.  Uses Flask + flask-socketio so the existing React dashboard and
the Python trainer can communicate without requiring Node.js.

Endpoints (compatible with the original server.js):
    POST /api/metrics          - receive step metrics from the trainer
    GET  /api/metrics/history  - return buffered history
    GET  /api/status           - health check
    POST /api/demo/start       - start server-side demo simulation
    POST /api/demo/stop        - stop demo simulation
    GET  /api/demo/status      - demo status
    GET  /api/osm/search       - proxy Nominatim geocoding
    POST /api/osm/import       - trigger OSM import pipeline

Socket.io event emitted:
    "step_metrics"  - broadcast to all connected dashboard clients
"""

from __future__ import annotations

import logging
import math
import os
import random
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports – these are only needed when the server actually starts
# ---------------------------------------------------------------------------

flask = None  # type: Any
flask_socketio = None  # type: Any
_app = None  # type: Any
_socketio = None  # type: Any

MAX_HISTORY = 500
_metrics_buffer: Deque[Dict] = deque(maxlen=MAX_HISTORY)

# Demo state
_demo_timer: Optional[threading.Timer] = None
_demo_running = False
_demo_step = 0
_demo_episode = 1
DEMO_INTERVAL = 0.15
DEMO_STEP_SIZE = 10
DEMO_MAX_STEPS = 3600
DEMO_TOTAL_EP = 50


def _ensure_imports() -> None:
    """Import Flask and flask-socketio on first use."""
    global flask, flask_socketio
    if flask is None:
        import flask as _flask
        import flask_socketio as _flask_sio
        flask = _flask
        flask_socketio = _flask_sio


def _generate_demo_metric(step: int, episode: int) -> Dict:
    """Generate one synthetic step-metrics payload mimicking RL training."""
    ep_progress = min((episode - 1) / DEMO_TOTAL_EP, 1.0)
    phase_progress = step / DEMO_MAX_STEPS

    total_reward = round(
        -12 + ep_progress * 20
        + math.sin(phase_progress * math.pi * 4) * 2
        + (random.random() - 0.5) * 3,
        3,
    )
    avg_speed = round(
        6 + ep_progress * 8
        + math.sin(phase_progress * math.pi * 3) * 1.5
        + (random.random() - 0.5) * 0.8,
        2,
    )
    vehicles_in_network = round(
        120 - ep_progress * 80
        - math.sin(phase_progress * math.pi * 2) * 10
        + (random.random() - 0.5) * 15
    )
    co2_emissions = round(
        800 - ep_progress * 400
        + math.sin(phase_progress * math.pi * 2) * 60
        + (random.random() - 0.5) * 40,
        1,
    )

    return {
        "step": step,
        "episode": episode,
        "total_reward": total_reward,
        "avg_speed": avg_speed,
        "vehicles_in_network": vehicles_in_network,
        "co2_emissions": co2_emissions,
    }


def _demo_tick() -> None:
    """Called periodically to emit demo metrics."""
    global _demo_step, _demo_episode, _demo_timer, _demo_running

    if not _demo_running:
        return

    metric = _generate_demo_metric(_demo_step, _demo_episode)
    metric["serverTs"] = int(time.time() * 1000)
    _metrics_buffer.append(metric)

    if _socketio is not None:
        _socketio.emit("step_metrics", metric)

    _demo_step += DEMO_STEP_SIZE
    if _demo_step >= DEMO_MAX_STEPS:
        _demo_step = 0
        _demo_episode = 1 if _demo_episode >= DEMO_TOTAL_EP else _demo_episode + 1

    if _demo_running:
        _demo_timer = threading.Timer(DEMO_INTERVAL, _demo_tick)
        _demo_timer.daemon = True
        _demo_timer.start()


def create_app(
    static_folder: Optional[str] = None,
    serve_react: bool = False,
) -> Any:
    """
    Create and configure the Flask application.

    Parameters
    ----------
    static_folder : str | None
        Path to the React build folder to serve as static files.
    serve_react : bool
        If True, serve the React dashboard at /.
    """
    _ensure_imports()
    global _app, _socketio

    if static_folder and serve_react:
        _app = flask.Flask(
            __name__,
            static_folder=static_folder,
            static_url_path="",
        )
    else:
        _app = flask.Flask(__name__)

    _app.config["SECRET_KEY"] = "iutms-telemetry"
    _socketio = flask_socketio.SocketIO(
        _app,
        cors_allowed_origins="*",
        async_mode="threading",
    )

    # -----------------------------------------------------------------------
    # REST endpoints
    # -----------------------------------------------------------------------

    @_app.route("/api/metrics", methods=["POST"])
    def post_metrics():
        payload = flask.request.get_json(silent=True)
        if not payload or "step" not in payload:
            return flask.jsonify({"error": "Missing required field: step"}), 400

        numeric_fields = [
            "episode", "step", "avg_speed", "co2_emissions",
            "total_reward", "vehicles_in_network",
        ]
        for field in numeric_fields:
            if field in payload and not isinstance(payload[field], (int, float)):
                return flask.jsonify(
                    {"error": f'Field "{field}" must be a number'}
                ), 400

        payload["serverTs"] = int(time.time() * 1000)
        _metrics_buffer.append(payload)
        _socketio.emit("step_metrics", payload)
        return flask.jsonify({"ok": True}), 200

    @_app.route("/api/metrics/history", methods=["GET"])
    def get_history():
        limit = min(
            int(flask.request.args.get("limit", MAX_HISTORY)),
            MAX_HISTORY,
        )
        data = list(_metrics_buffer)[-limit:]
        return flask.jsonify({"count": len(data), "data": data})

    @_app.route("/api/status", methods=["GET"])
    def get_status():
        return flask.jsonify({
            "status": "ok",
            "metricsBuffered": len(_metrics_buffer),
            "demoRunning": _demo_running,
        })

    # -----------------------------------------------------------------------
    # Demo endpoints
    # -----------------------------------------------------------------------

    @_app.route("/api/demo/start", methods=["POST"])
    def demo_start():
        global _demo_running, _demo_step, _demo_episode, _demo_timer
        if not _demo_running:
            _demo_running = True
            _demo_step = 0
            _demo_episode = 1
            _demo_timer = threading.Timer(DEMO_INTERVAL, _demo_tick)
            _demo_timer.daemon = True
            _demo_timer.start()
        return flask.jsonify({"ok": True, "message": "Demo simulation started."})

    @_app.route("/api/demo/stop", methods=["POST"])
    def demo_stop():
        global _demo_running, _demo_timer
        _demo_running = False
        if _demo_timer is not None:
            _demo_timer.cancel()
            _demo_timer = None
        return flask.jsonify({"ok": True, "message": "Demo simulation stopped."})

    @_app.route("/api/demo/status", methods=["GET"])
    def demo_status():
        return flask.jsonify({
            "running": _demo_running,
            "episode": _demo_episode,
            "step": _demo_step,
        })

    # -----------------------------------------------------------------------
    # OSM endpoints
    # -----------------------------------------------------------------------

    @_app.route("/api/osm/search", methods=["GET"])
    def osm_search():
        query = (flask.request.args.get("q") or "").strip()
        if not query:
            return flask.jsonify({"error": "Missing query parameter: q"}), 400
        limit = min(int(flask.request.args.get("limit", 5)), 10)

        try:
            import requests as req_lib
            params = {"q": query, "format": "json", "limit": str(limit)}
            headers = {
                "User-Agent": "IUTMS-TrafficSim/1.0 "
                "(https://github.com/tbadrinath/MARLTSOIOSU)"
            }
            resp = req_lib.get(
                "https://nominatim.openstreetmap.org/search",
                params=params,
                headers=headers,
                timeout=15,
            )
            data = resp.json()
            results = [
                {
                    "display_name": r.get("display_name", ""),
                    "lat": float(r.get("lat", 0)),
                    "lon": float(r.get("lon", 0)),
                    "boundingbox": r.get("boundingbox", []),
                    "osm_type": r.get("osm_type", ""),
                    "osm_id": r.get("osm_id", ""),
                }
                for r in (data if isinstance(data, list) else [])
            ]
            return flask.jsonify({"count": len(results), "results": results})
        except Exception as exc:
            return flask.jsonify(
                {"error": f"Nominatim request failed: {exc}"}
            ), 502

    @_app.route("/api/osm/import", methods=["POST"])
    def osm_import():
        body = flask.request.get_json(silent=True) or {}
        location = (body.get("location") or "").strip()
        if not location:
            return flask.jsonify(
                {"error": "Missing required field: location"}
            ), 400

        num_vehicles = body.get("num_vehicles", 400)
        seed = body.get("seed", 42)

        try:
            # Determine repo root
            repo_root = Path(__file__).resolve().parent.parent
            sys.path.insert(0, str(repo_root))
            from simulation.osm_importer import import_map

            sanitised = "".join(
                c if c.isalnum() or c in ("_", "-") else "_"
                for c in location
            )[:64]
            output_dir = str(repo_root / "maps" / "osm" / sanitised)

            result = import_map(
                location, output_dir,
                num_vehicles=num_vehicles, seed=seed,
            )
            return flask.jsonify(result)
        except Exception as exc:
            return flask.jsonify(
                {"error": "OSM import failed", "detail": str(exc)[:1000]}
            ), 500

    # -----------------------------------------------------------------------
    # Serve React dashboard (if configured)
    # -----------------------------------------------------------------------

    if serve_react and static_folder:
        @_app.route("/")
        def serve_index():
            return flask.send_from_directory(static_folder, "index.html")

        @_app.errorhandler(404)
        def fallback(_e):
            index = os.path.join(static_folder, "index.html")
            if os.path.isfile(index):
                return flask.send_from_directory(static_folder, "index.html")
            return flask.jsonify({"error": "Not found"}), 404

    return _app, _socketio


def start_server(
    host: str = "0.0.0.0",
    port: int = 3001,
    static_folder: Optional[str] = None,
    serve_react: bool = False,
) -> None:
    """Start the telemetry server (blocking)."""
    app, sio = create_app(static_folder=static_folder, serve_react=serve_react)
    logger.info("Telemetry server starting on http://%s:%d", host, port)
    sio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)


def start_server_thread(
    host: str = "127.0.0.1",
    port: int = 3001,
    static_folder: Optional[str] = None,
    serve_react: bool = False,
) -> threading.Thread:
    """Start the telemetry server in a background daemon thread."""
    app, sio = create_app(static_folder=static_folder, serve_react=serve_react)

    def _run():
        sio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)

    t = threading.Thread(target=_run, daemon=True, name="telemetry-server")
    t.start()
    logger.info("Telemetry server started in background on http://%s:%d", host, port)
    return t


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_server()
