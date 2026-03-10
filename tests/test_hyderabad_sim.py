"""
tests/test_hyderabad_sim.py
---------------------------
Unit tests for simulation/run_hyderabad.py.

All SUMO/TraCI calls and heavy I/O are mocked so that:
  • No real SUMO process is started.
  • No network access is required.
  • matplotlib rendering is skipped (Agg backend tested separately).
  • Tests run fast in CI without SUMO installed.
"""

from __future__ import annotations

import os
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import numpy as np

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_net_xml(tmp_path: Path) -> str:
    """Write a minimal SUMO net.xml with two TLS junctions."""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <junction id="J1" type="traffic_light" x="100.0" y="100.0" incLanes="e1_0" intLanes="" shape=""/>
  <junction id="J2" type="traffic_light" x="200.0" y="200.0" incLanes="e2_0" intLanes="" shape=""/>
  <junction id=":J1_0" type="internal" x="100.0" y="100.0" incLanes="" intLanes="" shape=""/>
  <edge id="e1" from="J1" to="J2" priority="1">
    <lane id="e1_0" index="0" speed="13.89" length="150.0"
          shape="100.0,100.0 200.0,200.0"/>
  </edge>
  <edge id="e2" from="J2" to="J1" priority="1">
    <lane id="e2_0" index="0" speed="13.89" length="150.0"
          shape="200.0,200.0 100.0,100.0"/>
  </edge>
  <tlLogic id="J1" type="static" programID="0" offset="0">
    <phase duration="30" state="GrGr"/>
    <phase duration="5"  state="yryr"/>
    <phase duration="30" state="rGrG"/>
    <phase duration="5"  state="ryry"/>
  </tlLogic>
</net>
"""
    p = tmp_path / "map.net.xml"
    p.write_text(content)
    return str(p)


def _make_rou_xml(tmp_path: Path) -> str:
    """Write a minimal SUMO route file."""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" maxSpeed="13.89"/>
  <route id="r0" edges="e1"/>
  <vehicle id="v0" type="car" route="r0" depart="0"/>
  <vehicle id="v1" type="car" route="r0" depart="10"/>
</routes>
"""
    p = tmp_path / "map.rou.xml"
    p.write_text(content)
    return str(p)


# ---------------------------------------------------------------------------
# OSM file presence
# ---------------------------------------------------------------------------

class TestOsmFilePresent:
    def test_hyderabad_osm_exists(self):
        """The pre-embedded Hyderabad HITEC City OSM file must be present."""
        osm = Path(__file__).resolve().parent.parent / "maps" / "hyderabad_hitec.osm"
        assert osm.exists(), f"OSM file missing: {osm}"

    def test_osm_is_valid_xml(self):
        """The OSM file must be well-formed XML."""
        osm = Path(__file__).resolve().parent.parent / "maps" / "hyderabad_hitec.osm"
        if not osm.exists():
            pytest.skip("OSM file not found")
        tree = ET.parse(str(osm))
        root = tree.getroot()
        assert root.tag == "osm"

    def test_osm_has_nodes(self):
        """OSM file must have at least 10 nodes."""
        osm = Path(__file__).resolve().parent.parent / "maps" / "hyderabad_hitec.osm"
        if not osm.exists():
            pytest.skip("OSM file not found")
        tree = ET.parse(str(osm))
        nodes = tree.getroot().findall("node")
        assert len(nodes) >= 10, f"Expected ≥10 nodes, got {len(nodes)}"

    def test_osm_has_ways(self):
        """OSM file must have at least 4 road ways."""
        osm = Path(__file__).resolve().parent.parent / "maps" / "hyderabad_hitec.osm"
        if not osm.exists():
            pytest.skip("OSM file not found")
        tree = ET.parse(str(osm))
        ways = tree.getroot().findall("way")
        assert len(ways) >= 4, f"Expected ≥4 ways, got {len(ways)}"

    def test_osm_highway_tags(self):
        """OSM ways must include highway tags."""
        osm = Path(__file__).resolve().parent.parent / "maps" / "hyderabad_hitec.osm"
        if not osm.exists():
            pytest.skip("OSM file not found")
        tree = ET.parse(str(osm))
        highway_tags = [
            t for way in tree.getroot().findall("way")
            for t in way.findall("tag")
            if t.get("k") == "highway"
        ]
        assert len(highway_tags) >= 4

    def test_osm_hitec_city_coordinates(self):
        """OSM nodes must be in the Hyderabad HITEC City bounding box."""
        osm = Path(__file__).resolve().parent.parent / "maps" / "hyderabad_hitec.osm"
        if not osm.exists():
            pytest.skip("OSM file not found")
        tree = ET.parse(str(osm))
        lats = [float(n.get("lat", 0)) for n in tree.getroot().findall("node")]
        lons = [float(n.get("lon", 0)) for n in tree.getroot().findall("node")]
        # Hyderabad is roughly 17°N, 78°E
        assert all(17.0 <= lat <= 18.0 for lat in lats), "Nodes outside Hyderabad latitude range"
        assert all(78.0 <= lon <= 79.0 for lon in lons), "Nodes outside Hyderabad longitude range"


# ---------------------------------------------------------------------------
# _load_network_geometry
# ---------------------------------------------------------------------------

class TestLoadNetworkGeometry:
    def test_parses_junctions(self, tmp_path):
        from simulation.run_hyderabad import _load_network_geometry
        net_file = _make_net_xml(tmp_path)
        geom = _load_network_geometry(net_file)
        junctions = geom["junctions"]
        # J1 and J2 should be present (internal junction :J1_0 excluded)
        ids = [j[0] for j in junctions]
        assert "J1" in ids
        assert "J2" in ids
        assert ":J1_0" not in ids

    def test_parses_tls_junctions(self, tmp_path):
        from simulation.run_hyderabad import _load_network_geometry
        net_file = _make_net_xml(tmp_path)
        geom = _load_network_geometry(net_file)
        assert "J1" in geom["tls_junctions"]
        assert "J2" in geom["tls_junctions"]

    def test_parses_edges(self, tmp_path):
        from simulation.run_hyderabad import _load_network_geometry
        net_file = _make_net_xml(tmp_path)
        geom = _load_network_geometry(net_file)
        assert len(geom["edges"]) >= 1

    def test_bbox_computed(self, tmp_path):
        from simulation.run_hyderabad import _load_network_geometry
        net_file = _make_net_xml(tmp_path)
        geom = _load_network_geometry(net_file)
        xmin, xmax, ymin, ymax = geom["bbox"]
        assert xmin < xmax
        assert ymin < ymax

    def test_empty_net_returns_defaults(self, tmp_path):
        from simulation.run_hyderabad import _load_network_geometry
        net_file = str(tmp_path / "empty.net.xml")
        Path(net_file).write_text("<net/>")
        geom = _load_network_geometry(net_file)
        assert geom["junctions"] == []
        assert geom["bbox"] == (0, 1, 0, 1)


# ---------------------------------------------------------------------------
# render_frame
# ---------------------------------------------------------------------------

class TestRenderFrame:
    def _make_geom(self):
        return {
            "junctions":     [("J1", 100.0, 100.0), ("J2", 200.0, 200.0)],
            "edges":         [[(100.0, 100.0), (200.0, 200.0)]],
            "tls_junctions": {"J1"},
            "bbox":          (80.0, 220.0, 80.0, 220.0),
        }

    def test_creates_png(self, tmp_path):
        from simulation.run_hyderabad import render_frame
        out = str(tmp_path / "frame.png")
        frame_data = {
            "step": 50,
            "vehicle_positions": [(120.0, 130.0)],
            "vehicle_speeds":    [10.0],
            "tls_states":        {"J1": "GrGr", "J2": "rGrG"},
            "metrics": {"vehicles": 5, "avg_wait": 0.5, "queue": 2, "reward": 0.3},
        }
        render_frame(self._make_geom(), frame_data, out, total_steps=400)
        assert Path(out).exists(), "render_frame did not create PNG"

    def test_creates_png_with_empty_vehicles(self, tmp_path):
        from simulation.run_hyderabad import render_frame
        out = str(tmp_path / "frame_empty.png")
        frame_data = {
            "step": 0, "vehicle_positions": [], "vehicle_speeds": [],
            "tls_states": {}, "metrics": {},
        }
        render_frame(self._make_geom(), frame_data, out, total_steps=100)
        assert Path(out).exists()

    def test_creates_png_with_stopped_vehicles(self, tmp_path):
        from simulation.run_hyderabad import render_frame
        out = str(tmp_path / "frame_stop.png")
        frame_data = {
            "step": 100,
            "vehicle_positions": [(130.0, 140.0), (150.0, 160.0)],
            "vehicle_speeds":    [0.0, 2.0],
            "tls_states":        {"J1": "rrrr"},
            "metrics": {"vehicles": 2, "avg_wait": 10.0, "queue": 2, "reward": -0.5},
        }
        render_frame(self._make_geom(), frame_data, out, total_steps=200)
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# make_gif
# ---------------------------------------------------------------------------

class TestMakeGif:
    def _make_geom(self):
        return {
            "junctions":     [("J1", 100.0, 100.0)],
            "edges":         [[(100.0, 100.0), (150.0, 150.0)]],
            "tls_junctions": {"J1"},
            "bbox":          (80.0, 170.0, 80.0, 170.0),
        }

    def test_creates_gif_from_frame_data(self, tmp_path):
        from simulation.run_hyderabad import make_gif
        frames = [
            {
                "step": i * 50,
                "vehicle_positions": [(100.0 + i * 5, 110.0)],
                "vehicle_speeds":    [8.0],
                "tls_states":        {"J1": "GrGr"},
                "metrics": {"vehicles": i * 3, "avg_wait": 0.1 * i,
                             "queue": i, "reward": 0.2},
            }
            for i in range(1, 4)
        ]
        gif_path = str(tmp_path / "test.gif")
        result = make_gif(
            frames=frames, gif_path=gif_path,
            net_geom=self._make_geom(), total_steps=150,
        )
        assert result is True
        assert Path(gif_path).exists()
        assert Path(gif_path).stat().st_size > 0

    def test_empty_frames_returns_false(self, tmp_path):
        from simulation.run_hyderabad import make_gif
        gif_path = str(tmp_path / "empty.gif")
        result = make_gif(frames=[], gif_path=gif_path)
        assert result is False
        assert not Path(gif_path).exists()

    def test_no_ffmpeg_returns_false(self, tmp_path):
        from simulation.run_hyderabad import make_gif
        frames = [{"step": 1, "vehicle_positions": [], "vehicle_speeds": [],
                   "tls_states": {}, "metrics": {}}]
        gif_path = str(tmp_path / "noffmpeg.gif")
        with patch("shutil.which", return_value=None):
            result = make_gif(frames=frames, gif_path=gif_path)
        assert result is False


# ---------------------------------------------------------------------------
# plot_metrics
# ---------------------------------------------------------------------------

class TestPlotMetrics:
    def _sample_metrics(self, n: int = 20) -> list:
        return [
            {
                "step":               i * 10,
                "vehicles_in_network": i * 5,
                "avg_wait_s":         float(i) * 0.1,
                "total_queue":        i,
                "total_reward":       0.3 - i * 0.01,
                "avg_speed_ms":       9.0,
                "arrived":            max(0, i - 10),
            }
            for i in range(1, n + 1)
        ]

    def test_creates_png(self, tmp_path):
        from simulation.run_hyderabad import plot_metrics
        png = str(tmp_path / "metrics.png")
        rewards = {"J1": 1.5, "J2": -0.3}
        plot_metrics(self._sample_metrics(), png, rewards)
        assert Path(png).exists()
        assert Path(png).stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path):
        from simulation.run_hyderabad import plot_metrics
        png = str(tmp_path / "deep" / "nested" / "metrics.png")
        plot_metrics(self._sample_metrics(), png, {"J1": 0.5})
        assert Path(png).exists()

    def test_empty_metrics_no_crash(self, tmp_path):
        from simulation.run_hyderabad import plot_metrics
        # Should log a warning and return without raising
        plot_metrics([], str(tmp_path / "empty.png"), {})

    def test_single_agent_reward(self, tmp_path):
        from simulation.run_hyderabad import plot_metrics
        png = str(tmp_path / "single.png")
        plot_metrics(self._sample_metrics(5), png, {"only_agent": 2.0})
        assert Path(png).exists()


# ---------------------------------------------------------------------------
# build_sumo_network (mocked)
# ---------------------------------------------------------------------------

class TestBuildSumoNetwork:
    def test_calls_convert_and_generate(self, tmp_path):
        from simulation import run_hyderabad as rh

        # build_sumo_network imports from simulation.osm_importer inside the function,
        # so we patch those names at their source module.
        with patch("simulation.osm_importer.convert_to_sumo",
                   return_value=str(tmp_path / "map.net.xml")) as mc, \
             patch("simulation.osm_importer.generate_routes",
                   return_value=str(tmp_path / "map.rou.xml")) as mr, \
             patch("simulation.osm_importer.generate_sumo_config",
                   return_value=str(tmp_path / "sim.sumocfg")) as cfg:
            rh.build_sumo_network(
                osm_file=str(tmp_path / "dummy.osm"),
                output_dir=str(tmp_path),
                num_vehicles=50,
                seed=7,
            )
            mc.assert_called_once()
            mr.assert_called_once()
            cfg.assert_called_once()


# ---------------------------------------------------------------------------
# _lane_exists
# ---------------------------------------------------------------------------

class TestLaneExists:
    def test_returns_true_when_lane_ok(self):
        from simulation.run_hyderabad import _lane_exists
        mock_traci = MagicMock()
        mock_traci.lane.getLength.return_value = 100.0
        assert _lane_exists(mock_traci, "lane_0") is True

    def test_returns_false_on_exception(self):
        from simulation.run_hyderabad import _lane_exists
        mock_traci = MagicMock()
        mock_traci.lane.getLength.side_effect = Exception("no lane")
        assert _lane_exists(mock_traci, "bad_lane") is False


# ---------------------------------------------------------------------------
# CLI smoke test (parse_args)
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self):
        from simulation.run_hyderabad import parse_args, DEFAULT_STEPS, DEFAULT_SEED
        with patch("sys.argv", ["run_hyderabad"]):
            args = parse_args()
        assert args.steps        == DEFAULT_STEPS
        assert args.seed         == DEFAULT_SEED
        assert args.gui          is False
        assert args.skip_convert is False

    def test_custom_steps(self):
        from simulation.run_hyderabad import parse_args
        with patch("sys.argv", ["run_hyderabad", "--steps", "200"]):
            args = parse_args()
        assert args.steps == 200

    def test_gui_flag(self):
        from simulation.run_hyderabad import parse_args
        with patch("sys.argv", ["run_hyderabad", "--gui"]):
            args = parse_args()
        assert args.gui is True

    def test_skip_convert_flag(self):
        from simulation.run_hyderabad import parse_args
        with patch("sys.argv", ["run_hyderabad", "--skip-convert"]):
            args = parse_args()
        assert args.skip_convert is True
