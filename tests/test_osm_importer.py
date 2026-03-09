"""
tests/test_osm_importer.py
--------------------------
Unit tests for simulation/osm_importer.py.

All external HTTP calls and subprocess invocations are mocked so that:
  • No real network requests are made.
  • SUMO tools do not need to be installed.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure the repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.osm_importer import (
    _clamp_bbox,
    _find_sumo_tool,
    _write_synthetic_routes,
    search_location,
    download_osm,
    convert_to_sumo,
    generate_routes,
    import_map,
    BBOX_MARGIN,
    MAX_BBOX_SIDE,
)


# ---------------------------------------------------------------------------
# _clamp_bbox
# ---------------------------------------------------------------------------

class TestClampBbox:
    def test_margin_applied(self):
        min_lat, max_lat, min_lon, max_lon = _clamp_bbox(10.0, 10.02, 20.0, 20.02)
        assert min_lat < 10.0
        assert max_lat > 10.02
        assert min_lon < 20.0
        assert max_lon > 20.02

    def test_lat_clamped_to_max_side(self):
        # Very tall bbox – lat span should be capped at MAX_BBOX_SIDE
        min_lat, max_lat, _, _ = _clamp_bbox(0.0, 5.0, 10.0, 10.01)
        assert (max_lat - min_lat) <= MAX_BBOX_SIDE + 1e-9

    def test_lon_clamped_to_max_side(self):
        # Very wide bbox – lon span should be capped at MAX_BBOX_SIDE
        _, _, min_lon, max_lon = _clamp_bbox(10.0, 10.01, 0.0, 5.0)
        assert (max_lon - min_lon) <= MAX_BBOX_SIDE + 1e-9

    def test_small_bbox_passthrough(self):
        # A bbox smaller than MAX_BBOX_SIDE – only margin is added
        min_lat, max_lat, min_lon, max_lon = _clamp_bbox(10.0, 10.01, 20.0, 20.01)
        expected_span = 0.01 + 2 * BBOX_MARGIN
        assert abs((max_lat - min_lat) - expected_span) < 1e-9
        assert abs((max_lon - min_lon) - expected_span) < 1e-9


# ---------------------------------------------------------------------------
# _write_synthetic_routes
# ---------------------------------------------------------------------------

class TestWriteSyntheticRoutes:
    def test_file_is_valid_xml(self, tmp_path):
        route_file = str(tmp_path / "routes.rou.xml")
        _write_synthetic_routes(route_file, num_vehicles=5)
        content = Path(route_file).read_text()
        assert content.startswith('<?xml')
        assert '<routes' in content
        assert '</routes>' in content

    def test_vehicle_count(self, tmp_path):
        route_file = str(tmp_path / "routes.rou.xml")
        _write_synthetic_routes(route_file, num_vehicles=3)
        content = Path(route_file).read_text()
        assert content.count('<vehicle') == 3

    def test_creates_parent_dirs(self, tmp_path):
        route_file = str(tmp_path / "deep" / "dir" / "routes.rou.xml")
        Path(route_file).parent.mkdir(parents=True, exist_ok=True)
        _write_synthetic_routes(route_file, num_vehicles=1)
        assert Path(route_file).exists()


# ---------------------------------------------------------------------------
# search_location
# ---------------------------------------------------------------------------

class TestSearchLocation:
    def _nominatim_response(self):
        return [
            {
                "display_name": "Manhattan, New York, USA",
                "lat": "40.7831",
                "lon": "-73.9712",
                "boundingbox": ["40.6960", "40.8820", "-74.0479", "-73.9067"],
                "osm_type": "relation",
                "osm_id": "8398124",
            }
        ]

    def test_returns_list_of_dicts(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._nominatim_response()
        mock_resp.raise_for_status = MagicMock()

        with patch("simulation.osm_importer._get", return_value=mock_resp):
            results = search_location("Manhattan, New York")

        assert isinstance(results, list)
        assert len(results) == 1

    def test_result_fields(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._nominatim_response()
        mock_resp.raise_for_status = MagicMock()

        with patch("simulation.osm_importer._get", return_value=mock_resp):
            results = search_location("Manhattan")

        r = results[0]
        assert "display_name" in r
        assert "lat" in r and isinstance(r["lat"], float)
        assert "lon" in r and isinstance(r["lon"], float)
        assert "boundingbox" in r

    def test_empty_response_raises(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()

        with patch("simulation.osm_importer._get", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="No results found"):
                search_location("xyzzy-nonexistent-place")

    def test_limit_clamp(self):
        """Limit should be clamped between 1 and 10."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._nominatim_response()
        mock_resp.raise_for_status = MagicMock()

        with patch("simulation.osm_importer._get", return_value=mock_resp) as mock_get:
            search_location("test", limit=99)
            _, kwargs_or_args = mock_get.call_args[0], mock_get.call_args
            # The 'limit' in params should not exceed 10
            params = mock_get.call_args[0][1]
            assert int(params["limit"]) <= 10


# ---------------------------------------------------------------------------
# download_osm
# ---------------------------------------------------------------------------

class TestDownloadOsm:
    def test_writes_file(self, tmp_path):
        osm_data = b"<osm><node id='1' lat='10.0' lon='20.0'/></osm>"
        mock_resp = MagicMock()
        mock_resp.content = osm_data
        mock_resp.raise_for_status = MagicMock()

        out_file = str(tmp_path / "map.osm")
        with patch("simulation.osm_importer._get", return_value=mock_resp):
            result_path = download_osm(10.0, 10.05, 20.0, 20.05, out_file)

        assert Path(result_path).exists()
        assert Path(result_path).read_bytes() == osm_data

    def test_creates_parent_dir(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.content = b"<osm/>"
        mock_resp.raise_for_status = MagicMock()

        out_file = str(tmp_path / "nested" / "dir" / "map.osm")
        with patch("simulation.osm_importer._get", return_value=mock_resp):
            download_osm(10.0, 10.01, 20.0, 20.01, out_file)

        assert Path(out_file).exists()


# ---------------------------------------------------------------------------
# convert_to_sumo
# ---------------------------------------------------------------------------

class TestConvertToSumo:
    def test_calls_netconvert(self, tmp_path):
        osm_file = str(tmp_path / "map.osm")
        Path(osm_file).write_text("<osm/>")
        net_file = str(tmp_path / "map.net.xml")

        fake_netconvert = str(tmp_path / "netconvert")
        Path(fake_netconvert).write_text("")   # just needs to exist

        with patch("simulation.osm_importer._find_sumo_tool", return_value=fake_netconvert):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
                result = convert_to_sumo(osm_file, net_file)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert fake_netconvert in cmd
        assert "--osm-files" in cmd
        assert osm_file in cmd

    def test_raises_when_netconvert_missing(self, tmp_path):
        with patch("simulation.osm_importer._find_sumo_tool", return_value=None):
            with pytest.raises(RuntimeError, match="netconvert"):
                convert_to_sumo("map.osm", str(tmp_path / "map.net.xml"))

    def test_raises_on_nonzero_exit(self, tmp_path):
        osm_file = str(tmp_path / "map.osm")
        Path(osm_file).write_text("<osm/>")

        with patch("simulation.osm_importer._find_sumo_tool", return_value="/usr/bin/netconvert"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=1, stderr="some error", stdout=""
                )
                with pytest.raises(RuntimeError, match="netconvert failed"):
                    convert_to_sumo(osm_file, str(tmp_path / "out.net.xml"))


# ---------------------------------------------------------------------------
# generate_routes
# ---------------------------------------------------------------------------

class TestGenerateRoutes:
    def test_fallback_synthetic_when_no_tools(self, tmp_path):
        net_file   = str(tmp_path / "map.net.xml")
        route_file = str(tmp_path / "map.rou.xml")
        Path(net_file).write_text("<net/>")

        with patch("simulation.osm_importer._find_sumo_tool", return_value=None):
            result = generate_routes(net_file, route_file, num_vehicles=3)

        assert Path(result).exists()
        content = Path(result).read_text()
        assert "<routes" in content
        assert content.count("<vehicle") == 3

    def test_uses_randomtrips_when_available(self, tmp_path):
        net_file    = str(tmp_path / "map.net.xml")
        route_file  = str(tmp_path / "map.rou.xml")
        trips_file  = route_file.replace(".rou.xml", ".trips.xml")
        Path(net_file).write_text("<net/>")
        # Simulate duarouter writing the route file
        def fake_run(cmd, **kwargs):
            if "duarouter" in str(cmd):
                Path(route_file).write_text("<routes/>")
            return MagicMock(returncode=0, stderr="", stdout="")

        with patch(
            "simulation.osm_importer._find_sumo_tool",
            side_effect=lambda t: "/fake/randomTrips.py" if "random" in t else "/fake/duarouter",
        ):
            with patch("subprocess.run", side_effect=fake_run):
                result = generate_routes(net_file, route_file, num_vehicles=10)

        assert Path(result).exists()


# ---------------------------------------------------------------------------
# import_map (integration – all sub-steps mocked)
# ---------------------------------------------------------------------------

class TestImportMap:
    def test_full_pipeline(self, tmp_path):
        location = "Test City, Testland"
        output_dir = str(tmp_path / "output")

        nominatim_result = [{
            "display_name": "Test City, Testland",
            "lat": 12.34,
            "lon": 56.78,
            "boundingbox": ["12.30", "12.38", "56.74", "56.82"],
            "osm_type": "relation",
            "osm_id": "9999",
        }]

        with (
            patch("simulation.osm_importer.search_location", return_value=nominatim_result),
            patch("simulation.osm_importer.download_osm",    return_value=str(tmp_path / "map.osm")),
            patch("simulation.osm_importer.convert_to_sumo", return_value=str(tmp_path / "map.net.xml")),
            patch("simulation.osm_importer.generate_routes", return_value=str(tmp_path / "map.rou.xml")),
        ):
            result = import_map(location, output_dir)

        assert "display_name" in result
        assert "net_file" in result
        assert "route_file" in result
        assert "bbox" in result
        assert len(result["bbox"]) == 4

    def test_pipeline_propagates_search_error(self, tmp_path):
        with patch(
            "simulation.osm_importer.search_location",
            side_effect=RuntimeError("No results found"),
        ):
            with pytest.raises(RuntimeError, match="No results found"):
                import_map("xyzzy", str(tmp_path / "output"))
