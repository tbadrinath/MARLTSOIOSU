/**
 * server.js
 * ---------
 * Express + Socket.io telemetry server for the Intelligent Urban Traffic
 * Management System (IUTMS).
 *
 * Responsibilities:
 *  1. Accept POST /api/metrics from the Python simulation (step-level data).
 *  2. Broadcast the metric payload to all connected dashboard clients via
 *     Socket.io (event: "step_metrics").
 *  3. Maintain an in-memory ring-buffer of the last MAX_HISTORY data points
 *     so that newly connected clients can hydrate their charts immediately
 *     (GET /api/metrics/history).
 *  4. Expose GET /api/status for health checks.
 */

"use strict";

const http     = require("http");
const https    = require("https");
const path     = require("path");
const { execFile } = require("child_process");
const os       = require("os");
const fs       = require("fs");

const cors = require("cors");
const express = require("express");
const { Server } = require("socket.io");

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const PORT = process.env.PORT || 3001;
const MAX_HISTORY = 500; // number of data points to keep in memory

// ---------------------------------------------------------------------------
// Application setup
// ---------------------------------------------------------------------------

const app = express();
const server = http.createServer(app);

const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"],
  },
});

app.use(cors());
app.use(express.json());

// ---------------------------------------------------------------------------
// In-memory ring-buffer for metric history (O(1) writes, O(k) reads)
// ---------------------------------------------------------------------------
//
// Performance note: the previous implementation used Array.push() + shift().
// Array.shift() is O(n) because every element must be moved one index left.
// At MAX_HISTORY = 500 entries this is negligible at low throughput, but under
// a high-frequency simulation the repeated O(n) shifts add up.  A circular
// ring-buffer keeps both writes and the full-capacity eviction at O(1).

/** @type {Array<Object|null>} */
const metricsRing = new Array(MAX_HISTORY).fill(null);
let ringHead  = 0;   // index of the *next* write slot
let ringCount = 0;   // number of valid entries currently stored

/**
 * Append one metrics payload to the ring-buffer (O(1)).
 * @param {Object} payload
 */
function appendMetric(payload) {
  metricsRing[ringHead] = { ...payload, serverTs: Date.now() };
  ringHead = (ringHead + 1) % MAX_HISTORY;
  if (ringCount < MAX_HISTORY) ringCount++;
}

/**
 * Return the most-recent `limit` entries in chronological order (O(k)).
 * @param {number} limit
 * @returns {Array<Object>}
 */
function getRecentHistory(limit) {
  const count  = Math.min(limit, ringCount);
  const result = new Array(count);
  // Oldest retained entry sits at (ringHead - ringCount) mod MAX_HISTORY.
  // We want the last `count` entries, so start from (ringHead - count).
  const start = (ringHead - count + MAX_HISTORY) % MAX_HISTORY;
  for (let i = 0; i < count; i++) {
    result[i] = metricsRing[(start + i) % MAX_HISTORY];
  }
  return result;
}

// ---------------------------------------------------------------------------
// REST endpoints
// ---------------------------------------------------------------------------

/**
 * POST /api/metrics
 * -----------------
 * Receive a step_metrics payload from the Python simulation.
 *
 * Expected body (all fields optional except "step"):
 * {
 *   "episode":             <number>,
 *   "step":                <number>,
 *   "avg_speed":           <number>,   // m/s
 *   "co2_emissions":       <number>,   // mg/s
 *   "total_reward":        <number>,
 *   "vehicles_in_network": <number>
 * }
 */
app.post("/api/metrics", (req, res) => {
  const payload = req.body;

  if (!payload || payload.step === undefined) {
    return res.status(400).json({ error: "Missing required field: step" });
  }

  // Validate that all numeric fields are actually numbers
  const numericFields = [
    "episode",
    "step",
    "avg_speed",
    "co2_emissions",
    "total_reward",
    "vehicles_in_network",
  ];
  for (const field of numericFields) {
    if (payload[field] !== undefined && typeof payload[field] !== "number") {
      return res
        .status(400)
        .json({ error: `Field "${field}" must be a number` });
    }
  }

  appendMetric(payload);
  io.emit("step_metrics", { ...payload, serverTs: Date.now() });

  return res.status(200).json({ ok: true });
});

/**
 * GET /api/metrics/history
 * ------------------------
 * Return the buffered history so that newly connected clients can populate
 * their charts without waiting for new events.
 *
 * Query params:
 *   limit  – max number of records to return (default: MAX_HISTORY)
 */
app.get("/api/metrics/history", (req, res) => {
  const limit = Math.min(
    parseInt(req.query.limit, 10) || MAX_HISTORY,
    MAX_HISTORY
  );
  const data = getRecentHistory(limit);
  return res.json({ count: data.length, data });
});

/**
 * GET /api/status
 * ---------------
 * Health-check endpoint.
 */
app.get("/api/status", (_req, res) => {
  res.json({
    status: "ok",
    uptime: process.uptime(),
    metricsBuffered: ringCount,
    connectedClients: io.engine.clientsCount,
  });
});

// ---------------------------------------------------------------------------
// OSM / Map endpoints
// ---------------------------------------------------------------------------

/**
 * Internal helper: perform an HTTPS GET and return the response body as a
 * parsed JSON object.  Uses Node's built-in `https` module so no extra
 * dependency is needed.
 *
 * @param {string} url  Full URL including query string.
 * @returns {Promise<any>}
 */
function httpsGetJson(url) {
  return new Promise((resolve, reject) => {
    const req = https.get(
      url,
      { headers: { "User-Agent": "IUTMS-TrafficSim/1.0 (github.com/tbadrinath/MARLTSOIOSU)" } },
      (res) => {
        let data = "";
        res.on("data", (chunk) => { data += chunk; });
        res.on("end", () => {
          try { resolve(JSON.parse(data)); }
          catch (e) { reject(new Error("Invalid JSON from upstream: " + e.message)); }
        });
      }
    );
    req.on("error", reject);
    req.setTimeout(15000, () => { req.destroy(new Error("Request timed out")); });
  });
}

/**
 * GET /api/osm/search?q=<location>&limit=<n>
 * -------------------------------------------
 * Proxy to the Nominatim geocoding API.  Proxying avoids browser CORS
 * restrictions and ensures the correct User-Agent is sent as required by
 * the Nominatim usage policy.
 *
 * Query params:
 *   q     – free-form location string (required)
 *   limit – number of results to return, 1–10 (default: 5)
 */
app.get("/api/osm/search", async (req, res) => {
  const query = (req.query.q || "").trim();
  if (!query) {
    return res.status(400).json({ error: "Missing query parameter: q" });
  }
  const limit = Math.min(parseInt(req.query.limit, 10) || 5, 10);

  try {
    const params = new URLSearchParams({
      q:      query,
      format: "json",
      limit:  String(limit),
    });
    const url = `https://nominatim.openstreetmap.org/search?${params}`;
    const data = await httpsGetJson(url);

    const results = (Array.isArray(data) ? data : []).map((r) => ({
      display_name: r.display_name || "",
      lat:          parseFloat(r.lat) || 0,
      lon:          parseFloat(r.lon) || 0,
      boundingbox:  r.boundingbox || [],
      osm_type:     r.osm_type || "",
      osm_id:       r.osm_id   || "",
    }));

    return res.json({ count: results.length, results });
  } catch (err) {
    console.error("[OSM search] Error:", err.message);
    return res.status(502).json({ error: "Nominatim request failed: " + err.message });
  }
});

/**
 * POST /api/osm/import
 * --------------------
 * Download OSM data and convert it to a SUMO network by invoking the
 * Python `simulation/osm_importer.py` pipeline.
 *
 * Request body (JSON):
 * {
 *   "location":      <string>,   // human-readable city/place name
 *   "num_vehicles":  <number>,   // optional, default 400
 *   "seed":          <number>    // optional, default 42
 * }
 *
 * Response (JSON):
 * {
 *   "display_name": <string>,
 *   "net_file":     <string>,   // absolute path to .net.xml
 *   "route_file":   <string>,   // absolute path to .rou.xml
 *   "bbox":         [minLat, maxLat, minLon, maxLon]
 * }
 */
app.post("/api/osm/import", (req, res) => {
  const { location, num_vehicles = 400, seed = 42 } = req.body || {};

  if (!location || typeof location !== "string" || !location.trim()) {
    return res.status(400).json({ error: "Missing required field: location" });
  }

  // Resolve path to the Python helper script relative to the repo root.
  // server.js lives in  web/server/; repo root is two levels up.
  const repoRoot = path.resolve(__dirname, "..", "..");
  const scriptPath = path.join(repoRoot, "simulation", "osm_importer.py");

  // Output directory: <repo>/maps/osm/<sanitised-location>/
  const sanitised = location.trim().replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
  const outputDir  = path.join(repoRoot, "maps", "osm", sanitised);

  const pythonCode = [
    "import sys, json",
    `sys.path.insert(0, ${JSON.stringify(repoRoot)})`,
    "from simulation.osm_importer import import_map",
    `result = import_map(${JSON.stringify(location)}, ${JSON.stringify(outputDir)}, num_vehicles=${num_vehicles}, seed=${seed})`,
    "print(json.dumps(result))",
  ].join("; ");

  const python = process.env.PYTHON_BIN || "python3";

  execFile(
    python,
    ["-c", pythonCode],
    { timeout: 180_000 },   // 3-minute timeout for large city downloads
    (err, stdout, stderr) => {
      if (err) {
        console.error("[OSM import] Python error:", stderr || err.message);
        return res.status(500).json({
          error: "OSM import failed",
          detail: (stderr || err.message || "").slice(0, 1000),
        });
      }

      let result;
      try {
        result = JSON.parse(stdout.trim());
      } catch (parseErr) {
        return res.status(500).json({
          error: "Failed to parse Python output",
          detail: stdout.slice(0, 500),
        });
      }

      // Broadcast the new map info to all connected dashboard clients
      io.emit("osm_import_complete", result);

      return res.json(result);
    }
  );
});

// ---------------------------------------------------------------------------
// Socket.io connection handling
// ---------------------------------------------------------------------------

io.on("connection", (socket) => {
  console.log(`[Socket.io] Client connected: ${socket.id}`);

  // Send recent history to the newly connected client
  const history = getRecentHistory(MAX_HISTORY);
  socket.emit("history", {
    count: history.length,
    data: history,
  });

  socket.on("disconnect", () => {
    console.log(`[Socket.io] Client disconnected: ${socket.id}`);
  });
});

// ---------------------------------------------------------------------------
// Start server
// ---------------------------------------------------------------------------

server.listen(PORT, () => {
  console.log(`IUTMS telemetry server running on http://localhost:${PORT}`);
  console.log(`  POST /api/metrics          – ingest step metrics`);
  console.log(`  GET  /api/metrics/history  – fetch metric history`);
  console.log(`  GET  /api/status           – health check`);
  console.log(`  GET  /api/osm/search       – geocode a location (Nominatim)`);
  console.log(`  POST /api/osm/import       – download OSM map & convert to SUMO`);
  console.log(`  Socket.io on ws://localhost:${PORT}`);
});

module.exports = { app, server, io };
