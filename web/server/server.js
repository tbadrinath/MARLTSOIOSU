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
const rateLimit = require("express-rate-limit");
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
const REPO_ROOT = path.resolve(__dirname, "..", "..");
const PYTHON_BIN = process.env.PYTHON_BIN || "python3";
const REPO_NAME = path.basename(REPO_ROOT);

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
    demoRunning: _demoTimer !== null,
  });
});

function removeTempDir(dirPath) {
  fs.rm(dirPath, { recursive: true, force: true }, (err) => {
    if (err) {
      console.warn("[Codebase export] Cleanup warning for", dirPath, ":", err.message);
    }
  });
}

function parseEnvInt(value, fallback) {
  const parsed = parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function buildCodebaseArchive(outputPath, callback) {
  execFile(
    PYTHON_BIN,
    ["-m", "simulation.codebase_exporter", "--output", outputPath, "--repo-root", REPO_ROOT],
    {
      cwd: REPO_ROOT,
      timeout: parseEnvInt(process.env.CODEBASE_EXPORT_TIMEOUT_MS, 120_000),
    },
    callback
  );
}

const codebaseExportRateLimiter = rateLimit({
  windowMs: parseEnvInt(process.env.CODEBASE_EXPORT_RATE_WINDOW_MS, 60_000),
  limit: parseEnvInt(process.env.CODEBASE_EXPORT_RATE_LIMIT, 5),
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: "Too many export requests. Please wait before downloading another archive.",
  },
});

/**
 * GET /api/export/codebase
 * ------------------------
 * Package the repository source into a downloadable zip archive.
 */
app.get("/api/export/codebase", codebaseExportRateLimiter, (_req, res) => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "iutms-export-"));
  const archiveName = `${REPO_NAME}-codebase-${new Date().toISOString().replace(/[:.]/g, "-")}.zip`;
  const archivePath = path.join(tempDir, archiveName);

  buildCodebaseArchive(archivePath, (err, _stdout, stderr) => {
    if (err) {
      console.error("[Codebase export] Error:", stderr || err.message);
      removeTempDir(tempDir);
      return res.status(500).json({
        error: "Codebase export failed. Please try again.",
      });
    }

    return res.download(archivePath, archiveName, (downloadErr) => {
      if (downloadErr) {
        console.error("[Codebase export] Download error:", downloadErr.message);
      }
      removeTempDir(tempDir);
    });
  });
});

// ---------------------------------------------------------------------------
// Server-side autonomous demo simulation
// ---------------------------------------------------------------------------
//
// POST /api/demo/start  – begin pushing synthetic traffic metrics to all
//                         connected Socket.io clients at a fixed interval.
// POST /api/demo/stop   – stop the autonomous simulation.
// GET  /api/demo/status – check whether the demo is currently running.
//
// The generated data mirrors the client-side generateDemoMetric() function so
// that the dashboard displays realistic, smoothly-improving RL training curves
// without needing SUMO or any Python environment.

const DEMO_INTERVAL_MS = 150;  // ms between emitted data-points
const DEMO_STEP_SIZE   = 10;   // simulation steps per tick
const DEMO_MAX_STEPS   = 3600; // steps per episode
const DEMO_TOTAL_EP    = 50;   // training episodes

let _demoTimer   = null;
let _demoStep    = 0;
let _demoEpisode = 1;

/**
 * Generate one synthetic step-metrics payload that mimics realistic MARL
 * training progress (reward improves, congestion drops, speed rises).
 */
function generateServerDemoMetric(step, episode) {
  const epProgress    = Math.min((episode - 1) / DEMO_TOTAL_EP, 1);
  const phaseProgress = step / DEMO_MAX_STEPS;

  // Reward improves from roughly -12 → +8 across training episodes
  const baseReward    = -12 + epProgress * 20;
  const totalReward   = +(baseReward + Math.sin(phaseProgress * Math.PI * 4) * 2
                          + (Math.random() - 0.5) * 3).toFixed(3);

  // Average speed rises from ~6 m/s to ~14 m/s as agents learn
  const avgSpeed      = +(6 + epProgress * 8
                          + Math.sin(phaseProgress * Math.PI * 3) * 1.5
                          + (Math.random() - 0.5) * 0.8).toFixed(2);

  // Vehicles in network (congestion) drops from ~120 to ~40
  const vehiclesInNetwork = Math.round(
    120 - epProgress * 80
    - Math.sin(phaseProgress * Math.PI * 2) * 10
    + (Math.random() - 0.5) * 15
  );

  // CO₂ emissions fall as traffic flows more efficiently
  const co2Emissions  = +(800 - epProgress * 400
                           + Math.sin(phaseProgress * Math.PI * 2) * 60
                           + (Math.random() - 0.5) * 40).toFixed(1);

  return {
    step,
    episode,
    total_reward:        totalReward,
    avg_speed:           avgSpeed,
    vehicles_in_network: vehiclesInNetwork,
    co2_emissions:       co2Emissions,
  };
}

/** Start the server-side demo timer (idempotent). */
function startServerDemo() {
  if (_demoTimer) return; // already running
  _demoStep    = 0;
  _demoEpisode = 1;
  _demoTimer = setInterval(() => {
    const metric = generateServerDemoMetric(_demoStep, _demoEpisode);
    appendMetric(metric);
    io.emit("step_metrics", { ...metric, serverTs: Date.now() });

    _demoStep += DEMO_STEP_SIZE;
    if (_demoStep >= DEMO_MAX_STEPS) {
      _demoStep = 0;
      // Cycle back to episode 1 after completing all training episodes so the
      // demo loops indefinitely with the same improving-reward trajectory.
      _demoEpisode = (_demoEpisode >= DEMO_TOTAL_EP) ? 1 : _demoEpisode + 1;
    }
  }, DEMO_INTERVAL_MS);
  console.log("[Demo] Server-side autonomous demo started.");
}

/** Stop the server-side demo timer (idempotent). */
function stopServerDemo() {
  if (!_demoTimer) return;
  clearInterval(_demoTimer);
  _demoTimer = null;
  console.log("[Demo] Server-side autonomous demo stopped.");
}

app.post("/api/demo/start", (_req, res) => {
  startServerDemo();
  res.json({ ok: true, message: "Demo simulation started." });
});

app.post("/api/demo/stop", (_req, res) => {
  stopServerDemo();
  res.json({ ok: true, message: "Demo simulation stopped." });
});

app.get("/api/demo/status", (_req, res) => {
  res.json({
    running:  _demoTimer !== null,
    episode:  _demoEpisode,
    step:     _demoStep,
  });
});

// ---------------------------------------------------------------------------
// OSM / Map endpoints
// ---------------------------------------------------------------------------

/**
 * Simple in-memory rate limiter for the OSM import endpoint.
 *
 * Each unique IP may trigger at most OSM_RATE_LIMIT requests within
 * OSM_RATE_WINDOW_MS milliseconds.  This prevents abuse of the endpoint that
 * spawns a child process and makes external HTTP calls.
 */
const OSM_RATE_LIMIT      = parseInt(process.env.OSM_RATE_LIMIT,      10) || 5;
const OSM_RATE_WINDOW_MS  = parseInt(process.env.OSM_RATE_WINDOW_MS,  10) || 60_000;

/** @type {Map<string, {count: number, resetAt: number}>} */
const _osmRateMap = new Map();

function osmRateLimiter(req, res, next) {
  const ip  = req.ip || req.socket.remoteAddress || "unknown";
  const now = Date.now();
  let entry = _osmRateMap.get(ip);

  if (!entry || now >= entry.resetAt) {
    entry = { count: 0, resetAt: now + OSM_RATE_WINDOW_MS };
    _osmRateMap.set(ip, entry);
  }

  entry.count += 1;
  if (entry.count > OSM_RATE_LIMIT) {
    const retryAfterSec = Math.ceil((entry.resetAt - now) / 1000);
    res.set("Retry-After", String(retryAfterSec));
    return res
      .status(429)
      .json({ error: "Too many requests. Please wait before importing another map." });
  }

  next();
}

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
      { headers: { "User-Agent": "IUTMS-TrafficSim/1.0 (https://github.com/tbadrinath/MARLTSOIOSU)" } },
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
app.post("/api/osm/import", osmRateLimiter, (req, res) => {
  const { location, num_vehicles = 400, seed = 42 } = req.body || {};

  if (!location || typeof location !== "string" || !location.trim()) {
    return res.status(400).json({ error: "Missing required field: location" });
  }

  // ── Rate limit check (belt-and-suspenders guard alongside the middleware) ──
  const clientIp  = req.ip || req.socket.remoteAddress || "unknown";
  const now       = Date.now();
  let   rlEntry   = _osmRateMap.get(clientIp);
  if (!rlEntry || now >= rlEntry.resetAt) {
    rlEntry = { count: 0, resetAt: now + OSM_RATE_WINDOW_MS };
    _osmRateMap.set(clientIp, rlEntry);
  }
  if (rlEntry.count > OSM_RATE_LIMIT) {
    const retryAfterSec = Math.ceil((rlEntry.resetAt - now) / 1000);
    res.set("Retry-After", String(retryAfterSec));
    return res.status(429).json({ error: "Too many requests. Please wait before importing another map." });
  }

  // Output directory: <repo>/maps/osm/<sanitised-location>/
  const sanitised = location.trim().replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
  const outputDir  = path.join(REPO_ROOT, "maps", "osm", sanitised);

  const pythonCode = [
    "import sys, json",
    `sys.path.insert(0, ${JSON.stringify(REPO_ROOT)})`,
    "from simulation.osm_importer import import_map",
    `result = import_map(${JSON.stringify(location)}, ${JSON.stringify(outputDir)}, num_vehicles=${num_vehicles}, seed=${seed})`,
    "print(json.dumps(result))",
  ].join("; ");

  execFile(
    PYTHON_BIN,
    ["-c", pythonCode],
    { timeout: parseInt(process.env.OSM_IMPORT_TIMEOUT_MS, 10) || 180_000 },   // default 3-min; override with OSM_IMPORT_TIMEOUT_MS
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
// Static file serving – serve the pre-built React app from /api/* fallthrough
// ---------------------------------------------------------------------------
//
// When the React app has been built (`cd web/client && npm run build`), the
// server can serve it directly so the entire stack runs on a single port.
// This is optional – the dev server (`npm start` in web/client) still works.

const CLIENT_BUILD    = path.join(__dirname, "..", "client", "build");
// Pre-resolve once so the route handler never derives a path from user input.
const CLIENT_INDEX    = path.join(CLIENT_BUILD, "index.html");

if (fs.existsSync(CLIENT_BUILD)) {
  app.use(express.static(CLIENT_BUILD));

  // Read index.html once at startup so the catch-all route serves it from
  // memory rather than performing a file system access per request.
  // The path is a compile-time constant derived from __dirname – never from
  // request data – so there is no path-traversal risk.
  const indexHtml = fs.existsSync(CLIENT_INDEX)
    ? fs.readFileSync(CLIENT_INDEX)
    : null;

  if (indexHtml) {
    // Any non-API route falls through to index.html (client-side routing).
    app.get(/^(?!\/api\/).*/, (_req, res) => {
      res.type("html").send(indexHtml);
    });
  }

  console.log(`[Static] Serving React build from ${CLIENT_BUILD}`);
}

// ---------------------------------------------------------------------------
// Start server
// ---------------------------------------------------------------------------

server.listen(PORT, () => {
  console.log(`IUTMS telemetry server running on http://localhost:${PORT}`);
  console.log(`  POST /api/metrics          – ingest step metrics`);
  console.log(`  GET  /api/metrics/history  – fetch metric history`);
  console.log(`  GET  /api/status           – health check`);
  console.log(`  GET  /api/export/codebase  – download the full project as zip`);
  console.log(`  GET  /api/osm/search       – geocode a location (Nominatim)`);
  console.log(`  POST /api/osm/import       – download OSM map & convert to SUMO`);
  console.log(`  POST /api/demo/start       – start autonomous demo simulation`);
  console.log(`  POST /api/demo/stop        – stop autonomous demo simulation`);
  console.log(`  GET  /api/demo/status      – demo simulation status`);
  console.log(`  Socket.io on ws://localhost:${PORT}`);

  // Auto-start the server-side demo when DEMO=true is set in the environment.
  // This lets CI/CD or the README quick-start command spin up a fully live
  // demonstration without any additional steps.
  if (process.env.DEMO === "true") {
    startServerDemo();
    console.log(`[Demo] Auto-started (DEMO=true). Open http://localhost:${PORT}/?demo=true`);
  }
});

module.exports = { app, server, io, startServerDemo, stopServerDemo };
