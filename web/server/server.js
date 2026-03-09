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

const http = require("http");
const path = require("path");

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
  console.log(`  Socket.io on ws://localhost:${PORT}`);
});

module.exports = { app, server, io };
