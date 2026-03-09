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
// In-memory ring-buffer for metric history
// ---------------------------------------------------------------------------

/** @type {Array<Object>} */
const metricsHistory = [];

function appendMetric(payload) {
  metricsHistory.push({ ...payload, serverTs: Date.now() });
  if (metricsHistory.length > MAX_HISTORY) {
    metricsHistory.shift();
  }
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
  const data = metricsHistory.slice(-limit);
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
    metricsBuffered: metricsHistory.length,
    connectedClients: io.engine.clientsCount,
  });
});

// ---------------------------------------------------------------------------
// Socket.io connection handling
// ---------------------------------------------------------------------------

io.on("connection", (socket) => {
  console.log(`[Socket.io] Client connected: ${socket.id}`);

  // Send recent history to the newly connected client
  socket.emit("history", {
    count: metricsHistory.length,
    data: metricsHistory.slice(-MAX_HISTORY),
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
