/**
 * App.js
 * ------
 * Root component for the IUTMS real-time dashboard.
 *
 * Connects to the Socket.io server and renders live charts for:
 *   • Total reward per step
 *   • Average vehicle speed (m/s)
 *   • Congestion Index (vehicles in network)
 *   • CO₂ emissions (mg/s)
 *
 * A summary KPI row shows the latest values for quick at-a-glance
 * monitoring.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Filler,
} from "chart.js";
import { Line } from "react-chartjs-2";
import { io } from "socket.io-client";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const SERVER_URL = process.env.REACT_APP_SERVER_URL || "http://localhost:3001";
const MAX_POINTS = 200; // max data points to keep per chart series

// ---------------------------------------------------------------------------
// Chart colour palette
// ---------------------------------------------------------------------------

const COLORS = {
  reward:    { border: "#00e5ff", bg: "rgba(0,229,255,0.12)" },
  speed:     { border: "#69ff47", bg: "rgba(105,255,71,0.12)" },
  congestion:{ border: "#ff9800", bg: "rgba(255,152,0,0.12)" },
  co2:       { border: "#ff4d6d", bg: "rgba(255,77,109,0.12)" },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeDataset(label, color, data) {
  return {
    label,
    data,
    borderColor: color.border,
    backgroundColor: color.bg,
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.3,
    fill: true,
  };
}

function chartOptions(title, yLabel) {
  return {
    responsive: true,
    animation: false,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: title,
        color: "#c0c0d0",
        font: { size: 13, weight: "600" },
      },
      tooltip: {
        mode: "index",
        intersect: false,
        backgroundColor: "rgba(20,20,40,0.9)",
        titleColor: "#fff",
        bodyColor: "#c0c0d0",
      },
    },
    scales: {
      x: {
        ticks: { color: "#666", maxTicksLimit: 8 },
        grid: { color: "rgba(255,255,255,0.05)" },
      },
      y: {
        title: { display: true, text: yLabel, color: "#888", font: { size: 11 } },
        ticks: { color: "#666" },
        grid: { color: "rgba(255,255,255,0.05)" },
      },
    },
  };
}

// ---------------------------------------------------------------------------
// KPI card
// ---------------------------------------------------------------------------

function KpiCard({ label, value, unit, color }) {
  return (
    <div style={styles.kpiCard}>
      <div style={{ ...styles.kpiValue, color }}>{value}</div>
      <div style={styles.kpiLabel}>
        {label}
        {unit && <span style={styles.kpiUnit}> {unit}</span>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main App component
// ---------------------------------------------------------------------------

export default function App() {
  const socketRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [episode, setEpisode] = useState(null);

  // Rolling chart data
  const [steps,      setSteps]      = useState([]);
  const [rewards,    setRewards]    = useState([]);
  const [speeds,     setSpeeds]     = useState([]);
  const [congestion, setCongestion] = useState([]);
  const [co2,        setCo2]        = useState([]);

  // Latest KPI values
  const [latestReward,    setLatestReward]    = useState("—");
  const [latestSpeed,     setLatestSpeed]     = useState("—");
  const [latestCongestion,setLatestCongestion]= useState("—");
  const [latestCo2,       setLatestCo2]       = useState("—");

  // ------------------------------------------------------------------
  // Process incoming metric payload
  // ------------------------------------------------------------------
  const handleMetric = useCallback((m) => {
    const step  = m.step  ?? 0;
    const rw    = typeof m.total_reward        === "number" ? +m.total_reward.toFixed(3)        : null;
    const spd   = typeof m.avg_speed           === "number" ? +m.avg_speed.toFixed(2)           : null;
    const cong  = typeof m.vehicles_in_network === "number" ? m.vehicles_in_network              : null;
    const emission = typeof m.co2_emissions    === "number" ? +m.co2_emissions.toFixed(1)       : null;

    if (m.episode != null) setEpisode(m.episode);

    setSteps      ((prev) => [...prev.slice(-MAX_POINTS + 1), step]);
    setRewards    ((prev) => [...prev.slice(-MAX_POINTS + 1), rw]);
    setSpeeds     ((prev) => [...prev.slice(-MAX_POINTS + 1), spd]);
    setCongestion ((prev) => [...prev.slice(-MAX_POINTS + 1), cong]);
    setCo2        ((prev) => [...prev.slice(-MAX_POINTS + 1), emission]);

    if (rw    != null) setLatestReward    (rw);
    if (spd   != null) setLatestSpeed     (spd);
    if (cong  != null) setLatestCongestion(cong);
    if (emission != null) setLatestCo2    (emission);
  }, []);

  // ------------------------------------------------------------------
  // Socket.io connection
  // ------------------------------------------------------------------
  useEffect(() => {
    const socket = io(SERVER_URL, { transports: ["websocket", "polling"] });
    socketRef.current = socket;

    socket.on("connect", () => setConnected(true));
    socket.on("disconnect", () => setConnected(false));

    // Hydrate charts from server-side history buffer
    socket.on("history", ({ data }) => {
      if (!Array.isArray(data)) return;
      data.forEach(handleMetric);
    });

    socket.on("step_metrics", handleMetric);

    return () => {
      socket.disconnect();
    };
  }, [handleMetric]);

  // ------------------------------------------------------------------
  // Chart datasets
  // ------------------------------------------------------------------
  const labels = steps.map(String);

  const rewardData = {
    labels,
    datasets: [makeDataset("Total Reward", COLORS.reward, rewards)],
  };
  const speedData = {
    labels,
    datasets: [makeDataset("Avg Speed (m/s)", COLORS.speed, speeds)],
  };
  const congestionData = {
    labels,
    datasets: [makeDataset("Vehicles in Network", COLORS.congestion, congestion)],
  };
  const co2Data = {
    labels,
    datasets: [makeDataset("CO₂ Emissions (mg/s)", COLORS.co2, co2)],
  };

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------
  return (
    <div style={styles.root}>
      {/* ── Header ──────────────────────────────────────────────── */}
      <header style={styles.header}>
        <div style={styles.headerTitle}>
          🚦 IUTMS — Intelligent Urban Traffic Management System
        </div>
        <div style={styles.headerMeta}>
          <span style={{ ...styles.dot, background: connected ? "#69ff47" : "#ff4d6d" }} />
          {connected ? "Live" : "Disconnected"}
          {episode != null && (
            <span style={styles.episodeBadge}>Episode {episode}</span>
          )}
        </div>
      </header>

      {/* ── KPI Row ─────────────────────────────────────────────── */}
      <div style={styles.kpiRow}>
        <KpiCard label="Total Reward"    value={latestReward}     color={COLORS.reward.border}     />
        <KpiCard label="Avg Speed"       value={latestSpeed}      unit="m/s"  color={COLORS.speed.border}      />
        <KpiCard label="Congestion Index" value={latestCongestion} unit="veh" color={COLORS.congestion.border} />
        <KpiCard label="CO₂ Emissions"   value={latestCo2}        unit="mg/s" color={COLORS.co2.border}        />
      </div>

      {/* ── Charts ──────────────────────────────────────────────── */}
      <div style={styles.grid}>
        <div style={styles.chartCard}>
          <Line
            data={rewardData}
            options={chartOptions("Reward per Step", "Reward")}
          />
        </div>
        <div style={styles.chartCard}>
          <Line
            data={speedData}
            options={chartOptions("Average Vehicle Speed", "m/s")}
          />
        </div>
        <div style={styles.chartCard}>
          <Line
            data={congestionData}
            options={chartOptions("Congestion Index (Vehicles in Network)", "Vehicles")}
          />
        </div>
        <div style={styles.chartCard}>
          <Line
            data={co2Data}
            options={chartOptions("CO₂ Emissions", "mg/s")}
          />
        </div>
      </div>

      {/* ── Footer ──────────────────────────────────────────────── */}
      <footer style={styles.footer}>
        MARL Traffic Signal Control · SUMO 1.15 · PyTorch · Socket.io
      </footer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Inline styles (no external CSS dependency)
// ---------------------------------------------------------------------------

const styles = {
  root: {
    minHeight: "100vh",
    background: "#0f0f1a",
    color: "#e0e0e0",
    fontFamily: "'Segoe UI', system-ui, sans-serif",
    display: "flex",
    flexDirection: "column",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "12px 24px",
    background: "linear-gradient(90deg,#1a1a2e,#16213e)",
    borderBottom: "1px solid rgba(255,255,255,0.08)",
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 700,
    letterSpacing: "0.5px",
    color: "#fff",
  },
  headerMeta: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    fontSize: 13,
    color: "#aaa",
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: "50%",
    display: "inline-block",
  },
  episodeBadge: {
    background: "rgba(0,229,255,0.15)",
    color: "#00e5ff",
    borderRadius: 4,
    padding: "2px 8px",
    fontSize: 12,
    fontWeight: 600,
    marginLeft: 8,
  },
  kpiRow: {
    display: "flex",
    gap: 16,
    padding: "16px 24px",
    flexWrap: "wrap",
  },
  kpiCard: {
    flex: "1 1 160px",
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 10,
    padding: "14px 18px",
    textAlign: "center",
  },
  kpiValue: {
    fontSize: 28,
    fontWeight: 700,
    fontVariantNumeric: "tabular-nums",
  },
  kpiLabel: {
    marginTop: 4,
    fontSize: 12,
    color: "#888",
    textTransform: "uppercase",
    letterSpacing: "0.8px",
  },
  kpiUnit: {
    color: "#666",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(440px, 1fr))",
    gap: 16,
    padding: "0 24px 24px",
    flex: 1,
  },
  chartCard: {
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.07)",
    borderRadius: 10,
    padding: "16px",
  },
  footer: {
    textAlign: "center",
    padding: "10px",
    fontSize: 11,
    color: "#444",
    borderTop: "1px solid rgba(255,255,255,0.05)",
  },
};
