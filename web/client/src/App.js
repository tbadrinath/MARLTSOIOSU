/**
 * App.js
 * ------
 * Root component for the IUTMS real-time dashboard.
 *
 * Features:
 *   • Live mode  – connects to the Socket.io server and streams real simulation
 *                  metrics (total reward, avg speed, congestion, CO₂).
 *   • Demo mode  – generates realistic simulated traffic data locally so the
 *                  dashboard can be explored without a running SUMO backend.
 *                  This makes the app fully functional when deployed to Vercel.
 *   • Config panel – shows training hyper-parameters and system information.
 *   • CSV export – download the current chart data as a CSV file.
 *
 * Performance improvements over v1:
 *   • useMemo for all chart-data objects – avoids recreating five identical
 *     objects on every state update that doesn't touch the relevant series.
 *   • handleMetric wrapped in useCallback with stable deps – prevents the
 *     Socket.io useEffect from re-running on every render.
 *   • Demo-state stored in a ref (demoRef) instead of React state, so the
 *     interval timer never causes an extra render cycle per tick.
 */

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
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

const SERVER_URL  = process.env.REACT_APP_SERVER_URL || "http://localhost:3001";
const MAX_POINTS  = 200; // max data points kept per chart series

// Demo-mode timing
const DEMO_INTERVAL_MS  = 150;   // ms between generated data points
const DEMO_STEP_SIZE    = 10;    // simulation steps advanced per interval tick
const DEMO_MAX_STEPS    = 3600;  // steps per simulated episode
const DEMO_TOTAL_EP     = 50;    // episodes shown in demo before cycling

// ---------------------------------------------------------------------------
// Chart colour palette
// ---------------------------------------------------------------------------

const COLORS = {
  reward:     { border: "#00e5ff", bg: "rgba(0,229,255,0.12)" },
  speed:      { border: "#69ff47", bg: "rgba(105,255,71,0.12)" },
  congestion: { border: "#ff9800", bg: "rgba(255,152,0,0.12)"  },
  co2:        { border: "#ff4d6d", bg: "rgba(255,77,109,0.12)" },
};

// ---------------------------------------------------------------------------
// Demo data generator
// Simulates improving agent performance over episodes (DQN learning curve).
// ---------------------------------------------------------------------------

function generateDemoMetric(step, episode) {
  const epProgress    = Math.min((episode - 1) / DEMO_TOTAL_EP, 1);
  const phaseProgress = step / DEMO_MAX_STEPS;
  const n = () => (Math.random() - 0.5);

  return {
    episode,
    step,
    avg_speed:           Math.max(0, 2 + 8 * epProgress + 3 * phaseProgress + n() * 1.5),
    co2_emissions:       Math.max(0, 8000 - 5000 * epProgress - 1000 * phaseProgress + n() * 400),
    total_reward:        -4 + 6 * epProgress + 3 * phaseProgress + n() * 0.4,
    vehicles_in_network: Math.max(0, Math.round(
      100 - 60 * epProgress + 40 * Math.sin(phaseProgress * Math.PI) + n() * 8
    )),
  };
}

// ---------------------------------------------------------------------------
// Shared chart helpers
// ---------------------------------------------------------------------------

function makeDataset(label, color, data) {
  return {
    label,
    data,
    borderColor:     color.border,
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
        grid:  { color: "rgba(255,255,255,0.05)" },
      },
      y: {
        title: { display: true, text: yLabel, color: "#888", font: { size: 11 } },
        ticks: { color: "#666" },
        grid:  { color: "rgba(255,255,255,0.05)" },
      },
    },
  };
}

// ---------------------------------------------------------------------------
// Sub-components
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

function Badge({ children, color = "#00e5ff", bg = "rgba(0,229,255,0.15)" }) {
  return (
    <span style={{ ...styles.badge, color, background: bg }}>
      {children}
    </span>
  );
}

/** Collapsible info panel explaining the MARL system. */
function AboutPanel() {
  const [open, setOpen] = useState(false);
  return (
    <div style={styles.aboutWrapper}>
      <button style={styles.toggleBtn} onClick={() => setOpen(o => !o)}>
        {open ? "▲ Hide" : "▼ About this project"}
      </button>
      {open && (
        <div style={styles.aboutBody}>
          <h3 style={styles.aboutTitle}>
            🚦 Intelligent Urban Traffic Management System (IUTMS)
          </h3>
          <p style={styles.aboutText}>
            IUTMS applies <strong>Multi-Agent Reinforcement Learning (MARL)</strong>{" "}
            to adaptive traffic signal control. Each intersection is controlled by
            an independent <strong>Deep Q-Network (DQN)</strong> agent that learns
            to minimise congestion, waiting time, and CO₂ emissions while maximising
            vehicle throughput.
          </p>
          <div style={styles.aboutGrid}>
            <div style={styles.aboutCard}>
              <div style={styles.aboutCardTitle}>Observation Space</div>
              <div style={styles.aboutCardText}>
                Normalised vehicle count + lane occupancy per incoming lane, plus a
                downstream spillback flag per outgoing lane.
              </div>
            </div>
            <div style={styles.aboutCard}>
              <div style={styles.aboutCardTitle}>Reward Function</div>
              <div style={{ ...styles.aboutCardText, fontFamily: "monospace", fontSize: 12 }}>
                R = α·Throughput − β·Queue − γ·WaitTime − δ·Spillback
              </div>
            </div>
            <div style={styles.aboutCard}>
              <div style={styles.aboutCardTitle}>Agent Architecture</div>
              <div style={styles.aboutCardText}>
                FC(64) → ReLU → FC(32) → ReLU → Q-values.
                Experience replay (10k), target network, ε-greedy decay.
              </div>
            </div>
            <div style={styles.aboutCard}>
              <div style={styles.aboutCardTitle}>Simulator</div>
              <div style={styles.aboutCardText}>
                SUMO 1.15 via TraCI. 3,600-step episodes on a synthetic grid network.
                Python backend streams metrics to this dashboard over Socket.io.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/** Collapsible panel showing default training hyper-parameters. */
function ConfigPanel() {
  const [open, setOpen] = useState(false);
  const params = [
    ["Learning rate",     "0.001"],
    ["Discount (γ)",      "0.99"],
    ["ε start / min",     "1.0 → 0.05"],
    ["ε decay",           "0.995"],
    ["Batch size",        "64"],
    ["Replay buffer",     "10 000"],
    ["Target update",     "every 100 steps"],
    ["Phase duration",    "10 sim steps"],
    ["Max steps / ep.",   "3,600"],
    ["Reward α/β/γ/δ",   "0.4 / 0.3 / 0.2 / 0.5"],
  ];
  return (
    <div style={styles.aboutWrapper}>
      <button style={styles.toggleBtn} onClick={() => setOpen(o => !o)}>
        {open ? "▲ Hide" : "▼ Training configuration"}
      </button>
      {open && (
        <div style={styles.configBody}>
          {params.map(([k, v]) => (
            <div key={k} style={styles.configRow}>
              <span style={styles.configKey}>{k}</span>
              <span style={styles.configVal}>{v}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// CSV export helper
// ---------------------------------------------------------------------------

function exportCSV(steps, rewards, speeds, congestion, co2) {
  const rows = ["step,total_reward,avg_speed,vehicles_in_network,co2_emissions"];
  const len = steps.length;
  for (let i = 0; i < len; i++) {
    rows.push(
      [steps[i] ?? "", rewards[i] ?? "", speeds[i] ?? "", congestion[i] ?? "", co2[i] ?? ""].join(",")
    );
  }
  const blob = new Blob([rows.join("\n")], { type: "text/csv" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = `iutms_metrics_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Main App component
// ---------------------------------------------------------------------------

export default function App() {
  const socketRef = useRef(null);
  const demoRef   = useRef({ step: 0, episode: 1 }); // mutable demo state (no render)

  const [connected, setConnected] = useState(false);
  const [demoMode,  setDemoMode]  = useState(false);
  const [episode,   setEpisode]   = useState(null);

  // Rolling chart series
  const [steps,      setSteps]      = useState([]);
  const [rewards,    setRewards]    = useState([]);
  const [speeds,     setSpeeds]     = useState([]);
  const [congestion, setCongestion] = useState([]);
  const [co2,        setCo2]        = useState([]);

  // Latest KPI values
  const [latestReward,     setLatestReward]     = useState("—");
  const [latestSpeed,      setLatestSpeed]      = useState("—");
  const [latestCongestion, setLatestCongestion] = useState("—");
  const [latestCo2,        setLatestCo2]        = useState("—");

  // -------------------------------------------------------------------------
  // Process one incoming metrics payload
  // -------------------------------------------------------------------------
  const handleMetric = useCallback((m) => {
    const step     = m.step  ?? 0;
    const rw       = typeof m.total_reward        === "number" ? +m.total_reward.toFixed(3)   : null;
    const spd      = typeof m.avg_speed           === "number" ? +m.avg_speed.toFixed(2)       : null;
    const cong     = typeof m.vehicles_in_network === "number" ? m.vehicles_in_network          : null;
    const emission = typeof m.co2_emissions       === "number" ? +m.co2_emissions.toFixed(1)   : null;

    if (m.episode != null) setEpisode(m.episode);

    setSteps      (prev => [...prev.slice(-(MAX_POINTS - 1)), step]);
    setRewards    (prev => [...prev.slice(-(MAX_POINTS - 1)), rw]);
    setSpeeds     (prev => [...prev.slice(-(MAX_POINTS - 1)), spd]);
    setCongestion (prev => [...prev.slice(-(MAX_POINTS - 1)), cong]);
    setCo2        (prev => [...prev.slice(-(MAX_POINTS - 1)), emission]);

    if (rw       != null) setLatestReward    (rw);
    if (spd      != null) setLatestSpeed     (spd);
    if (cong     != null) setLatestCongestion(cong);
    if (emission != null) setLatestCo2       (emission);
  }, []);

  // -------------------------------------------------------------------------
  // Socket.io connection (live mode)
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (demoMode) return; // demo mode: don't open a socket

    const socket = io(SERVER_URL, { transports: ["websocket", "polling"] });
    socketRef.current = socket;

    socket.on("connect",    () => setConnected(true));
    socket.on("disconnect", () => setConnected(false));

    // Hydrate charts from server-side history buffer
    socket.on("history", ({ data }) => {
      if (Array.isArray(data)) data.forEach(handleMetric);
    });

    socket.on("step_metrics", handleMetric);

    return () => {
      socket.disconnect();
      setConnected(false);
    };
  }, [demoMode, handleMetric]);

  // -------------------------------------------------------------------------
  // Demo mode – generate simulated data with a timer
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (!demoMode) return;

    // Clear previous chart data when entering demo
    setSteps([]); setRewards([]); setSpeeds([]); setCongestion([]); setCo2([]);
    setEpisode(1);
    demoRef.current = { step: 0, episode: 1 };

    const timer = setInterval(() => {
      const { step, episode } = demoRef.current;
      handleMetric(generateDemoMetric(step, episode));

      const nextStep = step + DEMO_STEP_SIZE;
      if (nextStep >= DEMO_MAX_STEPS) {
        demoRef.current = { step: 0, episode: episode + 1 };
      } else {
        demoRef.current.step = nextStep;
      }
    }, DEMO_INTERVAL_MS);

    return () => clearInterval(timer);
  }, [demoMode, handleMetric]);

  // -------------------------------------------------------------------------
  // Memoised chart datasets – only rebuild when the underlying series change
  // -------------------------------------------------------------------------
  const labels = useMemo(() => steps.map(String), [steps]);

  const rewardData = useMemo(() => ({
    labels,
    datasets: [makeDataset("Total Reward", COLORS.reward, rewards)],
  }), [labels, rewards]);

  const speedData = useMemo(() => ({
    labels,
    datasets: [makeDataset("Avg Speed (m/s)", COLORS.speed, speeds)],
  }), [labels, speeds]);

  const congestionData = useMemo(() => ({
    labels,
    datasets: [makeDataset("Vehicles in Network", COLORS.congestion, congestion)],
  }), [labels, congestion]);

  const co2Data = useMemo(() => ({
    labels,
    datasets: [makeDataset("CO₂ Emissions (mg/s)", COLORS.co2, co2)],
  }), [labels, co2]);

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  const statusColor  = demoMode ? "#ff9800" : (connected ? "#69ff47" : "#ff4d6d");
  const statusLabel  = demoMode ? "Demo" : (connected ? "Live" : "Disconnected");

  return (
    <div style={styles.root}>

      {/* ── Header ───────────────────────────────────────────────── */}
      <header style={styles.header}>
        <div style={styles.headerTitle}>
          🚦 IUTMS — Intelligent Urban Traffic Management System
        </div>
        <div style={styles.headerMeta}>
          <span style={{ ...styles.dot, background: statusColor }} />
          <span style={{ color: statusColor, fontWeight: 600 }}>{statusLabel}</span>
          {episode != null && (
            <Badge>Episode {episode}</Badge>
          )}
          <button
            style={{ ...styles.demoBtn, ...(demoMode ? styles.demoBtnActive : {}) }}
            onClick={() => setDemoMode(m => !m)}
            title={demoMode ? "Stop demo mode" : "Run demo simulation (no server required)"}
          >
            {demoMode ? "⏹ Stop Demo" : "▶ Demo Mode"}
          </button>
        </div>
      </header>

      {/* ── Connection hint when disconnected and not in demo ─────── */}
      {!connected && !demoMode && (
        <div style={styles.hint}>
          <span>⚡ No server connection — </span>
          <strong>Click "Demo Mode"</strong> to explore the dashboard with
          simulated data, or start the Node.js server at{" "}
          <code style={styles.code}>{SERVER_URL}</code>.
        </div>
      )}

      {/* ── KPI Row ──────────────────────────────────────────────── */}
      <div style={styles.kpiRow}>
        <KpiCard label="Total Reward"     value={latestReward}     color={COLORS.reward.border}     />
        <KpiCard label="Avg Speed"        value={latestSpeed}      unit="m/s"  color={COLORS.speed.border}      />
        <KpiCard label="Congestion Index" value={latestCongestion} unit="veh"  color={COLORS.congestion.border} />
        <KpiCard label="CO₂ Emissions"    value={latestCo2}        unit="mg/s" color={COLORS.co2.border}        />
      </div>

      {/* ── Charts ───────────────────────────────────────────────── */}
      <div style={styles.grid}>
        <div style={styles.chartCard}>
          <Line data={rewardData}     options={chartOptions("Reward per Step", "Reward")} />
        </div>
        <div style={styles.chartCard}>
          <Line data={speedData}      options={chartOptions("Average Vehicle Speed", "m/s")} />
        </div>
        <div style={styles.chartCard}>
          <Line data={congestionData} options={chartOptions("Congestion Index (Vehicles in Network)", "Vehicles")} />
        </div>
        <div style={styles.chartCard}>
          <Line data={co2Data}        options={chartOptions("CO₂ Emissions", "mg/s")} />
        </div>
      </div>

      {/* ── Export & Info panels ─────────────────────────────────── */}
      <div style={styles.panelRow}>
        <button
          style={styles.exportBtn}
          onClick={() => exportCSV(steps, rewards, speeds, congestion, co2)}
          disabled={steps.length === 0}
          title="Download current chart data as CSV"
        >
          ⬇ Export CSV
        </button>
      </div>

      <div style={styles.panelSection}>
        <AboutPanel  />
        <ConfigPanel />
      </div>

      {/* ── Footer ───────────────────────────────────────────────── */}
      <footer style={styles.footer}>
        MARL Traffic Signal Control · SUMO 1.15 · PyTorch · Socket.io ·{" "}
        <a
          href="https://github.com/tbadrinath/MARLTSOIOSU"
          target="_blank"
          rel="noreferrer"
          style={styles.footerLink}
        >
          GitHub
        </a>
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
    flexWrap: "wrap",
    gap: 8,
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
    gap: 10,
    fontSize: 13,
    color: "#aaa",
    flexWrap: "wrap",
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: "50%",
    display: "inline-block",
    flexShrink: 0,
  },
  badge: {
    borderRadius: 4,
    padding: "2px 8px",
    fontSize: 12,
    fontWeight: 600,
  },
  demoBtn: {
    background: "rgba(255,152,0,0.1)",
    border: "1px solid rgba(255,152,0,0.4)",
    color: "#ff9800",
    borderRadius: 6,
    padding: "4px 12px",
    fontSize: 12,
    fontWeight: 600,
    cursor: "pointer",
    transition: "background 0.2s",
  },
  demoBtnActive: {
    background: "rgba(255,152,0,0.25)",
    border: "1px solid #ff9800",
  },
  hint: {
    background: "rgba(0,229,255,0.06)",
    borderBottom: "1px solid rgba(0,229,255,0.12)",
    color: "#aaa",
    fontSize: 13,
    padding: "10px 24px",
    textAlign: "center",
  },
  code: {
    fontFamily: "monospace",
    background: "rgba(255,255,255,0.06)",
    padding: "1px 6px",
    borderRadius: 3,
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
    padding: "0 24px 16px",
    flex: 1,
  },
  chartCard: {
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.07)",
    borderRadius: 10,
    padding: "16px",
  },
  panelRow: {
    display: "flex",
    justifyContent: "flex-end",
    padding: "0 24px 8px",
    gap: 8,
  },
  exportBtn: {
    background: "rgba(105,255,71,0.08)",
    border: "1px solid rgba(105,255,71,0.3)",
    color: "#69ff47",
    borderRadius: 6,
    padding: "6px 16px",
    fontSize: 12,
    fontWeight: 600,
    cursor: "pointer",
  },
  panelSection: {
    padding: "0 24px 16px",
    display: "flex",
    flexDirection: "column",
    gap: 8,
  },
  aboutWrapper: {
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.07)",
    borderRadius: 10,
    overflow: "hidden",
  },
  toggleBtn: {
    width: "100%",
    background: "transparent",
    border: "none",
    color: "#888",
    padding: "10px 16px",
    textAlign: "left",
    cursor: "pointer",
    fontSize: 13,
    fontWeight: 600,
  },
  aboutBody: {
    padding: "0 16px 16px",
  },
  aboutTitle: {
    fontSize: 15,
    fontWeight: 700,
    color: "#e0e0e0",
    marginBottom: 8,
  },
  aboutText: {
    fontSize: 13,
    color: "#999",
    lineHeight: 1.6,
    marginBottom: 12,
  },
  aboutGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
    gap: 10,
  },
  aboutCard: {
    background: "rgba(255,255,255,0.04)",
    borderRadius: 8,
    padding: "10px 12px",
  },
  aboutCardTitle: {
    fontSize: 12,
    fontWeight: 700,
    color: "#00e5ff",
    textTransform: "uppercase",
    letterSpacing: "0.5px",
    marginBottom: 4,
  },
  aboutCardText: {
    fontSize: 12,
    color: "#888",
    lineHeight: 1.5,
  },
  configBody: {
    padding: "0 16px 16px",
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
    gap: "4px 24px",
  },
  configRow: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: 12,
    padding: "4px 0",
    borderBottom: "1px solid rgba(255,255,255,0.04)",
  },
  configKey: {
    color: "#888",
  },
  configVal: {
    color: "#c0c0d0",
    fontFamily: "monospace",
    fontWeight: 600,
  },
  footer: {
    textAlign: "center",
    padding: "10px",
    fontSize: 11,
    color: "#444",
    borderTop: "1px solid rgba(255,255,255,0.05)",
  },
  footerLink: {
    color: "#555",
    textDecoration: "none",
  },
};
