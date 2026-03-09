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
 *   • OSM Map    – search any city/location via OpenStreetMap, preview the map,
 *                  download the OSM data, convert it to a SUMO network, and
 *                  start a simulation on the imported map.
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
            to adaptive traffic signal control. Each intersection runs an independent
            RL agent — choose between <strong>DQN</strong> (ε-greedy, experience
            replay, target network) and <strong>PPO</strong> (clipped surrogate
            objective, GAE, actor-critic). Two reward modes are available:{" "}
            <em>composite</em> (throughput + queue + wait + spillback) and{" "}
            <em>pressure</em> (inspired by sumo-rl).
          </p>
          <div style={styles.aboutGrid}>
            <div style={styles.aboutCard}>
              <div style={styles.aboutCardTitle}>Observation Space</div>
              <div style={styles.aboutCardText}>
                Normalised vehicle count + lane occupancy per incoming lane, plus a
                downstream spillback flag. Optional: current phase index &amp;
                time-in-phase features (<code>--phase-obs</code>).
              </div>
            </div>
            <div style={styles.aboutCard}>
              <div style={styles.aboutCardTitle}>DQN Agent</div>
              <div style={{ ...styles.aboutCardText, fontFamily: "monospace", fontSize: 11 }}>
                FC(64)→ReLU→FC(32)→ReLU→Q(a){"\n"}
                Replay 10k · target update 100
              </div>
            </div>
            <div style={styles.aboutCard}>
              <div style={styles.aboutCardTitle}>PPO Agent</div>
              <div style={{ ...styles.aboutCardText, fontFamily: "monospace", fontSize: 11 }}>
                Shared FC(128→64)→Actor+Critic{"\n"}
                GAE λ=0.95 · clip ε=0.2 · n=512
              </div>
            </div>
            <div style={styles.aboutCard}>
              <div style={styles.aboutCardTitle}>Reward Modes</div>
              <div style={{ ...styles.aboutCardText, fontFamily: "monospace", fontSize: 11 }}>
                composite: α·T−β·Q−γ·W−δ·S{"\n"}
                pressure:  −|in−out|/lanes
              </div>
            </div>
            <div style={styles.aboutCard}>
              <div style={styles.aboutCardTitle}>OSM Map Import</div>
              <div style={styles.aboutCardText}>
                Search any city, download OSM data, auto-convert to SUMO network
                + routes, and run the simulation on real street layouts.
              </div>
            </div>
            <div style={styles.aboutCard}>
              <div style={styles.aboutCardTitle}>Simulator</div>
              <div style={styles.aboutCardText}>
                SUMO 1.15 via TraCI. 3,600-step episodes. Python backend streams
                metrics to this dashboard over Socket.io.
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
  const [open,    setOpen]    = useState(false);
  const [algo,    setAlgo]    = useState("dqn");
  const [reward,  setReward]  = useState("composite");

  const dqnParams = [
    ["Algorithm",      "DQN (ε-greedy)"],
    ["Learning rate",  "0.001"],
    ["Discount (γ)",   "0.99"],
    ["ε start / min",  "1.0 → 0.05"],
    ["ε decay",        "0.995"],
    ["Batch size",     "64"],
    ["Replay buffer",  "10 000"],
    ["Target update",  "every 100 steps"],
  ];
  const ppoParams = [
    ["Algorithm",         "PPO (actor-critic)"],
    ["Learning rate",     "0.0003"],
    ["Discount (γ)",      "0.99"],
    ["GAE λ",             "0.95"],
    ["Clip ε",            "0.2"],
    ["Value loss coef",   "0.5"],
    ["Entropy coef",      "0.01"],
    ["Rollout steps",     "512"],
    ["Epochs / update",   "10"],
    ["Batch size",        "64"],
  ];
  const compositeReward = [
    ["Reward mode",    "composite"],
    ["α (throughput)", "0.4"],
    ["β (queue)",      "0.3"],
    ["γ (wait time)",  "0.2"],
    ["δ (spillback)",  "0.5"],
  ];
  const pressureReward = [
    ["Reward mode",  "pressure (sumo-rl)"],
    ["Formula",      "−|in_queue − out_queue| / lanes"],
  ];
  const sharedParams = [
    ["Phase duration",  "10 sim steps"],
    ["Max steps / ep.", "3,600"],
    ["Phase obs",       "optional (+2 features)"],
  ];

  const algoRows   = algo   === "ppo"       ? ppoParams       : dqnParams;
  const rewardRows = reward === "pressure"  ? pressureReward  : compositeReward;

  const cliCommand = [
    "python -m simulation.trainer",
    `  --algo ${algo}`,
    `  --reward ${reward}`,
    algo === "ppo" ? "  --ppo-n-steps 512" : "  --epsilon-start 1.0",
    reward === "pressure" ? "" : "  --alpha 0.4 --beta 0.3",
    "  --episodes 200 --max-steps 3600",
  ].filter(Boolean).join(" \\\n");

  return (
    <div style={styles.aboutWrapper}>
      <button style={styles.toggleBtn} onClick={() => setOpen(o => !o)}>
        {open ? "▲ Hide" : "▼ Training configuration"}
      </button>
      {open && (
        <div style={{ padding: "0 16px 16px" }}>
          {/* ── Algorithm & reward selectors ── */}
          <div style={styles.cfgSelectorRow}>
            <div style={styles.cfgSelectorGroup}>
              <span style={styles.cfgSelectorLabel}>Algorithm</span>
              {["dqn", "ppo"].map(a => (
                <button
                  key={a}
                  style={{
                    ...styles.cfgToggleBtn,
                    ...(algo === a ? styles.cfgToggleBtnActive : {}),
                  }}
                  onClick={() => setAlgo(a)}
                >
                  {a.toUpperCase()}
                </button>
              ))}
            </div>
            <div style={styles.cfgSelectorGroup}>
              <span style={styles.cfgSelectorLabel}>Reward</span>
              {["composite", "pressure"].map(r => (
                <button
                  key={r}
                  style={{
                    ...styles.cfgToggleBtn,
                    ...(reward === r ? styles.cfgToggleBtnActive : {}),
                  }}
                  onClick={() => setReward(r)}
                >
                  {r}
                </button>
              ))}
            </div>
          </div>

          {/* ── Param table ── */}
          <div style={styles.configBody}>
            {[...algoRows, ...rewardRows, ...sharedParams].map(([k, v]) => (
              <div key={k} style={styles.configRow}>
                <span style={styles.configKey}>{k}</span>
                <span style={styles.configVal}>{v}</span>
              </div>
            ))}
          </div>

          {/* ── CLI command preview ── */}
          <div style={styles.cfgCliBox}>
            <div style={styles.cfgCliLabel}>CLI command</div>
            <pre style={styles.cfgCliPre}>{cliCommand}</pre>
          </div>
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
// OSM Map Panel
// ---------------------------------------------------------------------------

/** Status badge colours for import pipeline steps. */
const OSM_STATUS = {
  idle:      { color: "#888",    label: ""                          },
  searching: { color: "#ff9800", label: "🔍 Searching …"           },
  ready:     { color: "#00e5ff", label: "📌 Location found"        },
  importing: { color: "#ff9800", label: "⏳ Importing map …"       },
  done:      { color: "#69ff47", label: "✅ Map ready"             },
  error:     { color: "#ff4d6d", label: "❌ Error"                 },
};

/**
 * OsmMapPanel
 * -----------
 * A collapsible panel that lets the user:
 *  1. Search for any city / location using the Nominatim geocoding API
 *     (proxied through the Node.js server to satisfy the User-Agent policy).
 *  2. Preview the area on an embedded OpenStreetMap iframe.
 *  3. Click "Import & Simulate" to trigger the server-side Python pipeline
 *     (download OSM → netconvert → randomTrips → SUMO).
 *
 * In demo / standalone mode (no server connection) the search still works via
 * the server proxy, but the import step will fail gracefully with an error
 * message instead of silently hanging.
 */
function OsmMapPanel({ serverUrl }) {
  const [open,       setOpen]       = useState(false);
  const [query,      setQuery]      = useState("");
  const [results,    setResults]    = useState([]);
  const [selected,   setSelected]   = useState(null);   // chosen Nominatim result
  const [status,     setStatus]     = useState("idle");
  const [statusMsg,  setStatusMsg]  = useState("");
  const [importInfo, setImportInfo] = useState(null);   // result from /api/osm/import

  // Build an OSM embed URL from a selected result's lat/lon/bbox.
  const mapEmbedUrl = selected
    ? (() => {
        const bb = selected.boundingbox;
        // bbox: min_lat, max_lat, min_lon, max_lon  →  OSM needs: left, bottom, right, top
        const left   = parseFloat(bb[2]);
        const bottom = parseFloat(bb[0]);
        const right  = parseFloat(bb[3]);
        const top    = parseFloat(bb[1]);
        return (
          `https://www.openstreetmap.org/export/embed.html` +
          `?bbox=${left},${bottom},${right},${top}` +
          `&layer=mapnik` +
          `&marker=${selected.lat},${selected.lon}`
        );
      })()
    : null;

  const handleSearch = async () => {
    if (!query.trim()) return;
    setStatus("searching");
    setStatusMsg("");
    setResults([]);
    setSelected(null);
    setImportInfo(null);

    try {
      const resp = await fetch(
        `${serverUrl}/api/osm/search?q=${encodeURIComponent(query)}&limit=5`
      );
      if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
      const data = await resp.json();
      if (!data.results || data.results.length === 0) {
        setStatus("error");
        setStatusMsg("No results found. Try a more specific location name.");
        return;
      }
      setResults(data.results);
      setSelected(data.results[0]);
      setStatus("ready");
    } catch (err) {
      setStatus("error");
      setStatusMsg(`Search failed: ${err.message}`);
    }
  };

  const handleImport = async () => {
    if (!selected) return;
    setStatus("importing");
    setStatusMsg("Downloading OSM data and building SUMO network …");
    setImportInfo(null);

    try {
      const resp = await fetch(`${serverUrl}/api/osm/import`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          location:     selected.display_name,
          num_vehicles: 400,
          seed:         42,
        }),
      });

      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || data.error || `Server error ${resp.status}`);
      }

      setImportInfo(data);
      setStatus("done");
      setStatusMsg(
        `Network and routes saved. Start the trainer with:\n` +
        `  python -m simulation.trainer --net-file "${data.net_file}" --route-file "${data.route_file}"`
      );
    } catch (err) {
      setStatus("error");
      setStatusMsg(`Import failed: ${err.message}`);
    }
  };

  const st = OSM_STATUS[status] || OSM_STATUS.idle;

  return (
    <div style={styles.aboutWrapper}>
      <button style={styles.toggleBtn} onClick={() => setOpen(o => !o)}>
        {open ? "▲ Hide" : "▼ 🗺️ OSM Map Importer — simulate any city"}
      </button>
      {open && (
        <div style={styles.osmBody}>
          <p style={styles.osmIntro}>
            Search for any city or location, preview it on OpenStreetMap, then
            click <strong>Import &amp; Simulate</strong> to automatically download
            the road network, convert it to a SUMO network, generate vehicle
            routes, and run the MARL simulation.
          </p>

          {/* ── Search row ── */}
          <div style={styles.osmSearchRow}>
            <input
              style={styles.osmInput}
              type="text"
              placeholder="e.g. Manhattan, New York or Bangalore, India"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            />
            <button
              style={styles.osmSearchBtn}
              onClick={handleSearch}
              disabled={status === "searching" || !query.trim()}
            >
              Search
            </button>
          </div>

          {/* ── Result selector ── */}
          {results.length > 1 && (
            <select
              style={styles.osmSelect}
              value={results.indexOf(selected)}
              onChange={(e) => {
                setSelected(results[parseInt(e.target.value, 10)]);
                setStatus("ready");
                setImportInfo(null);
              }}
            >
              {results.map((r, i) => (
                <option key={i} value={i}>{r.display_name}</option>
              ))}
            </select>
          )}

          {/* ── Map preview ── */}
          {mapEmbedUrl && (
            <div style={styles.osmMapWrap}>
              <iframe
                title="OSM Map Preview"
                src={mapEmbedUrl}
                style={styles.osmIframe}
                loading="lazy"
                referrerPolicy="no-referrer"
              />
              <div style={styles.osmMapCaption}>
                <a
                  href={`https://www.openstreetmap.org/#map=14/${selected.lat}/${selected.lon}`}
                  target="_blank"
                  rel="noreferrer"
                  style={styles.osmMapLink}
                >
                  View full map on OpenStreetMap ↗
                </a>
              </div>
            </div>
          )}

          {/* ── Import button ── */}
          {selected && (
            <button
              style={{
                ...styles.osmImportBtn,
                ...(status === "importing" ? styles.osmImportBtnBusy : {}),
              }}
              onClick={handleImport}
              disabled={status === "importing"}
            >
              {status === "importing"
                ? "⏳ Importing …"
                : "🚀 Import & Simulate"}
            </button>
          )}

          {/* ── Status ── */}
          {status !== "idle" && (
            <div style={{ ...styles.osmStatus, color: st.color }}>
              <span style={styles.osmStatusBadge}>{st.label}</span>
              {statusMsg && (
                <pre style={styles.osmStatusMsg}>{statusMsg}</pre>
              )}
            </div>
          )}

          {/* ── Import result summary ── */}
          {importInfo && (
            <div style={styles.osmResultBox}>
              <div style={styles.osmResultTitle}>Generated files</div>
              {[
                ["Network",  importInfo.net_file],
                ["Routes",   importInfo.route_file],
                ["OSM data", importInfo.osm_file],
              ].map(([label, val]) => (
                <div key={label} style={styles.osmResultRow}>
                  <span style={styles.osmResultKey}>{label}</span>
                  <code style={styles.osmResultVal}>{val}</code>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

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
        <OsmMapPanel serverUrl={SERVER_URL} />
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
  // ── ConfigPanel algorithm/reward selector ──────────────────────────────
  cfgSelectorRow: {
    display: "flex",
    gap: 16,
    marginBottom: 12,
    flexWrap: "wrap",
  },
  cfgSelectorGroup: {
    display: "flex",
    alignItems: "center",
    gap: 6,
  },
  cfgSelectorLabel: {
    fontSize: 11,
    color: "#777",
    textTransform: "uppercase",
    letterSpacing: "0.5px",
    marginRight: 4,
  },
  cfgToggleBtn: {
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.12)",
    color: "#777",
    borderRadius: 5,
    padding: "3px 10px",
    fontSize: 12,
    fontWeight: 600,
    cursor: "pointer",
    fontFamily: "monospace",
  },
  cfgToggleBtnActive: {
    background: "rgba(0,229,255,0.12)",
    border: "1px solid rgba(0,229,255,0.4)",
    color: "#00e5ff",
  },
  cfgCliBox: {
    marginTop: 12,
    background: "rgba(0,0,0,0.3)",
    borderRadius: 6,
    padding: "8px 12px",
  },
  cfgCliLabel: {
    fontSize: 10,
    color: "#555",
    textTransform: "uppercase",
    letterSpacing: "0.5px",
    marginBottom: 4,
  },
  cfgCliPre: {
    margin: 0,
    fontFamily: "monospace",
    fontSize: 11,
    color: "#69ff47",
    whiteSpace: "pre-wrap",
    wordBreak: "break-all",
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

  // ── OSM Map Panel ────────────────────────────────────────────────────────
  osmBody: {
    padding: "0 16px 16px",
  },
  osmIntro: {
    fontSize: 13,
    color: "#999",
    lineHeight: 1.6,
    marginBottom: 12,
  },
  osmSearchRow: {
    display: "flex",
    gap: 8,
    marginBottom: 10,
  },
  osmInput: {
    flex: 1,
    background: "rgba(255,255,255,0.05)",
    border: "1px solid rgba(255,255,255,0.15)",
    borderRadius: 6,
    color: "#e0e0e0",
    fontSize: 13,
    padding: "7px 12px",
    outline: "none",
  },
  osmSearchBtn: {
    background: "rgba(0,229,255,0.12)",
    border: "1px solid rgba(0,229,255,0.4)",
    color: "#00e5ff",
    borderRadius: 6,
    padding: "6px 18px",
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
    flexShrink: 0,
  },
  osmSelect: {
    width: "100%",
    background: "rgba(255,255,255,0.05)",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: 6,
    color: "#c0c0d0",
    fontSize: 12,
    padding: "6px 10px",
    marginBottom: 10,
  },
  osmMapWrap: {
    marginBottom: 12,
    borderRadius: 8,
    overflow: "hidden",
    border: "1px solid rgba(255,255,255,0.1)",
  },
  osmIframe: {
    width: "100%",
    height: 300,
    border: "none",
    display: "block",
  },
  osmMapCaption: {
    background: "rgba(0,0,0,0.5)",
    padding: "4px 12px",
    fontSize: 11,
    textAlign: "right",
  },
  osmMapLink: {
    color: "#00e5ff",
    textDecoration: "none",
  },
  osmImportBtn: {
    background: "rgba(105,255,71,0.1)",
    border: "1px solid rgba(105,255,71,0.4)",
    color: "#69ff47",
    borderRadius: 6,
    padding: "8px 20px",
    fontSize: 13,
    fontWeight: 700,
    cursor: "pointer",
    marginBottom: 12,
    display: "block",
  },
  osmImportBtnBusy: {
    opacity: 0.5,
    cursor: "not-allowed",
  },
  osmStatus: {
    marginBottom: 10,
    fontSize: 13,
  },
  osmStatusBadge: {
    fontWeight: 700,
    display: "block",
    marginBottom: 4,
  },
  osmStatusMsg: {
    background: "rgba(255,255,255,0.04)",
    borderRadius: 6,
    padding: "8px 12px",
    fontSize: 12,
    color: "#ccc",
    whiteSpace: "pre-wrap",
    wordBreak: "break-all",
    margin: 0,
    fontFamily: "monospace",
  },
  osmResultBox: {
    background: "rgba(105,255,71,0.05)",
    border: "1px solid rgba(105,255,71,0.2)",
    borderRadius: 8,
    padding: "10px 14px",
    marginTop: 4,
  },
  osmResultTitle: {
    fontSize: 12,
    fontWeight: 700,
    color: "#69ff47",
    textTransform: "uppercase",
    letterSpacing: "0.5px",
    marginBottom: 8,
  },
  osmResultRow: {
    display: "flex",
    gap: 10,
    fontSize: 12,
    marginBottom: 4,
    alignItems: "flex-start",
  },
  osmResultKey: {
    color: "#888",
    minWidth: 60,
    flexShrink: 0,
  },
  osmResultVal: {
    color: "#c0c0d0",
    fontFamily: "monospace",
    wordBreak: "break-all",
    fontSize: 11,
  },
};
