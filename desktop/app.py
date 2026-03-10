"""
app.py
------
Tkinter-based desktop GUI for the Intelligent Urban Traffic Management System
(IUTMS).  Provides a complete graphical interface for:

* Configuring and launching DQN / PPO training
* OSM map import (search, preview, convert)
* Live metrics dashboard with matplotlib charts
* Demo mode (runs without SUMO)
* SUMO installation detection
* Telemetry server management (embedded Python server)

This is the main entry-point for the Windows .exe built with PyInstaller.
"""

from __future__ import annotations

import logging
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Determine the project root (works both when running from source and from
# a PyInstaller bundle where _MEIPASS is set).
# ---------------------------------------------------------------------------

if getattr(sys, "frozen", False):
    # Running as a PyInstaller bundle
    BUNDLE_DIR = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    PROJECT_ROOT = Path(sys.executable).parent
else:
    BUNDLE_DIR = Path(__file__).resolve().parent.parent
    PROJECT_ROOT = BUNDLE_DIR

MAPS_DIR = PROJECT_ROOT / "maps"


# ---------------------------------------------------------------------------
# Colour theme (dark)
# ---------------------------------------------------------------------------

THEME = {
    "bg":        "#1a1a2e",
    "bg2":       "#16213e",
    "bg3":       "#0f3460",
    "fg":        "#e0e0e0",
    "fg2":       "#a0a0b0",
    "accent":    "#00e5ff",
    "accent2":   "#69ff47",
    "warning":   "#ff9800",
    "error":     "#ff4d6d",
    "success":   "#69ff47",
    "card_bg":   "#1e1e3a",
    "entry_bg":  "#2a2a4a",
    "btn_bg":    "#0f3460",
    "btn_fg":    "#00e5ff",
}


def _find_sumo() -> Optional[str]:
    """Try to locate the SUMO installation directory."""
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home and Path(sumo_home).is_dir():
        return sumo_home

    if platform.system() == "Windows":
        candidates = [
            r"C:\Program Files (x86)\Eclipse\Sumo",
            r"C:\Program Files\Eclipse\Sumo",
            r"C:\Sumo",
            os.path.expandvars(r"%LOCALAPPDATA%\Sumo"),
        ]
    else:
        candidates = [
            "/usr/share/sumo",
            "/opt/sumo",
            "/usr/local/share/sumo",
            "/opt/homebrew/opt/sumo/share/sumo",
        ]

    for c in candidates:
        if Path(c).is_dir():
            return c

    # Check PATH for sumo binary
    sumo_bin = shutil.which("sumo")
    if sumo_bin:
        return str(Path(sumo_bin).parent.parent)

    return None


# ============================================================================
# Main Application Window
# ============================================================================

class IUTMSApp(tk.Tk):
    """Main application window for IUTMS Desktop."""

    def __init__(self) -> None:
        super().__init__()

        self.title("IUTMS - Intelligent Urban Traffic Management System")
        self.geometry("1100x750")
        self.minsize(900, 600)
        self.configure(bg=THEME["bg"])

        # State
        self._server_thread: Optional[threading.Thread] = None
        self._server_running = False
        self._training_process: Optional[subprocess.Popen] = None
        self._demo_running = False
        self._chart_data: Dict[str, List[float]] = {
            "steps": [], "rewards": [], "speeds": [],
            "congestion": [], "co2": [],
        }
        self._sumo_home = _find_sumo()

        # Track the after() callback id so we can cancel it
        self._chart_after_id: Optional[str] = None

        self._build_ui()
        self._start_telemetry_server()
        self._update_sumo_status()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the complete user interface."""
        # Top bar
        top = tk.Frame(self, bg=THEME["bg2"], height=50)
        top.pack(fill=tk.X)
        top.pack_propagate(False)

        tk.Label(
            top, text="\u2728 IUTMS Desktop",
            font=("Segoe UI", 16, "bold"),
            fg=THEME["accent"], bg=THEME["bg2"],
        ).pack(side=tk.LEFT, padx=16)

        tk.Label(
            top,
            text="Multi-Agent Deep RL for Traffic Signal Optimisation",
            font=("Segoe UI", 10),
            fg=THEME["fg2"], bg=THEME["bg2"],
        ).pack(side=tk.LEFT, padx=8)

        # SUMO status label (right side of top bar)
        self._sumo_label = tk.Label(
            top, text="", font=("Segoe UI", 9),
            fg=THEME["fg2"], bg=THEME["bg2"],
        )
        self._sumo_label.pack(side=tk.RIGHT, padx=16)

        # Server status
        self._server_label = tk.Label(
            top, text="Server: starting...",
            font=("Segoe UI", 9),
            fg=THEME["warning"], bg=THEME["bg2"],
        )
        self._server_label.pack(side=tk.RIGHT, padx=8)

        # Notebook (tabs)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=THEME["bg"])
        style.configure(
            "TNotebook.Tab",
            background=THEME["bg2"],
            foreground=THEME["fg"],
            padding=[14, 6],
            font=("Segoe UI", 10),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", THEME["bg3"])],
            foreground=[("selected", THEME["accent"])],
        )

        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Create tabs
        self._build_dashboard_tab()
        self._build_training_tab()
        self._build_osm_tab()
        self._build_sumo_tab()
        self._build_about_tab()

    # -----------------------------------------------------------------------
    # Dashboard Tab
    # -----------------------------------------------------------------------

    def _build_dashboard_tab(self) -> None:
        """Live metrics dashboard with matplotlib charts."""
        frame = tk.Frame(self._notebook, bg=THEME["bg"])
        self._notebook.add(frame, text="  Dashboard  ")

        # Controls bar
        ctrl = tk.Frame(frame, bg=THEME["bg2"], height=45)
        ctrl.pack(fill=tk.X, padx=4, pady=4)
        ctrl.pack_propagate(False)

        self._demo_btn = tk.Button(
            ctrl, text="Start Demo", font=("Segoe UI", 10, "bold"),
            fg=THEME["btn_fg"], bg=THEME["btn_bg"],
            activebackground=THEME["bg3"], activeforeground=THEME["accent"],
            relief=tk.FLAT, padx=16, pady=4,
            command=self._toggle_demo,
        )
        self._demo_btn.pack(side=tk.LEFT, padx=8, pady=6)

        tk.Button(
            ctrl, text="Open Web Dashboard", font=("Segoe UI", 10),
            fg=THEME["btn_fg"], bg=THEME["btn_bg"],
            activebackground=THEME["bg3"], activeforeground=THEME["accent"],
            relief=tk.FLAT, padx=16, pady=4,
            command=lambda: webbrowser.open("http://localhost:3001"),
        ).pack(side=tk.LEFT, padx=8, pady=6)

        tk.Button(
            ctrl, text="Clear Charts", font=("Segoe UI", 10),
            fg=THEME["fg2"], bg=THEME["bg2"],
            activebackground=THEME["bg3"],
            relief=tk.FLAT, padx=16, pady=4,
            command=self._clear_charts,
        ).pack(side=tk.LEFT, padx=8, pady=6)

        tk.Button(
            ctrl, text="Export CSV", font=("Segoe UI", 10),
            fg=THEME["fg2"], bg=THEME["bg2"],
            activebackground=THEME["bg3"],
            relief=tk.FLAT, padx=16, pady=4,
            command=self._export_csv,
        ).pack(side=tk.RIGHT, padx=8, pady=6)

        # KPI cards row
        kpi_frame = tk.Frame(frame, bg=THEME["bg"])
        kpi_frame.pack(fill=tk.X, padx=8, pady=4)

        self._kpi_labels: Dict[str, tk.Label] = {}
        for name, label, color in [
            ("reward", "Total Reward", THEME["accent"]),
            ("speed", "Avg Speed (m/s)", THEME["accent2"]),
            ("congestion", "Vehicles", THEME["warning"]),
            ("co2", "CO2 (mg/s)", THEME["error"]),
        ]:
            card = tk.Frame(kpi_frame, bg=THEME["card_bg"], padx=16, pady=8)
            card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            val_lbl = tk.Label(
                card, text="--", font=("Segoe UI", 18, "bold"),
                fg=color, bg=THEME["card_bg"],
            )
            val_lbl.pack()
            tk.Label(
                card, text=label, font=("Segoe UI", 9),
                fg=THEME["fg2"], bg=THEME["card_bg"],
            ).pack()
            self._kpi_labels[name] = val_lbl

        # Charts area
        chart_frame = tk.Frame(frame, bg=THEME["bg"])
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        try:
            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure

            self._fig = Figure(figsize=(10, 5), dpi=90, facecolor=THEME["bg"])
            self._fig.subplots_adjust(
                hspace=0.45, wspace=0.3,
                left=0.06, right=0.98, top=0.93, bottom=0.08,
            )

            titles = [
                ("Reward per Step", THEME["accent"]),
                ("Avg Speed (m/s)", THEME["accent2"]),
                ("Congestion (vehicles)", THEME["warning"]),
                ("CO2 Emissions (mg/s)", THEME["error"]),
            ]
            self._axes = []
            for i, (title, color) in enumerate(titles):
                ax = self._fig.add_subplot(2, 2, i + 1)
                ax.set_facecolor(THEME["bg2"])
                ax.set_title(title, color=color, fontsize=9, fontweight="bold")
                ax.tick_params(colors=THEME["fg2"], labelsize=7)
                for spine in ax.spines.values():
                    spine.set_color(THEME["bg3"])
                ax.grid(True, alpha=0.15, color=THEME["fg2"])
                self._axes.append(ax)

            self._canvas = FigureCanvasTkAgg(self._fig, master=chart_frame)
            self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._has_matplotlib = True

        except ImportError:
            self._has_matplotlib = False
            tk.Label(
                chart_frame,
                text=(
                    "matplotlib not installed.\n"
                    "Install it with: pip install matplotlib\n\n"
                    "You can still use the Web Dashboard button above."
                ),
                font=("Segoe UI", 12),
                fg=THEME["fg2"], bg=THEME["bg"],
                justify=tk.CENTER,
            ).pack(expand=True)

    # -----------------------------------------------------------------------
    # Training Tab
    # -----------------------------------------------------------------------

    def _build_training_tab(self) -> None:
        """Training configuration and launch panel."""
        frame = tk.Frame(self._notebook, bg=THEME["bg"])
        self._notebook.add(frame, text="  Training  ")

        # Scrollable content
        canvas = tk.Canvas(frame, bg=THEME["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=THEME["bg"])

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Title
        tk.Label(
            scroll_frame,
            text="Training Configuration",
            font=("Segoe UI", 14, "bold"),
            fg=THEME["accent"], bg=THEME["bg"],
        ).pack(anchor="w", padx=16, pady=(12, 4))

        self._train_vars: Dict[str, tk.Variable] = {}

        # --- Algorithm selection ---
        section = self._make_section(scroll_frame, "Algorithm & Reward")

        algo_frame = tk.Frame(section, bg=THEME["card_bg"])
        algo_frame.pack(fill=tk.X, pady=4)
        tk.Label(
            algo_frame, text="Algorithm:", font=("Segoe UI", 10),
            fg=THEME["fg"], bg=THEME["card_bg"],
        ).pack(side=tk.LEFT, padx=8)
        self._train_vars["algo"] = tk.StringVar(value="dqn")
        for val, text in [("dqn", "DQN"), ("ppo", "PPO")]:
            tk.Radiobutton(
                algo_frame, text=text, variable=self._train_vars["algo"],
                value=val, font=("Segoe UI", 10),
                fg=THEME["fg"], bg=THEME["card_bg"],
                selectcolor=THEME["bg3"],
                activebackground=THEME["card_bg"],
                activeforeground=THEME["accent"],
            ).pack(side=tk.LEFT, padx=8)

        reward_frame = tk.Frame(section, bg=THEME["card_bg"])
        reward_frame.pack(fill=tk.X, pady=4)
        tk.Label(
            reward_frame, text="Reward Mode:", font=("Segoe UI", 10),
            fg=THEME["fg"], bg=THEME["card_bg"],
        ).pack(side=tk.LEFT, padx=8)
        self._train_vars["reward"] = tk.StringVar(value="composite")
        for val, text in [("composite", "Composite"), ("pressure", "Pressure")]:
            tk.Radiobutton(
                reward_frame, text=text, variable=self._train_vars["reward"],
                value=val, font=("Segoe UI", 10),
                fg=THEME["fg"], bg=THEME["card_bg"],
                selectcolor=THEME["bg3"],
                activebackground=THEME["card_bg"],
                activeforeground=THEME["accent"],
            ).pack(side=tk.LEFT, padx=8)

        phase_frame = tk.Frame(section, bg=THEME["card_bg"])
        phase_frame.pack(fill=tk.X, pady=4)
        self._train_vars["phase_obs"] = tk.BooleanVar(value=False)
        tk.Checkbutton(
            phase_frame, text="Include phase-time observations",
            variable=self._train_vars["phase_obs"],
            font=("Segoe UI", 10),
            fg=THEME["fg"], bg=THEME["card_bg"],
            selectcolor=THEME["bg3"],
            activebackground=THEME["card_bg"],
        ).pack(side=tk.LEFT, padx=8)

        # --- SUMO files ---
        section2 = self._make_section(scroll_frame, "SUMO Files")

        for key, label, default in [
            ("net_file", "Network File (.net.xml):", "maps/grid.net.xml"),
            ("route_file", "Route File (.rou.xml):", "maps/grid.rou.xml"),
        ]:
            row = tk.Frame(section2, bg=THEME["card_bg"])
            row.pack(fill=tk.X, pady=2)
            tk.Label(
                row, text=label, font=("Segoe UI", 10),
                fg=THEME["fg"], bg=THEME["card_bg"], width=24, anchor="w",
            ).pack(side=tk.LEFT, padx=8)
            self._train_vars[key] = tk.StringVar(value=default)
            entry = tk.Entry(
                row, textvariable=self._train_vars[key],
                font=("Segoe UI", 10),
                bg=THEME["entry_bg"], fg=THEME["fg"],
                insertbackground=THEME["fg"],
                relief=tk.FLAT, width=40,
            )
            entry.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
            tk.Button(
                row, text="Browse", font=("Segoe UI", 9),
                fg=THEME["btn_fg"], bg=THEME["btn_bg"],
                relief=tk.FLAT, padx=8,
                command=lambda k=key: self._browse_file(k),
            ).pack(side=tk.LEFT, padx=4)

        # --- Training parameters ---
        section3 = self._make_section(scroll_frame, "Training Parameters")
        params = [
            ("episodes", "Episodes:", "200", "int"),
            ("max_steps", "Max Steps/Episode:", "3600", "int"),
            ("seed", "Random Seed:", "42", "int"),
            ("lr", "DQN Learning Rate:", "0.001", "float"),
            ("epsilon_start", "Epsilon Start:", "1.0", "float"),
            ("epsilon_min", "Epsilon Min:", "0.05", "float"),
            ("epsilon_decay", "Epsilon Decay:", "0.995", "float"),
            ("ppo_lr", "PPO Learning Rate:", "0.0003", "float"),
            ("ppo_n_steps", "PPO Rollout Steps:", "512", "int"),
            ("ppo_n_epochs", "PPO Epochs:", "10", "int"),
            ("ppo_clip_epsilon", "PPO Clip Epsilon:", "0.2", "float"),
            ("alpha", "Alpha (throughput):", "0.4", "float"),
            ("beta", "Beta (queue):", "0.3", "float"),
            ("gamma_reward", "Gamma (wait-time):", "0.2", "float"),
            ("delta", "Delta (spillback):", "0.5", "float"),
        ]

        for key, label, default, _ in params:
            row = tk.Frame(section3, bg=THEME["card_bg"])
            row.pack(fill=tk.X, pady=1)
            tk.Label(
                row, text=label, font=("Segoe UI", 10),
                fg=THEME["fg"], bg=THEME["card_bg"], width=24, anchor="w",
            ).pack(side=tk.LEFT, padx=8)
            self._train_vars[key] = tk.StringVar(value=default)
            tk.Entry(
                row, textvariable=self._train_vars[key],
                font=("Segoe UI", 10),
                bg=THEME["entry_bg"], fg=THEME["fg"],
                insertbackground=THEME["fg"],
                relief=tk.FLAT, width=16,
            ).pack(side=tk.LEFT, padx=4)

        # --- GUI / Misc ---
        section4 = self._make_section(scroll_frame, "Options")
        self._train_vars["use_gui"] = tk.BooleanVar(value=False)
        tk.Checkbutton(
            section4, text="Launch SUMO-GUI (requires SUMO + display)",
            variable=self._train_vars["use_gui"],
            font=("Segoe UI", 10),
            fg=THEME["fg"], bg=THEME["card_bg"],
            selectcolor=THEME["bg3"],
            activebackground=THEME["card_bg"],
        ).pack(anchor="w", padx=8, pady=4)

        self._train_vars["checkpoint_dir"] = tk.StringVar(value="checkpoints")
        row = tk.Frame(section4, bg=THEME["card_bg"])
        row.pack(fill=tk.X, pady=2)
        tk.Label(
            row, text="Checkpoint Directory:", font=("Segoe UI", 10),
            fg=THEME["fg"], bg=THEME["card_bg"], width=24, anchor="w",
        ).pack(side=tk.LEFT, padx=8)
        tk.Entry(
            row, textvariable=self._train_vars["checkpoint_dir"],
            font=("Segoe UI", 10),
            bg=THEME["entry_bg"], fg=THEME["fg"],
            insertbackground=THEME["fg"],
            relief=tk.FLAT, width=30,
        ).pack(side=tk.LEFT, padx=4)

        # --- Action buttons ---
        btn_frame = tk.Frame(scroll_frame, bg=THEME["bg"])
        btn_frame.pack(fill=tk.X, padx=16, pady=12)

        self._train_btn = tk.Button(
            btn_frame, text="Start Training",
            font=("Segoe UI", 12, "bold"),
            fg="#ffffff", bg=THEME["bg3"],
            activebackground=THEME["accent"],
            activeforeground="#000",
            relief=tk.FLAT, padx=24, pady=8,
            command=self._start_training,
        )
        self._train_btn.pack(side=tk.LEFT, padx=8)

        self._stop_train_btn = tk.Button(
            btn_frame, text="Stop Training",
            font=("Segoe UI", 12, "bold"),
            fg=THEME["error"], bg=THEME["bg2"],
            activebackground=THEME["error"],
            relief=tk.FLAT, padx=24, pady=8,
            command=self._stop_training,
            state=tk.DISABLED,
        )
        self._stop_train_btn.pack(side=tk.LEFT, padx=8)

        # CLI preview
        cli_section = self._make_section(scroll_frame, "CLI Command Preview")
        self._cli_text = tk.Text(
            cli_section, height=4, font=("Consolas", 10),
            bg=THEME["entry_bg"], fg=THEME["accent"],
            insertbackground=THEME["fg"],
            relief=tk.FLAT, wrap=tk.WORD,
        )
        self._cli_text.pack(fill=tk.X, padx=8, pady=4)
        self._update_cli_preview()

        # Bind variable changes to update CLI preview
        for var in self._train_vars.values():
            var.trace_add("write", lambda *_a: self._update_cli_preview())

        # Training log output
        log_section = self._make_section(scroll_frame, "Training Log")
        self._train_log = scrolledtext.ScrolledText(
            log_section, height=10, font=("Consolas", 9),
            bg=THEME["entry_bg"], fg=THEME["fg"],
            insertbackground=THEME["fg"],
            relief=tk.FLAT, state=tk.DISABLED,
        )
        self._train_log.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    # -----------------------------------------------------------------------
    # OSM Import Tab
    # -----------------------------------------------------------------------

    def _build_osm_tab(self) -> None:
        """OSM Map Import panel."""
        frame = tk.Frame(self._notebook, bg=THEME["bg"])
        self._notebook.add(frame, text="  OSM Map Import  ")

        tk.Label(
            frame,
            text="Import Real-World Maps from OpenStreetMap",
            font=("Segoe UI", 14, "bold"),
            fg=THEME["accent"], bg=THEME["bg"],
        ).pack(anchor="w", padx=16, pady=(12, 4))

        tk.Label(
            frame,
            text=(
                "Search for any city or area, preview the map, and convert it "
                "to a SUMO network for simulation."
            ),
            font=("Segoe UI", 10),
            fg=THEME["fg2"], bg=THEME["bg"],
            wraplength=700, justify=tk.LEFT,
        ).pack(anchor="w", padx=16, pady=(0, 8))

        # Search bar
        search_frame = tk.Frame(frame, bg=THEME["bg2"])
        search_frame.pack(fill=tk.X, padx=16, pady=4)

        tk.Label(
            search_frame, text="Location:", font=("Segoe UI", 10),
            fg=THEME["fg"], bg=THEME["bg2"],
        ).pack(side=tk.LEFT, padx=8, pady=8)

        self._osm_query = tk.StringVar()
        entry = tk.Entry(
            search_frame, textvariable=self._osm_query,
            font=("Segoe UI", 11),
            bg=THEME["entry_bg"], fg=THEME["fg"],
            insertbackground=THEME["fg"],
            relief=tk.FLAT, width=40,
        )
        entry.pack(side=tk.LEFT, padx=4, pady=8, fill=tk.X, expand=True)
        entry.bind("<Return>", lambda _e: self._osm_search())

        tk.Button(
            search_frame, text="Search", font=("Segoe UI", 10, "bold"),
            fg=THEME["btn_fg"], bg=THEME["btn_bg"],
            relief=tk.FLAT, padx=16, pady=4,
            command=self._osm_search,
        ).pack(side=tk.LEFT, padx=8, pady=8)

        # Results list
        self._osm_results_frame = tk.Frame(frame, bg=THEME["bg"])
        self._osm_results_frame.pack(fill=tk.X, padx=16, pady=4)

        # Status and import
        status_frame = tk.Frame(frame, bg=THEME["bg"])
        status_frame.pack(fill=tk.X, padx=16, pady=4)

        self._osm_status = tk.Label(
            status_frame, text="", font=("Segoe UI", 10),
            fg=THEME["fg2"], bg=THEME["bg"],
        )
        self._osm_status.pack(anchor="w")

        # Import options
        opt_frame = tk.Frame(frame, bg=THEME["bg2"])
        opt_frame.pack(fill=tk.X, padx=16, pady=8)

        tk.Label(
            opt_frame, text="Vehicles:", font=("Segoe UI", 10),
            fg=THEME["fg"], bg=THEME["bg2"],
        ).pack(side=tk.LEFT, padx=8, pady=8)
        self._osm_vehicles = tk.StringVar(value="400")
        tk.Entry(
            opt_frame, textvariable=self._osm_vehicles,
            font=("Segoe UI", 10), width=8,
            bg=THEME["entry_bg"], fg=THEME["fg"],
            insertbackground=THEME["fg"], relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=4, pady=8)

        self._osm_import_btn = tk.Button(
            opt_frame, text="Import & Generate SUMO Files",
            font=("Segoe UI", 11, "bold"),
            fg="#ffffff", bg=THEME["bg3"],
            activebackground=THEME["accent"],
            relief=tk.FLAT, padx=20, pady=6,
            command=self._osm_import,
            state=tk.DISABLED,
        )
        self._osm_import_btn.pack(side=tk.RIGHT, padx=8, pady=8)

        # Import log
        self._osm_log = scrolledtext.ScrolledText(
            frame, height=8, font=("Consolas", 9),
            bg=THEME["entry_bg"], fg=THEME["fg"],
            insertbackground=THEME["fg"],
            relief=tk.FLAT, state=tk.DISABLED,
        )
        self._osm_log.pack(fill=tk.BOTH, expand=True, padx=16, pady=4)

        self._osm_selected = None

    # -----------------------------------------------------------------------
    # SUMO Setup Tab
    # -----------------------------------------------------------------------

    def _build_sumo_tab(self) -> None:
        """SUMO installation detection and configuration."""
        frame = tk.Frame(self._notebook, bg=THEME["bg"])
        self._notebook.add(frame, text="  SUMO Setup  ")

        tk.Label(
            frame,
            text="SUMO Installation",
            font=("Segoe UI", 14, "bold"),
            fg=THEME["accent"], bg=THEME["bg"],
        ).pack(anchor="w", padx=16, pady=(12, 4))

        info_text = (
            "SUMO (Simulation of Urban Mobility) is required for running "
            "traffic simulations. The desktop app can function in demo mode "
            "without SUMO, but real simulations need it installed.\n\n"
            "Download SUMO from: https://sumo.dlr.de/docs/Downloads.php"
        )
        tk.Label(
            frame, text=info_text, font=("Segoe UI", 10),
            fg=THEME["fg2"], bg=THEME["bg"],
            wraplength=700, justify=tk.LEFT,
        ).pack(anchor="w", padx=16, pady=(0, 12))

        # Status card
        status_card = tk.Frame(frame, bg=THEME["card_bg"], padx=20, pady=16)
        status_card.pack(fill=tk.X, padx=16, pady=4)

        self._sumo_status_label = tk.Label(
            status_card, text="Checking...",
            font=("Segoe UI", 12, "bold"),
            fg=THEME["fg"], bg=THEME["card_bg"],
        )
        self._sumo_status_label.pack(anchor="w")

        self._sumo_path_label = tk.Label(
            status_card, text="",
            font=("Consolas", 10),
            fg=THEME["fg2"], bg=THEME["card_bg"],
        )
        self._sumo_path_label.pack(anchor="w", pady=(4, 0))

        # Manual path entry
        path_frame = tk.Frame(frame, bg=THEME["bg2"])
        path_frame.pack(fill=tk.X, padx=16, pady=12)

        tk.Label(
            path_frame, text="SUMO_HOME:", font=("Segoe UI", 10),
            fg=THEME["fg"], bg=THEME["bg2"],
        ).pack(side=tk.LEFT, padx=8, pady=8)

        self._sumo_home_var = tk.StringVar(value=self._sumo_home or "")
        tk.Entry(
            path_frame, textvariable=self._sumo_home_var,
            font=("Segoe UI", 10), width=50,
            bg=THEME["entry_bg"], fg=THEME["fg"],
            insertbackground=THEME["fg"], relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=4, pady=8, fill=tk.X, expand=True)

        tk.Button(
            path_frame, text="Browse", font=("Segoe UI", 9),
            fg=THEME["btn_fg"], bg=THEME["btn_bg"],
            relief=tk.FLAT, padx=12,
            command=self._browse_sumo_home,
        ).pack(side=tk.LEFT, padx=4, pady=8)

        tk.Button(
            path_frame, text="Set & Detect", font=("Segoe UI", 10, "bold"),
            fg=THEME["btn_fg"], bg=THEME["btn_bg"],
            relief=tk.FLAT, padx=16,
            command=self._set_sumo_home,
        ).pack(side=tk.LEFT, padx=8, pady=8)

        # Download link
        link_frame = tk.Frame(frame, bg=THEME["bg"])
        link_frame.pack(anchor="w", padx=16, pady=8)

        dl_btn = tk.Button(
            link_frame,
            text="Download SUMO (opens browser)",
            font=("Segoe UI", 11),
            fg=THEME["accent"], bg=THEME["bg"],
            relief=tk.FLAT, cursor="hand2",
            command=lambda: webbrowser.open(
                "https://sumo.dlr.de/docs/Downloads.php"
            ),
        )
        dl_btn.pack(side=tk.LEFT)

    # -----------------------------------------------------------------------
    # About Tab
    # -----------------------------------------------------------------------

    def _build_about_tab(self) -> None:
        """Project information and credits."""
        frame = tk.Frame(self._notebook, bg=THEME["bg"])
        self._notebook.add(frame, text="  About  ")

        text = tk.Text(
            frame, font=("Segoe UI", 11),
            bg=THEME["bg"], fg=THEME["fg"],
            relief=tk.FLAT, wrap=tk.WORD,
            padx=20, pady=16,
        )
        text.pack(fill=tk.BOTH, expand=True)

        about_content = """
IUTMS - Intelligent Urban Traffic Management System
====================================================

Multi-Agent Deep Reinforcement Learning for Traffic Signal
Optimisation in Oversaturated Urban Networks


ALGORITHMS
----------
- DQN: Deep Q-Network with epsilon-greedy exploration, experience
  replay buffer, and periodic target network updates
- PPO: Proximal Policy Optimisation with actor-critic architecture,
  Generalised Advantage Estimation (GAE), and clipped surrogate objective


REWARD MODES
------------
- Composite: R = alpha*Throughput - beta*Queue - gamma*WaitTime - delta*Spillback
- Pressure: R = -|incoming_queue - outgoing_queue| / num_lanes


FEATURES
--------
- Multi-agent training with independent learners per intersection
- Real-time telemetry dashboard with live charts
- OpenStreetMap map import for any city worldwide
- Phase-time observations for enhanced state representation
- Checkpoint saving and loading
- CSV data export
- Demo mode (works without SUMO installation)


SIMULATION
----------
- SUMO 1.15+ via TraCI interface
- Configurable 3x3 grid or any OSM-imported city
- Oversaturated traffic scenarios
- CO2 emission tracking


REPOSITORY
----------
https://github.com/tbadrinath/MARLTSOIOSU


LICENSE
-------
MIT License
"""
        text.insert(tk.END, about_content)
        text.configure(state=tk.DISABLED)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _make_section(self, parent: tk.Widget, title: str) -> tk.Frame:
        """Create a titled card section."""
        outer = tk.Frame(parent, bg=THEME["bg"])
        outer.pack(fill=tk.X, padx=16, pady=6)
        tk.Label(
            outer, text=title, font=("Segoe UI", 11, "bold"),
            fg=THEME["accent"], bg=THEME["bg"],
        ).pack(anchor="w", pady=(0, 4))
        inner = tk.Frame(outer, bg=THEME["card_bg"], padx=8, pady=8)
        inner.pack(fill=tk.X)
        return inner

    def _browse_file(self, var_key: str) -> None:
        path = filedialog.askopenfilename(
            title=f"Select {var_key}",
            filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
        )
        if path:
            self._train_vars[var_key].set(path)

    def _browse_sumo_home(self) -> None:
        path = filedialog.askdirectory(title="Select SUMO_HOME directory")
        if path:
            self._sumo_home_var.set(path)

    def _set_sumo_home(self) -> None:
        path = self._sumo_home_var.get().strip()
        if path and Path(path).is_dir():
            os.environ["SUMO_HOME"] = path
            self._sumo_home = path
            self._update_sumo_status()
            messagebox.showinfo("SUMO", f"SUMO_HOME set to:\n{path}")
        else:
            messagebox.showerror("Error", "Invalid directory path.")

    def _update_sumo_status(self) -> None:
        if self._sumo_home:
            self._sumo_label.config(
                text=f"SUMO: {self._sumo_home}",
                fg=THEME["success"],
            )
            if hasattr(self, "_sumo_status_label"):
                self._sumo_status_label.config(
                    text="SUMO detected",
                    fg=THEME["success"],
                )
                self._sumo_path_label.config(text=self._sumo_home)
        else:
            self._sumo_label.config(
                text="SUMO: not found",
                fg=THEME["warning"],
            )
            if hasattr(self, "_sumo_status_label"):
                self._sumo_status_label.config(
                    text="SUMO not detected (demo mode available)",
                    fg=THEME["warning"],
                )
                self._sumo_path_label.config(
                    text="Set SUMO_HOME or install SUMO to run simulations"
                )

    def _update_cli_preview(self) -> None:
        if not hasattr(self, "_cli_text"):
            return
        algo = self._train_vars.get("algo", tk.StringVar(value="dqn")).get()
        reward = self._train_vars.get(
            "reward", tk.StringVar(value="composite")
        ).get()
        episodes = self._train_vars.get(
            "episodes", tk.StringVar(value="200")
        ).get()
        net = self._train_vars.get(
            "net_file", tk.StringVar(value="maps/grid.net.xml")
        ).get()
        route = self._train_vars.get(
            "route_file", tk.StringVar(value="maps/grid.rou.xml")
        ).get()

        cmd = (
            f"python -m simulation.trainer \\\n"
            f"  --algo {algo} --reward {reward} \\\n"
            f"  --net-file {net} \\\n"
            f"  --route-file {route} \\\n"
            f"  --episodes {episodes}"
        )

        self._cli_text.configure(state=tk.NORMAL)
        self._cli_text.delete("1.0", tk.END)
        self._cli_text.insert(tk.END, cmd)
        self._cli_text.configure(state=tk.DISABLED)

    def _log_training(self, msg: str) -> None:
        """Append a message to the training log."""
        self._train_log.configure(state=tk.NORMAL)
        self._train_log.insert(tk.END, msg + "\n")
        self._train_log.see(tk.END)
        self._train_log.configure(state=tk.DISABLED)

    def _log_osm(self, msg: str) -> None:
        """Append a message to the OSM log."""
        self._osm_log.configure(state=tk.NORMAL)
        self._osm_log.insert(tk.END, msg + "\n")
        self._osm_log.see(tk.END)
        self._osm_log.configure(state=tk.DISABLED)

    # -----------------------------------------------------------------------
    # Telemetry server
    # -----------------------------------------------------------------------

    def _start_telemetry_server(self) -> None:
        """Start the embedded Python telemetry server."""
        def _run():
            try:
                from desktop.telemetry_server import start_server
                self._server_running = True
                self.after(100, lambda: self._server_label.config(
                    text="Server: running (port 3001)",
                    fg=THEME["success"],
                ))
                start_server(host="127.0.0.1", port=3001)
            except ImportError:
                self.after(100, lambda: self._server_label.config(
                    text="Server: flask not installed",
                    fg=THEME["error"],
                ))
            except Exception as err:
                msg = str(err)
                self.after(100, lambda: self._server_label.config(
                    text=f"Server: error - {msg}",
                    fg=THEME["error"],
                ))

        self._server_thread = threading.Thread(
            target=_run, daemon=True, name="telemetry"
        )
        self._server_thread.start()

    # -----------------------------------------------------------------------
    # Demo mode
    # -----------------------------------------------------------------------

    def _toggle_demo(self) -> None:
        if self._demo_running:
            self._stop_demo()
        else:
            self._start_demo()

    def _start_demo(self) -> None:
        self._demo_running = True
        self._demo_step = 0
        self._demo_episode = 1
        self._demo_btn.config(text="Stop Demo", fg=THEME["error"])
        self._tick_demo()

    def _stop_demo(self) -> None:
        self._demo_running = False
        self._demo_btn.config(text="Start Demo", fg=THEME["btn_fg"])
        if self._chart_after_id is not None:
            self.after_cancel(self._chart_after_id)
            self._chart_after_id = None

    def _tick_demo(self) -> None:
        if not self._demo_running:
            return

        ep = min((self._demo_episode - 1) / 50.0, 1.0)
        phase = self._demo_step / 3600.0

        reward = -4 + 6 * ep + 3 * phase + (random.random() - 0.5) * 0.4
        speed = max(0, 2 + 8 * ep + 3 * phase + (random.random() - 0.5) * 1.5)
        cong = max(0, round(
            100 - 60 * ep + 40 * math.sin(phase * math.pi)
            + (random.random() - 0.5) * 8
        ))
        co2 = max(0, 8000 - 5000 * ep - 1000 * phase
                  + (random.random() - 0.5) * 400)

        self._add_metric(self._demo_step, reward, speed, cong, co2)

        self._demo_step += 10
        if self._demo_step >= 3600:
            self._demo_step = 0
            self._demo_episode = (
                1 if self._demo_episode >= 50
                else self._demo_episode + 1
            )

        self._chart_after_id = self.after(150, self._tick_demo)

    # -----------------------------------------------------------------------
    # Chart updates
    # -----------------------------------------------------------------------

    def _add_metric(
        self,
        step: int,
        reward: float,
        speed: float,
        congestion: float,
        co2: float,
    ) -> None:
        """Add a data point and update charts + KPIs."""
        max_pts = 200
        for key, val in [
            ("steps", step),
            ("rewards", reward),
            ("speeds", speed),
            ("congestion", congestion),
            ("co2", co2),
        ]:
            self._chart_data[key].append(val)
            if len(self._chart_data[key]) > max_pts:
                self._chart_data[key] = self._chart_data[key][-max_pts:]

        # Update KPI cards
        self._kpi_labels["reward"].config(text=f"{reward:.2f}")
        self._kpi_labels["speed"].config(text=f"{speed:.1f}")
        self._kpi_labels["congestion"].config(text=f"{int(congestion)}")
        self._kpi_labels["co2"].config(text=f"{co2:.0f}")

        # Update matplotlib charts (every 5 points to avoid lag)
        if self._has_matplotlib and len(self._chart_data["steps"]) % 5 == 0:
            self._redraw_charts()

    def _redraw_charts(self) -> None:
        """Redraw all four matplotlib charts."""
        if not self._has_matplotlib:
            return

        data_keys = ["rewards", "speeds", "congestion", "co2"]
        colors = [THEME["accent"], THEME["accent2"], THEME["warning"], THEME["error"]]
        steps = self._chart_data["steps"]

        for i, (key, color) in enumerate(zip(data_keys, colors)):
            ax = self._axes[i]
            ax.clear()
            ax.set_facecolor(THEME["bg2"])
            vals = self._chart_data[key]
            ax.plot(steps, vals, color=color, linewidth=1.5)
            ax.fill_between(
                steps, vals, alpha=0.15, color=color,
            )
            ax.tick_params(colors=THEME["fg2"], labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(THEME["bg3"])
            ax.grid(True, alpha=0.15, color=THEME["fg2"])

        titles = [
            ("Reward per Step", THEME["accent"]),
            ("Avg Speed (m/s)", THEME["accent2"]),
            ("Congestion (vehicles)", THEME["warning"]),
            ("CO2 Emissions (mg/s)", THEME["error"]),
        ]
        for i, (title, color) in enumerate(titles):
            self._axes[i].set_title(title, color=color, fontsize=9, fontweight="bold")

        self._canvas.draw_idle()

    def _clear_charts(self) -> None:
        for key in self._chart_data:
            self._chart_data[key] = []
        for lbl in self._kpi_labels.values():
            lbl.config(text="--")
        if self._has_matplotlib:
            self._redraw_charts()

    def _export_csv(self) -> None:
        if not self._chart_data["steps"]:
            messagebox.showinfo("Export", "No data to export.")
            return

        path = filedialog.asksaveasfilename(
            title="Export CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"iutms_metrics_{int(time.time())}.csv",
        )
        if not path:
            return

        lines = ["step,total_reward,avg_speed,vehicles_in_network,co2_emissions"]
        for i in range(len(self._chart_data["steps"])):
            lines.append(
                f"{self._chart_data['steps'][i]},"
                f"{self._chart_data['rewards'][i]:.4f},"
                f"{self._chart_data['speeds'][i]:.2f},"
                f"{self._chart_data['congestion'][i]:.0f},"
                f"{self._chart_data['co2'][i]:.1f}"
            )
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        messagebox.showinfo("Export", f"Data exported to:\n{path}")

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def _start_training(self) -> None:
        if self._training_process is not None:
            messagebox.showwarning("Training", "Training is already running.")
            return

        if not self._sumo_home:
            if not messagebox.askyesno(
                "SUMO Not Found",
                "SUMO is not detected. Training requires SUMO.\n\n"
                "Do you want to continue anyway?",
            ):
                return

        self._train_btn.config(state=tk.DISABLED)
        self._stop_train_btn.config(state=tk.NORMAL)
        self._log_training("Starting training...")

        # Build command
        algo = self._train_vars["algo"].get()
        reward = self._train_vars["reward"].get()
        cmd = [
            sys.executable, "-m", "simulation.trainer",
            "--algo", algo,
            "--reward", reward,
            "--net-file", self._train_vars["net_file"].get(),
            "--route-file", self._train_vars["route_file"].get(),
            "--episodes", self._train_vars["episodes"].get(),
            "--max-steps", self._train_vars["max_steps"].get(),
            "--lr", self._train_vars["lr"].get(),
            "--epsilon-start", self._train_vars["epsilon_start"].get(),
            "--epsilon-min", self._train_vars["epsilon_min"].get(),
            "--epsilon-decay", self._train_vars["epsilon_decay"].get(),
            "--ppo-lr", self._train_vars["ppo_lr"].get(),
            "--ppo-n-steps", self._train_vars["ppo_n_steps"].get(),
            "--ppo-n-epochs", self._train_vars["ppo_n_epochs"].get(),
            "--ppo-clip-epsilon", self._train_vars["ppo_clip_epsilon"].get(),
            "--alpha", self._train_vars["alpha"].get(),
            "--beta", self._train_vars["beta"].get(),
            "--gamma-reward", self._train_vars["gamma_reward"].get(),
            "--delta", self._train_vars["delta"].get(),
            "--checkpoint-dir", self._train_vars["checkpoint_dir"].get(),
            "--seed", self._train_vars["seed"].get(),
            "--telemetry-url", "http://localhost:3001/api/metrics",
        ]

        if self._train_vars["phase_obs"].get():
            cmd.append("--phase-obs")
        if self._train_vars["use_gui"].get():
            cmd.append("--gui")

        self._log_training(f"Command: {' '.join(cmd)}")

        def _run_training():
            try:
                self._training_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                )
                for line in iter(self._training_process.stdout.readline, ""):
                    self.after(0, self._log_training, line.rstrip())
                self._training_process.wait()
                self.after(0, self._log_training, "Training finished.")
            except Exception as exc:
                self.after(0, self._log_training, f"Error: {exc}")
            finally:
                self._training_process = None
                self.after(0, lambda: self._train_btn.config(state=tk.NORMAL))
                self.after(
                    0, lambda: self._stop_train_btn.config(state=tk.DISABLED)
                )

        threading.Thread(
            target=_run_training, daemon=True, name="training"
        ).start()

    def _stop_training(self) -> None:
        if self._training_process is not None:
            self._training_process.terminate()
            self._log_training("Training stopped by user.")

    # -----------------------------------------------------------------------
    # OSM Import
    # -----------------------------------------------------------------------

    def _osm_search(self) -> None:
        query = self._osm_query.get().strip()
        if not query:
            return

        self._osm_status.config(
            text="Searching...", fg=THEME["warning"]
        )
        self._log_osm(f"Searching for: {query}")

        def _search():
            try:
                import requests as req_lib
                params = {"q": query, "format": "json", "limit": "5"}
                headers = {
                    "User-Agent": "IUTMS-TrafficSim/1.0 "
                    "(https://github.com/tbadrinath/MARLTSOIOSU)"
                }
                resp = req_lib.get(
                    "https://nominatim.openstreetmap.org/search",
                    params=params, headers=headers, timeout=15,
                )
                results = resp.json()
                self.after(0, self._display_osm_results, results)
            except Exception as err:
                msg = str(err)
                self.after(
                    0,
                    lambda: self._osm_status.config(
                        text=f"Search failed: {msg}", fg=THEME["error"]
                    ),
                )

        threading.Thread(target=_search, daemon=True).start()

    def _display_osm_results(self, results: list) -> None:
        # Clear previous results
        for widget in self._osm_results_frame.winfo_children():
            widget.destroy()

        if not results:
            self._osm_status.config(
                text="No results found.", fg=THEME["warning"]
            )
            return

        self._osm_status.config(
            text=f"Found {len(results)} results. Click to select:",
            fg=THEME["accent"],
        )

        for i, r in enumerate(results[:5]):
            name = r.get("display_name", "Unknown")
            btn = tk.Button(
                self._osm_results_frame,
                text=f"  {name[:80]}{'...' if len(name) > 80 else ''}",
                font=("Segoe UI", 9),
                fg=THEME["fg"], bg=THEME["card_bg"],
                activebackground=THEME["bg3"],
                relief=tk.FLAT, anchor="w",
                command=lambda res=r: self._select_osm_result(res),
            )
            btn.pack(fill=tk.X, pady=1)

    def _select_osm_result(self, result: dict) -> None:
        self._osm_selected = result
        name = result.get("display_name", "Unknown")
        bb = result.get("boundingbox", [])
        self._osm_status.config(
            text=f"Selected: {name[:60]} | bbox={bb}",
            fg=THEME["success"],
        )
        self._osm_import_btn.config(state=tk.NORMAL)
        self._log_osm(f"Selected: {name}")

    def _osm_import(self) -> None:
        if not self._osm_selected:
            return

        location = self._osm_selected.get("display_name", "")
        num_vehicles = int(self._osm_vehicles.get() or 400)

        self._osm_import_btn.config(state=tk.DISABLED)
        self._osm_status.config(
            text="Importing... (this may take a minute)",
            fg=THEME["warning"],
        )
        self._log_osm(f"Starting import for: {location}")

        def _import():
            try:
                sys.path.insert(0, str(PROJECT_ROOT))
                from simulation.osm_importer import import_map

                sanitised = "".join(
                    c if c.isalnum() or c in ("_", "-") else "_"
                    for c in location
                )[:64]
                output_dir = str(PROJECT_ROOT / "maps" / "osm" / sanitised)

                result = import_map(
                    location, output_dir,
                    num_vehicles=num_vehicles, seed=42,
                )

                self.after(0, self._log_osm, "Import complete!")
                self.after(
                    0, self._log_osm,
                    f"  Network: {result.get('net_file', 'N/A')}",
                )
                self.after(
                    0, self._log_osm,
                    f"  Routes: {result.get('route_file', 'N/A')}",
                )

                # Auto-fill training fields
                net = result.get("net_file", "")
                route = result.get("route_file", "")
                if net:
                    self.after(
                        0,
                        lambda: self._train_vars["net_file"].set(net),
                    )
                if route:
                    self.after(
                        0,
                        lambda: self._train_vars["route_file"].set(route),
                    )

                self.after(0, lambda: self._osm_status.config(
                    text="Import complete! Files set in Training tab.",
                    fg=THEME["success"],
                ))
            except Exception as err:
                msg = str(err)
                self.after(0, self._log_osm, f"Import failed: {msg}")
                self.after(0, lambda: self._osm_status.config(
                    text=f"Import failed: {msg}",
                    fg=THEME["error"],
                ))
            finally:
                self.after(
                    0,
                    lambda: self._osm_import_btn.config(state=tk.NORMAL),
                )

        threading.Thread(target=_import, daemon=True).start()

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def destroy(self) -> None:
        """Clean up before exit."""
        self._demo_running = False
        if self._chart_after_id is not None:
            self.after_cancel(self._chart_after_id)
        if self._training_process is not None:
            self._training_process.terminate()
        super().destroy()


def main() -> None:
    """Entry point for the desktop application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    app = IUTMSApp()
    app.mainloop()


if __name__ == "__main__":
    main()
