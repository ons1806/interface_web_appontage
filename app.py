import json
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from simulation_runner import SimulationRunner


# ==================================================
# Page config
# ==================================================
st.set_page_config(
    page_title="Dashboard - Appontage autonome de drone",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# CSS - dashboard clair inspiré du mockup
# ==================================================
st.markdown(
    """
<style>
:root {
    --bg: #f5f7fb;
    --panel: #ffffff;
    --panel-soft: #f8fafc;
    --border: #dbe3ef;
    --text: #0f172a;
    --muted: #64748b;
    --blue: #2563eb;
    --blue-soft: #eff6ff;
    --green: #16a34a;
    --red: #dc2626;
    --orange: #f59e0b;
    --purple: #7c3aed;
}

html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", Arial, sans-serif;
}

.stApp {
    background: var(--bg);
}

.block-container {
    max-width: 1680px;
    padding-top: 1.1rem;
    padding-bottom: 1.5rem;
}

/* Sidebar claire */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

.sidebar-title {
    font-size: 22px;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 4px;
}

.sidebar-subtitle {
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 18px;
}

.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px;
    margin-bottom: 6px;
    border-radius: 10px;
    color: #334155;
    font-size: 14px;
    border: 1px solid transparent;
}

.nav-item.active {
    background: var(--blue);
    color: white;
    font-weight: 700;
    box-shadow: 0 8px 18px rgba(37, 99, 235, 0.22);
}

.nav-item:not(.active) {
    background: #f8fafc;
    border-color: #edf2f7;
}

.section-title {
    font-size: 12px;
    font-weight: 800;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-top: 18px;
    margin-bottom: 8px;
    padding-bottom: 7px;
    border-bottom: 1px solid #e2e8f0;
}

/* Inputs */
.stSelectbox [data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {
    background: white;
    border-radius: 10px;
}

section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 800;
    padding: 0.75rem 1rem;
    box-shadow: 0 8px 18px rgba(37,99,235,0.25);
}

section[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 22px rgba(37,99,235,0.30);
}

/* Header */
.top-header {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 7px solid var(--blue);
    border-radius: 18px;
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow: 0 10px 28px rgba(15,23,42,0.06);
}

.header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
}

.header-title {
    font-size: 30px;
    font-weight: 850;
    color: var(--text);
    margin: 0 0 7px 0;
}

.header-subtitle {
    color: #475569;
    font-size: 15px;
    margin: 0;
}

.header-pill {
    display: inline-block;
    background: var(--blue-soft);
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 12px;
    font-weight: 800;
    margin-left: 8px;
}

/* Cards */
.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 15px 16px;
    box-shadow: 0 8px 24px rgba(15,23,42,0.055);
    height: 100%;
}

.card-title {
    font-size: 17px;
    font-weight: 850;
    color: var(--text);
    margin: 0 0 12px 0;
}

.card-subtitle {
    color: var(--muted);
    font-size: 12px;
    margin-top: -7px;
    margin-bottom: 8px;
}

/* Metrics */
.metric-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 8px 24px rgba(15,23,42,0.055);
    min-height: 104px;
}

.metric-label {
    color: #475569;
    font-size: 13px;
    margin-bottom: 8px;
}

.metric-value {
    color: var(--text);
    font-size: 31px;
    font-weight: 900;
    letter-spacing: -0.03em;
    line-height: 1;
}

.metric-note {
    font-size: 13px;
    color: var(--muted);
    margin-top: 8px;
}

.metric-success {
    color: var(--green);
    font-weight: 900;
}

.metric-danger {
    color: var(--red);
    font-weight: 900;
}

/* Remove excessive Streamlit gaps */
div[data-testid="stVerticalBlock"] {
    gap: 0.85rem;
}

div[data-testid="stHorizontalBlock"] {
    gap: 1rem;
}

/* Containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--border) !important;
    border-radius: 16px !important;
    background: white !important;
    box-shadow: 0 8px 24px rgba(15,23,42,0.055);
}

/* Tables */
.dataframe {
    font-size: 12px !important;
}

.status-badge {
    display: inline-block;
    padding: 8px 13px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 800;
}

.status-idle { background: #f1f5f9; color: #475569; }
.status-running { background: #dbeafe; color: #1d4ed8; }
.status-success { background: #dcfce7; color: #15803d; }
.status-error { background: #fee2e2; color: #b91c1c; }

.export-button-note {
    color: var(--muted);
    font-size: 13px;
    margin-top: -3px;
    margin-bottom: 8px;
}
</style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# Sidebar controls
# ==================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">Panneau de contrôle</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Configuration de la simulation PPO</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item active">Vue d’ensemble</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">Simulation PyBullet</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">Trajectoire 3D</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">Résultats temporels</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">Export des résultats</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Modèle et scénario</div>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Architecture PPO",
        [
            "PPO-PID : correction de position",
            "PPO-PID : correction de vitesse",
            "PPO-MPC-PID : correction de vitesse",
            "PPO-SMC-PID : correction de vitesse",
        ],
        index=0,
    )

    scenario_mode = st.selectbox(
        "Mode de scénario",
        [1, 2, 3, 4, 5],
        index=4,
    )

    st.markdown('<div class="section-title">Environnement</div>', unsafe_allow_html=True)
    wind_enabled = st.checkbox("Activer le vent Dryden", value=False)

    st.markdown('<div class="section-title">Simulation</div>', unsafe_allow_html=True)
    max_steps = st.slider("Nombre maximal de pas", 100, 3000, 1000, 100)
    frame_interval = st.slider("Intervalle de mise à jour", 10, 100, 30, 10)

    plot_interval = frame_interval

    run_button = st.button("Lancer la simulation", type="primary")

    st.markdown("---")
    reset_requested = st.button("Réinitialiser les résultats")

    with st.expander("Aide", expanded=False):
        st.markdown(
            """
Cette interface lance une simulation PyBullet en mode DIRECT et affiche les résultats du modèle PPO sélectionné.

Les courbes et la trajectoire 3D sont synchronisées avec l'image PyBullet afin d'éviter le retard visuel.
            """
        )


if reset_requested:
    for key in ["df", "summary", "model_choice", "scenario_mode", "wind_enabled"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


if model_choice == "PPO-PID : correction de position":
    model_type = "position"
elif model_choice == "PPO-PID : correction de vitesse":
    model_type = "vitesse"
elif model_choice == "PPO-SMC-PID : correction de vitesse":
    model_type = "vitesse_smc"
else:
    model_type = "vitesse_mpc"


# ==================================================
# Plot constants
# ==================================================
PLOTLY_TEMPLATE = "plotly_white"
DRONE_COLOR = "#2563eb"
PLATFORM_COLOR = "#ef4444"
REWARD_COLOR = "#7c3aed"
ACTION_COLORS = ["#2563eb", "#f97316", "#16a34a"]


# ==================================================
# Plot functions
# ==================================================
def _empty_fig(title: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=title, font=dict(size=14)),
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _sample_df(df: pd.DataFrame, max_points: int = 350) -> pd.DataFrame:
    if df is None or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    sampled = df.iloc[::step].copy()
    if sampled.index[-1] != df.index[-1]:
        sampled = pd.concat([sampled, df.iloc[[-1]]])
    return sampled


def plot_3d_trajectory(df: Optional[pd.DataFrame]) -> go.Figure:
    if df is None or len(df) == 0:
        return _empty_fig("Trajectoire 3D - drone et plateforme", height=500)

    fig = go.Figure()
    df_plot = _sample_df(df, max_points=350)

    has_drone = {"drone_x", "drone_y", "drone_z"}.issubset(df.columns)
    has_platform = {"platform_x", "platform_y", "platform_z"}.issubset(df.columns)

    if has_drone:
        fig.add_trace(go.Scatter3d(
            x=df_plot["drone_y"],
            y=df_plot["drone_x"],
            z=df_plot["drone_z"],
            mode="lines",
            name="Trajectoire drone",
            line=dict(color=DRONE_COLOR, width=6),
        ))
        fig.add_trace(go.Scatter3d(
            x=[df["drone_y"].iloc[-1]],
            y=[df["drone_x"].iloc[-1]],
            z=[df["drone_z"].iloc[-1]],
            mode="markers",
            name="Drone actuel",
            marker=dict(size=8, color=DRONE_COLOR, symbol="x"),
        ))

    if has_platform:
        fig.add_trace(go.Scatter3d(
            x=df_plot["platform_y"],
            y=df_plot["platform_x"],
            z=df_plot["platform_z"],
            mode="lines",
            name="Trajectoire plateforme",
            line=dict(color=PLATFORM_COLOR, width=5, dash="dash"),
        ))
        fig.add_trace(go.Scatter3d(
            x=[df["platform_y"].iloc[-1]],
            y=[df["platform_x"].iloc[-1]],
            z=[df["platform_z"].iloc[-1]],
            mode="markers",
            name="Plateforme actuelle",
            marker=dict(size=8, color="#7c3aed", symbol="diamond"),
        ))

    if has_drone and has_platform:
        fig.add_trace(go.Scatter3d(
            x=[df["drone_y"].iloc[-1], df["platform_y"].iloc[-1]],
            y=[df["drone_x"].iloc[-1], df["platform_x"].iloc[-1]],
            z=[df["drone_z"].iloc[-1], df["platform_z"].iloc[-1]],
            mode="lines",
            name="Erreur instantanée",
            line=dict(color="#f59e0b", width=2, dash="dot"),
        ))

    all_x, all_y, all_z = [], [], []
    if has_drone:
        all_x += df["drone_y"].dropna().tolist()
        all_y += df["drone_x"].dropna().tolist()
        all_z += df["drone_z"].dropna().tolist()
    if has_platform:
        all_x += df["platform_y"].dropna().tolist()
        all_y += df["platform_x"].dropna().tolist()
        all_z += df["platform_z"].dropna().tolist()

    def safe_range(vals, pad=0.14):
        if not vals:
            return None
        mn, mx = min(vals), max(vals)
        d = max(mx - mn, 0.1)
        return [mn - d * pad, mx + d * pad]

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Trajectoire 3D - drone et plateforme", font=dict(size=14)),
        height=500,
        margin=dict(l=0, r=0, t=45, b=0),
        paper_bgcolor="white",
        scene=dict(
            xaxis=dict(title="Y (m)", backgroundcolor="#f8fafc", gridcolor="#dbe3ef", showbackground=True, range=safe_range(all_x)),
            yaxis=dict(title="X (m)", backgroundcolor="#f8fafc", gridcolor="#dbe3ef", showbackground=True, range=safe_range(all_y)),
            zaxis=dict(title="Z (m)", backgroundcolor="#f8fafc", gridcolor="#dbe3ef", showbackground=True, range=safe_range(all_z)),
            camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.85), up=dict(x=0, y=0, z=1)),
            aspectmode="cube",
        ),
        legend=dict(
            x=0.01,
            y=0.98,
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="#dbe3ef",
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    return fig


def plot_error(df: Optional[pd.DataFrame]) -> go.Figure:
    fig = _empty_fig("Erreur de position XY et Z relatif", height=290)
    if df is not None and len(df) > 0:
        fig.add_trace(go.Scatter(x=df["step"], y=df["xy_error"], name="Erreur XY", line=dict(color=DRONE_COLOR, width=2.2)))
        fig.add_trace(go.Scatter(x=df["step"], y=np.abs(df["z_rel"]), name="|Z relatif|", line=dict(color=PLATFORM_COLOR, width=2.2)))
        fig.add_hline(y=0.15, line_dash="dot", line_color="#94a3b8", annotation_text="seuil")
    fig.update_layout(xaxis_title="Pas de simulation", yaxis_title="Erreur (m)", legend=dict(font=dict(size=11)))
    return fig


def plot_reward(df: Optional[pd.DataFrame]) -> go.Figure:
    fig = _empty_fig("Récompense cumulée", height=290)
    if df is not None and len(df) > 0:
        fig.add_trace(go.Scatter(x=df["step"], y=df["reward"], name="Reward instantanée", line=dict(color="#94a3b8", width=1.3)))
        fig.add_trace(go.Scatter(x=df["step"], y=df["total_reward"], name="Reward cumulée", line=dict(color=REWARD_COLOR, width=2.8)))
    fig.update_layout(xaxis_title="Pas de simulation", yaxis_title="Reward", legend=dict(font=dict(size=11)))
    return fig


def plot_actions(df: Optional[pd.DataFrame]) -> go.Figure:
    fig = _empty_fig("Actions PPO", height=290)
    if df is not None and len(df) > 0:
        for i, col in enumerate(["action_1", "action_2", "action_3"]):
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df["step"], y=df[col], name=f"Action {i+1}", line=dict(color=ACTION_COLORS[i], width=1.9)))
    fig.update_layout(xaxis_title="Pas de simulation", yaxis_title="Valeur", legend=dict(font=dict(size=11)))
    return fig


def plot_xy_distance(df: Optional[pd.DataFrame]) -> go.Figure:
    fig = _empty_fig("Distance drone-plateforme", height=290)
    if df is not None and {"drone_x", "drone_y", "platform_x", "platform_y"}.issubset(df.columns):
        dist = np.sqrt((df["drone_x"] - df["platform_x"]) ** 2 + (df["drone_y"] - df["platform_y"]) ** 2)
        fig.add_trace(go.Scatter(x=df["step"], y=dist, name="Distance XY", line=dict(color="#f59e0b", width=2.4)))
        fig.add_hline(y=0.15, line_dash="dot", line_color="#94a3b8", annotation_text="seuil")
    fig.update_layout(xaxis_title="Pas de simulation", yaxis_title="Distance (m)")
    return fig


def render_metric_card(label: str, value: str, note: str = "", success: Optional[bool] = None):
    cls = "metric-value"
    if success is True:
        cls += " metric-success"
    elif success is False:
        cls += " metric-danger"
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">{label}</div>
    <div class="{cls}">{value}</div>
    <div class="metric-note">{note}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# Header
# ==================================================
st.markdown(
    f"""
<div class="top-header">
    <div class="header-row">
        <div>
            <div class="header-title">Dashboard - Appontage autonome de drone</div>
            <p class="header-subtitle">Visualisation et analyse des performances des architectures PPO avec PyBullet, trajectoire 3D et courbes en temps réel.</p>
        </div>
        <div>
            <span class="header-pill">Mode {scenario_mode}</span>
            <span class="header-pill">{'Vent activé' if wind_enabled else 'Vent désactivé'}</span>
        </div>
    </div>
</div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# Metric placeholders
# ==================================================
metric_cols = st.columns(5)
metric_boxes = [c.empty() for c in metric_cols]

with metric_boxes[0]:
    render_metric_card("Statut", "En attente", "", None)
with metric_boxes[1]:
    render_metric_card("Erreur XY finale", "---", "m")
with metric_boxes[2]:
    render_metric_card("Erreur Z finale", "---", "m")
with metric_boxes[3]:
    render_metric_card("Récompense totale", "---", "")
with metric_boxes[4]:
    render_metric_card("Durée", "---", "pas de simulation")


# ==================================================
# Main layout
# ==================================================
left_col, sim_col, traj_col = st.columns([0.85, 1.45, 1.60])

with left_col:
    with st.container(border=True):
        st.markdown('<div class="card-title">Configuration</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
| Paramètre | Valeur |
|---|---|
| Architecture | {model_choice} |
| Scénario | Mode {scenario_mode} |
| Vent Dryden | {'Activé' if wind_enabled else 'Désactivé'} |
| Pas maximal | {max_steps} |
| Intervalle | {frame_interval} pas |
            """
        )
        status_placeholder = st.empty()
        status_placeholder.markdown('<span class="status-badge status-idle">En attente</span>', unsafe_allow_html=True)
        progress_bar = st.progress(0)

with sim_col:
    with st.container(border=True):
        st.markdown('<div class="card-title">Simulation PyBullet</div>', unsafe_allow_html=True)
        image_placeholder = st.empty()
        image_placeholder.info("Lancez la simulation depuis le panneau de contrôle.")
        st.caption("Vue caméra générée par PyBullet en mode DIRECT.")

with traj_col:
    with st.container(border=True):
        st.markdown('<div class="card-title">Trajectoire 3D</div>', unsafe_allow_html=True)
        trajectory3d_placeholder = st.empty()
        trajectory3d_placeholder.plotly_chart(plot_3d_trajectory(None), use_container_width=True)


# ==================================================
# Plots
# ==================================================
plot_cols = st.columns(4)

with plot_cols[0]:
    with st.container(border=True):
        st.markdown('<div class="card-title">Erreur de position</div>', unsafe_allow_html=True)
        error_placeholder = st.empty()
        error_placeholder.plotly_chart(plot_error(None), use_container_width=True)

with plot_cols[1]:
    with st.container(border=True):
        st.markdown('<div class="card-title">Récompense</div>', unsafe_allow_html=True)
        reward_placeholder = st.empty()
        reward_placeholder.plotly_chart(plot_reward(None), use_container_width=True)

with plot_cols[2]:
    with st.container(border=True):
        st.markdown('<div class="card-title">Actions PPO</div>', unsafe_allow_html=True)
        actions_placeholder = st.empty()
        actions_placeholder.plotly_chart(plot_actions(None), use_container_width=True)

with plot_cols[3]:
    with st.container(border=True):
        st.markdown('<div class="card-title">Distance drone-plateforme</div>', unsafe_allow_html=True)
        dist_placeholder = st.empty()
        dist_placeholder.plotly_chart(plot_xy_distance(None), use_container_width=True)


# ==================================================
# Run simulation
# ==================================================
if run_button:
    status_placeholder.markdown('<span class="status-badge status-running">Initialisation</span>', unsafe_allow_html=True)
    image_placeholder.empty()

    try:
        runner = SimulationRunner()
        runner.load_env_and_model(model_type=model_type, scenario_mode=scenario_mode, wind_enabled=wind_enabled)
    except Exception as e:
        status_placeholder.markdown('<span class="status-badge status-error">Erreur de chargement</span>', unsafe_allow_html=True)
        st.error(f"Erreur lors du chargement du modèle : {e}")
        st.stop()

    live_data = []

    def update_metric_cards(row=None, summary=None):
        if summary is not None:
            ok = bool(summary.get("success", False))
            xy = float(summary.get("final_xy_error", np.nan))
            z = float(summary.get("final_z_rel", np.nan))
            total = float(summary.get("total_reward", np.nan))
            ep_len = summary.get("episode_length", "---")
        elif row is not None:
            ok = bool(row.get("success", False))
            xy = float(row.get("xy_error", np.nan))
            z = float(row.get("z_rel", np.nan))
            total = float(row.get("total_reward", np.nan))
            ep_len = row.get("step", "---")
        else:
            ok, xy, z, total, ep_len = False, np.nan, np.nan, np.nan, "---"

        with metric_boxes[0]:
            render_metric_card("Statut", "Succès" if ok else "En cours", "", ok if ok else None)
        with metric_boxes[1]:
            render_metric_card("Erreur XY finale", f"{xy:.3f} m" if not np.isnan(xy) else "---", "seuil 0.15 m")
        with metric_boxes[2]:
            render_metric_card("Erreur Z finale", f"{z:.3f} m" if not np.isnan(z) else "---", "")
        with metric_boxes[3]:
            render_metric_card("Récompense totale", f"{total:.2f}" if not np.isnan(total) else "---", "")
        with metric_boxes[4]:
            render_metric_card("Durée", str(ep_len), "pas de simulation")

    def update_frame(frame, step):
        image_placeholder.image(frame, caption=f"PyBullet - step {step}", use_container_width=True)
        if len(live_data) > 0:
            df_live = pd.DataFrame(live_data)
            trajectory3d_placeholder.plotly_chart(plot_3d_trajectory(df_live), use_container_width=True)

    def update_step(row, step):
        live_data.append(row)
        progress_bar.progress(min(1.0, step / max_steps))

        if step % plot_interval != 0:
            return

        df_live = pd.DataFrame(live_data)
        update_metric_cards(row=row)

        error_placeholder.plotly_chart(plot_error(df_live), use_container_width=True)
        reward_placeholder.plotly_chart(plot_reward(df_live), use_container_width=True)
        actions_placeholder.plotly_chart(plot_actions(df_live), use_container_width=True)
        dist_placeholder.plotly_chart(plot_xy_distance(df_live), use_container_width=True)

    status_placeholder.markdown('<span class="status-badge status-running">Simulation en cours</span>', unsafe_allow_html=True)

    try:
        result = runner.run_episode(
            max_steps=max_steps,
            capture_frames=True,
            frame_interval=frame_interval,
            on_frame=update_frame,
            on_step=update_step,
        )
    except Exception as e:
        status_placeholder.markdown('<span class="status-badge status-error">Erreur de simulation</span>', unsafe_allow_html=True)
        st.error(f"Erreur pendant la simulation : {e}")
        st.stop()

    df = result[0]
    summary = result[1]

    st.session_state["df"] = df
    st.session_state["summary"] = summary
    st.session_state["model_choice"] = model_choice
    st.session_state["scenario_mode"] = scenario_mode
    st.session_state["wind_enabled"] = wind_enabled

    progress_bar.progress(1.0)
    status_placeholder.markdown('<span class="status-badge status-success">Simulation terminée</span>', unsafe_allow_html=True)

    update_metric_cards(summary=summary)
    trajectory3d_placeholder.plotly_chart(plot_3d_trajectory(df), use_container_width=True)
    error_placeholder.plotly_chart(plot_error(df), use_container_width=True)
    reward_placeholder.plotly_chart(plot_reward(df), use_container_width=True)
    actions_placeholder.plotly_chart(plot_actions(df), use_container_width=True)
    dist_placeholder.plotly_chart(plot_xy_distance(df), use_container_width=True)


# ==================================================
# Results and export
# ==================================================
if "df" in st.session_state:
    df = st.session_state["df"]
    summary = st.session_state["summary"]
    stored_model_choice = st.session_state.get("model_choice", model_choice)
    stored_scenario_mode = st.session_state.get("scenario_mode", scenario_mode)
    stored_wind_enabled = st.session_state.get("wind_enabled", wind_enabled)

    st.markdown("---")
    st.markdown("## Synthèse et export")

    res_col1, res_col2 = st.columns([2.2, 1])

    with res_col1:
        with st.container(border=True):
            st.markdown('<div class="card-title">Tableau des données enregistrées</div>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, height=320)

    with res_col2:
        with st.container(border=True):
            st.markdown('<div class="card-title">Résumé de l’épisode</div>', unsafe_allow_html=True)
            ok = bool(summary.get("success", False))
            badge_bg = "#dcfce7" if ok else "#fee2e2"
            badge_color = "#15803d" if ok else "#b91c1c"
            badge_text = "Atterrissage réussi" if ok else "Atterrissage échoué"
            st.markdown(
                f"""
<div style="text-align:center; padding:14px; background:{badge_bg}; color:{badge_color}; border-radius:12px; font-weight:900; margin-bottom:12px;">
    {badge_text}
</div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
| Métrique | Valeur |
|---|---|
| Erreur XY finale | `{summary.get('final_xy_error', 0):.4f} m` |
| z relatif final | `{summary.get('final_z_rel', 0):.4f} m` |
| Reward totale | `{summary.get('total_reward', 0):.2f}` |
| Durée épisode | `{summary.get('episode_length', '---')} pas` |
| Modèle | `{stored_model_choice}` |
| Scénario | `Mode {stored_scenario_mode}` |
| Vent | `{'Activé' if stored_wind_enabled else 'Désactivé'}` |
                """
            )

    exp_col1, exp_col2, exp_col3 = st.columns(3)

    with exp_col1:
        st.download_button(
            label="Télécharger CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"simulation_appontage_mode{stored_scenario_mode}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp_col2:
        st.download_button(
            label="Télécharger résumé JSON",
            data=json.dumps(summary, indent=2, default=str).encode("utf-8"),
            file_name=f"summary_appontage_mode{stored_scenario_mode}.json",
            mime="application/json",
            use_container_width=True,
        )

    with exp_col3:
        st.download_button(
            label="Télécharger statistiques",
            data=df.describe().to_csv().encode("utf-8"),
            file_name=f"stats_appontage_mode{stored_scenario_mode}.csv",
            mime="text/csv",
            use_container_width=True,
        )
