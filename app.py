import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from simulation_runner import SimulationRunner


# ==================================================
# Configuration générale
# ==================================================
st.set_page_config(
    page_title="Dashboard - Appontage autonome",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# Style CSS : dashboard clair / académique
# ==================================================
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: "Arial", sans-serif;
        background-color: #ffffff;
    }

    .stApp {
        background: #ffffff;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1.2rem;
        max-width: 1650px;
    }

    section[data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #0f172a;
    }

    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
        font-size: 15px;
        padding: 0.65rem 1rem;
        color: white;
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        border: none;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }

    .top-header {
        background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
        border: 1px solid #dbeafe;
        border-left: 6px solid #2563eb;
        padding: 18px 24px;
        border-radius: 16px;
        margin-bottom: 18px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
    }

    .top-header h1 {
        margin: 0 0 6px 0;
        font-size: 30px;
        line-height: 1.2;
        font-weight: 800;
        color: #0f172a;
    }

    .top-header p {
        margin: 0;
        color: #475569;
        font-size: 15px;
    }

    .section-label {
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 7px;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        padding: 15px 16px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
    }

    div[data-testid="stMetricLabel"] {
        color: #475569;
        font-weight: 700;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 800;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        border-color: #e2e8f0 !important;
        background: #ffffff !important;
        box-shadow: 0 2px 12px rgba(15, 23, 42, 0.05);
    }

    .status-badge {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
    }

    .status-idle {
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
    }

    .status-running {
        background: #dbeafe;
        color: #1d4ed8;
        border: 1px solid #93c5fd;
    }

    .status-success {
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        background: #f1f5f9;
        padding: 5px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 7px 18px;
        font-weight: 700;
        color: #334155;
    }

    .stDataFrame {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# En-tête
# ==================================================
st.markdown(
    """
    <div class="top-header">
        <h1>Dashboard - Appontage autonome de drone</h1>
        <p>Visualisation et analyse des performances des architectures PPO avec PyBullet, trajectoire 3D et courbes en temps réel.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# Barre latérale
# ==================================================
with st.sidebar:
    st.markdown("## Panneau de contrôle")
    st.markdown("---")

    st.markdown('<div class="section-label">Modèle et scénario</div>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Architecture PPO",
        [
            "PPO-PID : correction de position",
            "PPO-PID : correction de vitesse",
            "PPO-MPC-PID : correction de vitesse",
            "PPO-SMC-PID : correction de vitesse",
        ],
        help="Choisir l'architecture de commande à évaluer.",
    )

    scenario_mode = st.selectbox(
        "Mode de scénario",
        [1, 2, 3, 4, 5],
        index=4,
        help="Choisir le niveau de difficulté / scénario d'évaluation.",
    )

    st.markdown("---")
    st.markdown('<div class="section-label">Environnement</div>', unsafe_allow_html=True)

    wind_enabled = st.checkbox("Activer le vent Dryden", value=False)

    st.markdown("---")
    st.markdown('<div class="section-label">Simulation</div>', unsafe_allow_html=True)

    max_steps = st.slider(
        "Nombre maximal de pas",
        min_value=100,
        max_value=3000,
        value=1000,
        step=100,
    )

    frame_interval = st.slider(
        "Intervalle de mise à jour",
        min_value=10,
        max_value=100,
        value=30,
        step=10,
        help="Même intervalle pour la visualisation PyBullet, la trajectoire 3D et les courbes.",
    )

    plot_interval = frame_interval
    st.caption(f"Affichage synchronisé toutes les {frame_interval} étapes.")

    st.markdown("---")
    run_button = st.button("Lancer la simulation", type="primary")

    st.markdown("---")
    with st.expander("Aide", expanded=False):
        st.markdown(
            """
**Modèles disponibles**

- **PPO-PID : correction de position** : correction directe de la position.
- **PPO-PID : correction de vitesse** : correction des vitesses cibles.
- **PPO-MPC-PID : correction de vitesse** : filtrage / correction MPC avant la commande PID.
- **PPO-SMC-PID : correction de vitesse** : commande robuste SMC associée au PPO.

**Résultats affichés**

- visualisation PyBullet ;
- trajectoire 3D drone / plateforme ;
- erreurs de position ;
- récompense ;
- actions PPO ;
- export CSV / JSON.
            """
        )

if model_choice == "PPO-PID : correction de position":
    model_type = "position"
elif model_choice == "PPO-PID : correction de vitesse":
    model_type = "vitesse"
elif model_choice == "PPO-SMC-PID : correction de vitesse":
    model_type = "vitesse_smc"
else:
    model_type = "vitesse_mpc"


# ==================================================
# Métriques principales
# ==================================================
metric_cols = st.columns(5)
metric_success = metric_cols[0].empty()
metric_xy = metric_cols[1].empty()
metric_z = metric_cols[2].empty()
metric_reward = metric_cols[3].empty()
metric_step = metric_cols[4].empty()

metric_success.metric("Statut", "---")
metric_xy.metric("Erreur XY", "---")
metric_z.metric("Erreur Z", "---")
metric_reward.metric("Récompense totale", "---")
metric_step.metric("Durée", "---")


# ==================================================
# Constantes graphiques
# ==================================================
PLOTLY_TEMPLATE = "plotly_white"
DRONE_COLOR = "#2563eb"
PLATFORM_COLOR = "#dc2626"
REWARD_COLOR = "#7c3aed"
ACTION_COLORS = ["#2563eb", "#ea580c", "#16a34a"]
GRID_COLOR = "#e2e8f0"


# ==================================================
# Fonctions de tracé
# ==================================================
def _thin_dataframe(df: pd.DataFrame, max_points: int = 350) -> pd.DataFrame:
    if df is None or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    df_plot = df.iloc[::step].copy()
    if df_plot.index[-1] != df.index[-1]:
        df_plot = pd.concat([df_plot, df.iloc[[-1]]])
    return df_plot


def _safe_range(values, pad=0.12):
    clean = [float(v) for v in values if not pd.isna(v)]
    if not clean:
        return None
    mn, mx = min(clean), max(clean)
    d = max(mx - mn, 0.1)
    return [mn - d * pad, mx + d * pad]


def plot_3d_trajectory(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if df is None or len(df) == 0:
        fig.update_layout(
            height=500,
            template=PLOTLY_TEMPLATE,
            title=dict(text="Trajectoire 3D - Drone et plateforme", font=dict(size=14)),
            scene=dict(
                xaxis_title="Y (m)",
                yaxis_title="X (m)",
                zaxis_title="Z (m)",
                camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.85)),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, t=45, b=0),
        )
        return fig

    df_plot = _thin_dataframe(df)
    has_drone = {"drone_x", "drone_y", "drone_z"}.issubset(df.columns)
    has_platform = {"platform_x", "platform_y", "platform_z"}.issubset(df.columns)

    if has_drone:
        fig.add_trace(
            go.Scatter3d(
                x=df_plot["drone_y"],
                y=df_plot["drone_x"],
                z=df_plot["drone_z"],
                mode="lines",
                name="Trajectoire drone",
                line=dict(color=DRONE_COLOR, width=5),
                hovertemplate="Drone<br>Y: %{x:.3f} m<br>X: %{y:.3f} m<br>Z: %{z:.3f} m<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[df["drone_y"].iloc[-1]],
                y=[df["drone_x"].iloc[-1]],
                z=[df["drone_z"].iloc[-1]],
                mode="markers",
                name="Drone actuel",
                marker=dict(size=9, color="#1d4ed8", symbol="x", line=dict(color="white", width=2)),
            )
        )

    if has_platform:
        fig.add_trace(
            go.Scatter3d(
                x=df_plot["platform_y"],
                y=df_plot["platform_x"],
                z=df_plot["platform_z"],
                mode="lines",
                name="Trajectoire plateforme",
                line=dict(color=PLATFORM_COLOR, width=4, dash="dash"),
                hovertemplate="Plateforme<br>Y: %{x:.3f} m<br>X: %{y:.3f} m<br>Z: %{z:.3f} m<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[df["platform_y"].iloc[-1]],
                y=[df["platform_x"].iloc[-1]],
                z=[df["platform_z"].iloc[-1]],
                mode="markers",
                name="Plateforme actuelle",
                marker=dict(size=9, color="#7c3aed", symbol="diamond", line=dict(color="white", width=2)),
            )
        )

    if has_drone and has_platform:
        fig.add_trace(
            go.Scatter3d(
                x=[df["drone_y"].iloc[-1], df["platform_y"].iloc[-1]],
                y=[df["drone_x"].iloc[-1], df["platform_x"].iloc[-1]],
                z=[df["drone_z"].iloc[-1], df["platform_z"].iloc[-1]],
                mode="lines",
                name="Erreur instantanée",
                line=dict(color="#f59e0b", width=2, dash="dot"),
            )
        )

    all_x, all_y, all_z = [], [], []
    if has_drone:
        all_x += df["drone_y"].tolist()
        all_y += df["drone_x"].tolist()
        all_z += df["drone_z"].tolist()
    if has_platform:
        all_x += df["platform_y"].tolist()
        all_y += df["platform_x"].tolist()
        all_z += df["platform_z"].tolist()

    fig.update_layout(
        title=dict(text="Trajectoire 3D - Drone et plateforme", font=dict(size=14)),
        scene=dict(
            xaxis=dict(title="Y (m)", backgroundcolor="#f8fafc", gridcolor=GRID_COLOR, showbackground=True, range=_safe_range(all_x)),
            yaxis=dict(title="X (m)", backgroundcolor="#f8fafc", gridcolor=GRID_COLOR, showbackground=True, range=_safe_range(all_y)),
            zaxis=dict(title="Z (m)", backgroundcolor="#f8fafc", gridcolor=GRID_COLOR, showbackground=True, range=_safe_range(all_z)),
            camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.85), up=dict(x=0, y=0, z=1)),
            aspectmode="cube",
        ),
        height=500,
        margin=dict(l=0, r=0, t=45, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.92)", bordercolor="#e2e8f0", borderwidth=1, font=dict(size=11)),
        template=PLOTLY_TEMPLATE,
    )
    return fig


def plot_error(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df is not None and len(df) > 0:
        fig.add_trace(go.Scatter(x=df["step"], y=df["xy_error"], name="Erreur XY", line=dict(color=DRONE_COLOR, width=2)))
        fig.add_trace(go.Scatter(x=df["step"], y=np.abs(df["z_rel"]), name="|Z relatif|", line=dict(color=PLATFORM_COLOR, width=2)))
        fig.add_hline(y=0.15, line_dash="dot", line_color="#94a3b8", annotation_text="seuil XY", annotation_position="top right")
    fig.update_layout(title="Erreur de position", xaxis_title="Pas de simulation", yaxis_title="Erreur (m)", height=275, margin=dict(l=10, r=10, t=40, b=10), template=PLOTLY_TEMPLATE, legend=dict(font=dict(size=11)))
    return fig


def plot_reward(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df is not None and len(df) > 0:
        fig.add_trace(go.Scatter(x=df["step"], y=df["reward"], name="Reward instantanée", line=dict(color="#94a3b8", width=1.5)))
        fig.add_trace(go.Scatter(x=df["step"], y=df["total_reward"], name="Reward cumulée", line=dict(color=REWARD_COLOR, width=2.5)))
    fig.update_layout(title="Récompense", xaxis_title="Pas de simulation", yaxis_title="Reward", height=275, margin=dict(l=10, r=10, t=40, b=10), template=PLOTLY_TEMPLATE, legend=dict(font=dict(size=11)))
    return fig


def plot_actions(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df is not None and len(df) > 0:
        for i, col in enumerate(["action_1", "action_2", "action_3"]):
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df["step"], y=df[col], name=f"Action {i + 1}", line=dict(color=ACTION_COLORS[i], width=2)))
    fig.update_layout(title="Actions PPO", xaxis_title="Pas de simulation", yaxis_title="Valeur", height=275, margin=dict(l=10, r=10, t=40, b=10), template=PLOTLY_TEMPLATE, legend=dict(font=dict(size=11)))
    return fig


def plot_xy_distance(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df is not None and {"drone_x", "drone_y", "platform_x", "platform_y"}.issubset(df.columns):
        dist = np.sqrt((df["drone_x"] - df["platform_x"]) ** 2 + (df["drone_y"] - df["platform_y"]) ** 2)
        fig.add_trace(go.Scatter(x=df["step"], y=dist, name="Distance XY", line=dict(color="#f59e0b", width=2)))
        fig.add_hline(y=0.15, line_dash="dot", line_color="#94a3b8", annotation_text="seuil", annotation_position="top right")
    fig.update_layout(title="Distance drone-plateforme", xaxis_title="Pas de simulation", yaxis_title="Distance (m)", height=275, margin=dict(l=10, r=10, t=40, b=10), template=PLOTLY_TEMPLATE)
    return fig


# ==================================================
# Layout principal
# ==================================================
col_cfg, col_sim, col_traj = st.columns([0.82, 1.45, 1.55])

with col_cfg:
    with st.container(border=True):
        st.subheader("Configuration")
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

with col_sim:
    with st.container(border=True):
        st.subheader("Simulation PyBullet")
        image_placeholder = st.empty()
        image_placeholder.info("La simulation n'a pas encore démarré.")
        st.caption("Vue caméra générée par PyBullet en mode DIRECT.")

with col_traj:
    with st.container(border=True):
        st.subheader("Trajectoire 3D")
        trajectory3d_placeholder = st.empty()
        trajectory3d_placeholder.plotly_chart(plot_3d_trajectory(None), use_container_width=True)

st.markdown("### Résultats temporels")
plot_cols = st.columns(4)
with plot_cols[0]:
    with st.container(border=True):
        error_placeholder = st.empty()
        error_placeholder.plotly_chart(plot_error(None), use_container_width=True)
with plot_cols[1]:
    with st.container(border=True):
        reward_placeholder = st.empty()
        reward_placeholder.plotly_chart(plot_reward(None), use_container_width=True)
with plot_cols[2]:
    with st.container(border=True):
        actions_placeholder = st.empty()
        actions_placeholder.plotly_chart(plot_actions(None), use_container_width=True)
with plot_cols[3]:
    with st.container(border=True):
        dist_placeholder = st.empty()
        dist_placeholder.plotly_chart(plot_xy_distance(None), use_container_width=True)


# ==================================================
# Simulation
# ==================================================
if run_button:
    status_placeholder.markdown('<span class="status-badge status-running">Initialisation</span>', unsafe_allow_html=True)
    image_placeholder.empty()

    try:
        runner = SimulationRunner()
        runner.load_env_and_model(model_type=model_type, scenario_mode=scenario_mode, wind_enabled=wind_enabled)
    except Exception as exc:
        st.error(f"Erreur lors du chargement du modèle : {exc}")
        st.stop()

    live_data = []

    def update_frame(frame, step):
        image_placeholder.image(frame, caption=f"PyBullet - step {step}", use_container_width=True)
        if live_data:
            df_live = pd.DataFrame(live_data)
            trajectory3d_placeholder.plotly_chart(plot_3d_trajectory(df_live), use_container_width=True)

    def update_step(row, step):
        live_data.append(row)
        progress_bar.progress(min(1.0, step / max_steps))

        if step % plot_interval != 0:
            return

        df_live = pd.DataFrame(live_data)
        xy = row.get("xy_error", float("nan"))
        z = row.get("z_rel", float("nan"))
        total = row.get("total_reward", float("nan"))
        ok = bool(row.get("success", False))

        metric_success.metric("Statut", "Succès" if ok else "En cours")
        metric_xy.metric("Erreur XY", f"{xy:.3f} m")
        metric_z.metric("Erreur Z", f"{abs(z):.3f} m")
        metric_reward.metric("Récompense totale", f"{total:.2f}")
        metric_step.metric("Pas", int(step))

        error_placeholder.plotly_chart(plot_error(df_live), use_container_width=True)
        reward_placeholder.plotly_chart(plot_reward(df_live), use_container_width=True)
        actions_placeholder.plotly_chart(plot_actions(df_live), use_container_width=True)
        dist_placeholder.plotly_chart(plot_xy_distance(df_live), use_container_width=True)

    status_placeholder.markdown('<span class="status-badge status-running">Simulation en cours</span>', unsafe_allow_html=True)

    try:
        result = runner.run_episode(max_steps=max_steps, capture_frames=True, frame_interval=frame_interval, on_frame=update_frame, on_step=update_step)
    except Exception as exc:
        st.error(f"Erreur pendant la simulation : {exc}")
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

    trajectory3d_placeholder.plotly_chart(plot_3d_trajectory(df), use_container_width=True)
    error_placeholder.plotly_chart(plot_error(df), use_container_width=True)
    reward_placeholder.plotly_chart(plot_reward(df), use_container_width=True)
    actions_placeholder.plotly_chart(plot_actions(df), use_container_width=True)
    dist_placeholder.plotly_chart(plot_xy_distance(df), use_container_width=True)

    ok = bool(summary.get("success", False))
    metric_success.metric("Statut", "Succès" if ok else "Échec")
    metric_xy.metric("Erreur XY finale", f"{summary.get('final_xy_error', 0):.3f} m")
    metric_z.metric("Erreur Z finale", f"{abs(summary.get('final_z_rel', 0)):.3f} m")
    metric_reward.metric("Récompense totale", f"{summary.get('total_reward', 0):.2f}")
    metric_step.metric("Durée", summary.get("episode_length", "---"))


# ==================================================
# Résultats et export
# ==================================================
if "df" in st.session_state:
    df = st.session_state["df"]
    summary = st.session_state["summary"]
    stored_model_choice = st.session_state.get("model_choice", model_choice)
    stored_scenario_mode = st.session_state.get("scenario_mode", scenario_mode)
    stored_wind_enabled = st.session_state.get("wind_enabled", wind_enabled)

    st.markdown("---")
    st.markdown("### Synthèse et export")

    res_col1, res_col2 = st.columns([2.1, 1])

    with res_col1:
        with st.container(border=True):
            st.markdown("**Tableau des données enregistrées**")
            st.dataframe(df, use_container_width=True, height=300)

    with res_col2:
        with st.container(border=True):
            st.markdown("**Résumé de l'épisode**")
            ok = bool(summary.get("success", False))
            color = "#15803d" if ok else "#b91c1c"
            background = "#dcfce7" if ok else "#fee2e2"
            st.markdown(
                f"""
<div style="text-align:center; padding:12px; background:{background};
     border-radius:10px; margin-bottom:12px;">
    <div style="font-size:16px; font-weight:700; color:{color}">
        {'Atterrissage réussi' if ok else 'Atterrissage échoué'}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
| Métrique | Valeur |
|---|---|
| Erreur XY finale | `{summary.get('final_xy_error', 0):.4f} m` |
| Erreur Z finale | `{abs(summary.get('final_z_rel', 0)):.4f} m` |
| Récompense totale | `{summary.get('total_reward', 0):.2f}` |
| Durée épisode | `{summary.get('episode_length', '---')} pas` |
| Architecture | `{stored_model_choice}` |
| Scénario | `Mode {stored_scenario_mode}` |
| Vent | `{'Activé' if stored_wind_enabled else 'Désactivé'}` |
"""
            )

    export_cols = st.columns(3)
    with export_cols[0]:
        st.download_button(
            label="Télécharger CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"simulation_appontage_mode{stored_scenario_mode}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_cols[1]:
        st.download_button(
            label="Télécharger résumé JSON",
            data=json.dumps(summary, indent=2, default=str).encode("utf-8"),
            file_name=f"summary_appontage_mode{stored_scenario_mode}.json",
            mime="application/json",
            use_container_width=True,
        )
    with export_cols[2]:
        st.download_button(
            label="Télécharger statistiques",
            data=df.describe().to_csv().encode("utf-8"),
            file_name=f"stats_appontage_mode{stored_scenario_mode}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ==================================================
# Note technique finale
# ==================================================
st.caption(
    "Interface Streamlit connectée à PyBullet en mode DIRECT. Les trajectoires et courbes sont mises à jour selon l'intervalle choisi dans le panneau de contrôle."
)


# Fin du fichier


