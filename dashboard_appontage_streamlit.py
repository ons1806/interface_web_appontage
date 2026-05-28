import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from simulation_runner import SimulationRunner

# ==================================================
# Configuration générale
# ==================================================
st.set_page_config(
    page_title="Dashboard appontage autonome",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# Style CSS
# ==================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .small-note {
        color: #64748b;
        font-size: 0.86rem;
    }
    .status-ok {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
        padding: 0.55rem 0.8rem;
        border-radius: 0.75rem;
        font-weight: 700;
        text-align: center;
    }
    .status-ko {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        padding: 0.55rem 0.8rem;
        border-radius: 0.75rem;
        font-weight: 700;
        text-align: center;
    }
    .status-run {
        background: #dbeafe;
        color: #1d4ed8;
        border: 1px solid #93c5fd;
        padding: 0.55rem 0.8rem;
        border-radius: 0.75rem;
        font-weight: 700;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# Constantes
# ==================================================
PLOTLY_TEMPLATE = "plotly_white"
DRONE_COLOR = "#ef4444"
PLATFORM_COLOR = "#22c55e"
REWARD_COLOR = "#2563eb"
ACTION_COLORS = ["#f59e0b", "#7c3aed", "#0891b2"]
WIND_COLOR = "#0ea5e9"

MODEL_OPTIONS = {
    "PPO–PID : correction de position": "position",
    "PPO–PID : correction de vitesse": "vitesse",
    "PPO–MPC–PID : correction de vitesse": "vitesse_mpc",
    "PPO–SMC–PID : correction de vitesse": "vitesse_smc",
}

MODEL_DESCRIPTIONS = {
    "PPO–PID : correction de position": "L'agent PPO génère une correction de position autour de la cible d'appontage.",
    "PPO–PID : correction de vitesse": "L'agent PPO génère une correction de vitesse transmise au suivi PID.",
    "PPO–MPC–PID : correction de vitesse": "L'action PPO est filtrée ou corrigée par un module MPC avant le suivi PID.",
    "PPO–SMC–PID : correction de vitesse": "L'action PPO est associée à une correction SMC afin d'améliorer la robustesse du suivi.",
}

# ==================================================
# Fonctions utilitaires
# ==================================================
def empty_figure(title: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=height,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def downsample_df(df: pd.DataFrame, max_points: int = 450) -> pd.DataFrame:
    if df is None or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    df_plot = df.iloc[::step].copy()
    if df_plot.index[-1] != df.index[-1]:
        df_plot = pd.concat([df_plot, df.iloc[[-1]]])
    return df_plot


def safe_metric_value(value, unit="", digits=3):
    try:
        if value is None or pd.isna(value):
            return "---"
        return f"{float(value):.{digits}f}{unit}"
    except Exception:
        return "---"


def safe_range(values, pad=0.12):
    values = [float(v) for v in values if not pd.isna(v)]
    if not values:
        return None
    mn, mx = min(values), max(values)
    delta = max(mx - mn, 0.1)
    return [mn - delta * pad, mx + delta * pad]


def build_summary_table(summary: dict, model_label: str, scenario_mode: int, wind_enabled: bool) -> pd.DataFrame:
    rows = [
        ["Modèle", model_label],
        ["Scénario", f"Mode {scenario_mode}"],
        ["Vent", "Activé" if wind_enabled else "Désactivé"],
        ["Succès", "Oui" if summary.get("success", False) else "Non"],
        ["Erreur XY finale", safe_metric_value(summary.get("final_xy_error"), " m", 4)],
        ["z relatif final", safe_metric_value(summary.get("final_z_rel"), " m", 4)],
        ["Reward totale", safe_metric_value(summary.get("total_reward"), "", 2)],
        ["Durée épisode", f"{summary.get('episode_length', '---')} pas"],
        ["Contact stable max", summary.get("max_contact_stable_steps", "---")],
        ["Seed mode", summary.get("mode_seed", "---")],
    ]
    return pd.DataFrame(rows, columns=["Grandeur", "Valeur"])

# ==================================================
# Fonctions de tracé
# ==================================================
def plot_3d_trajectory(df: pd.DataFrame | None) -> go.Figure:
    fig = go.Figure()

    if df is None or len(df) == 0:
        fig.update_layout(
            title=dict(text="Trajectoire 3D drone / plateforme", font=dict(size=13)),
            height=500,
            template=PLOTLY_TEMPLATE,
            scene=dict(
                xaxis_title="Y (m)",
                yaxis_title="X (m)",
                zaxis_title="Z (m)",
                camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.8)),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, t=45, b=0),
        )
        return fig

    df_plot = downsample_df(df, max_points=450)
    has_drone = {"drone_x", "drone_y", "drone_z"}.issubset(df_plot.columns)
    has_platform = {"platform_x", "platform_y", "platform_z"}.issubset(df_plot.columns)

    if has_drone:
        fig.add_trace(
            go.Scatter3d(
                x=df_plot["drone_y"],
                y=df_plot["drone_x"],
                z=df_plot["drone_z"],
                mode="lines",
                name="Drone",
                line=dict(color=DRONE_COLOR, width=5),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[df["drone_y"].iloc[-1]],
                y=[df["drone_x"].iloc[-1]],
                z=[df["drone_z"].iloc[-1]],
                mode="markers",
                name="Drone actuel",
                marker=dict(size=8, color="#1d4ed8", symbol="x"),
            )
        )

    if has_platform:
        fig.add_trace(
            go.Scatter3d(
                x=df_plot["platform_y"],
                y=df_plot["platform_x"],
                z=df_plot["platform_z"],
                mode="lines",
                name="Plateforme",
                line=dict(color=PLATFORM_COLOR, width=5),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[df["platform_y"].iloc[-1]],
                y=[df["platform_x"].iloc[-1]],
                z=[df["platform_z"].iloc[-1]],
                mode="markers",
                name="Plateforme actuelle",
                marker=dict(size=9, color="#7c3aed", symbol="diamond"),
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
                line=dict(color="#f59e0b", width=3, dash="dash"),
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
        title=dict(text="Trajectoire 3D drone / plateforme", font=dict(size=13)),
        scene=dict(
            xaxis=dict(title="Y (m)", backgroundcolor="#f8fafc", gridcolor="#e2e8f0", showbackground=True, range=safe_range(all_x)),
            yaxis=dict(title="X (m)", backgroundcolor="#f1f5f9", gridcolor="#e2e8f0", showbackground=True, range=safe_range(all_y)),
            zaxis=dict(title="Z (m)", backgroundcolor="#eef2ff", gridcolor="#e2e8f0", showbackground=True, range=safe_range(all_z)),
            camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.8), up=dict(x=0, y=0, z=1)),
            aspectmode="cube",
        ),
        height=500,
        margin=dict(l=0, r=0, t=45, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.90)", bordercolor="#e2e8f0", borderwidth=1, font=dict(size=11)),
        template=PLOTLY_TEMPLATE,
    )
    return fig


def plot_error(df: pd.DataFrame | None) -> go.Figure:
    fig = empty_figure("Erreur de position", height=300)
    if df is None or len(df) == 0:
        return fig
    df_plot = downsample_df(df)
    if "xy_error" in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot["step"], y=df_plot["xy_error"], name="Erreur XY", line=dict(color=DRONE_COLOR, width=2.2), fill="tozeroy", fillcolor="rgba(239,68,68,0.08)"))
    if "z_rel" in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot["step"], y=np.abs(df_plot["z_rel"]), name="|z_rel|", line=dict(color=PLATFORM_COLOR, width=2.2), fill="tozeroy", fillcolor="rgba(34,197,94,0.08)"))
    fig.add_hline(y=0.15, line_dash="dot", line_color="#94a3b8", annotation_text="seuil indicatif")
    fig.update_layout(xaxis_title="Pas", yaxis_title="Erreur (m)", legend=dict(font=dict(size=11)))
    return fig


def plot_reward(df: pd.DataFrame | None) -> go.Figure:
    fig = empty_figure("Récompense", height=300)
    if df is None or len(df) == 0:
        return fig
    df_plot = downsample_df(df)
    if "reward" in df_plot.columns:
        fig.add_trace(go.Bar(x=df_plot["step"], y=df_plot["reward"], name="Reward instantanée", marker_color=REWARD_COLOR, opacity=0.40))
    if "total_reward" in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot["step"], y=df_plot["total_reward"], name="Reward cumulée", line=dict(color="#0f172a", width=2.6)))
    fig.update_layout(xaxis_title="Pas", yaxis_title="Reward", barmode="overlay", legend=dict(font=dict(size=11)))
    return fig


def plot_actions(df: pd.DataFrame | None) -> go.Figure:
    fig = empty_figure("Actions générées par l'agent PPO", height=300)
    if df is None or len(df) == 0:
        return fig
    df_plot = downsample_df(df)
    for i, col in enumerate(["action_1", "action_2", "action_3"]):
        if col in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot["step"], y=df_plot[col], name=f"Action {i + 1}", line=dict(color=ACTION_COLORS[i], width=2.2)))
    fig.update_layout(xaxis_title="Pas", yaxis_title="Valeur", legend=dict(font=dict(size=11)))
    return fig


def plot_distance(df: pd.DataFrame | None) -> go.Figure:
    fig = empty_figure("Distance horizontale drone-plateforme", height=300)
    if df is None or len(df) == 0:
        return fig
    df_plot = downsample_df(df)
    if {"drone_x", "drone_y", "platform_x", "platform_y"}.issubset(df_plot.columns):
        distance = np.sqrt((df_plot["drone_x"] - df_plot["platform_x"]) ** 2 + (df_plot["drone_y"] - df_plot["platform_y"]) ** 2)
    elif "xy_error" in df_plot.columns:
        distance = df_plot["xy_error"]
    else:
        return fig
    fig.add_trace(go.Scatter(x=df_plot["step"], y=distance, name="Distance XY", line=dict(color="#f59e0b", width=2.2), fill="tozeroy", fillcolor="rgba(245,158,11,0.08)"))
    fig.add_hline(y=0.15, line_dash="dot", line_color="#94a3b8", annotation_text="seuil indicatif")
    fig.update_layout(xaxis_title="Pas", yaxis_title="Distance (m)")
    return fig


def plot_attitude(df: pd.DataFrame | None) -> go.Figure:
    fig = empty_figure("Angles d'attitude", height=300)
    if df is None or len(df) == 0:
        return fig
    df_plot = downsample_df(df)
    for col in ["roll", "pitch", "yaw"]:
        if col in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot["step"], y=df_plot[col], name=col))
    fig.update_layout(xaxis_title="Pas", yaxis_title="Angle (rad)")
    return fig


def plot_wind(df: pd.DataFrame | None) -> go.Figure:
    fig = empty_figure("Composantes du vent", height=300)
    if df is None or len(df) == 0:
        return fig
    df_plot = downsample_df(df)
    for col in ["wind_x", "wind_y", "wind_z"]:
        if col in df_plot.columns and not df_plot[col].isna().all():
            fig.add_trace(go.Scatter(x=df_plot["step"], y=df_plot[col], name=col))
    fig.update_layout(xaxis_title="Pas", yaxis_title="Vitesse du vent (m/s)")
    return fig

# ==================================================
# En-tête
# ==================================================
st.markdown('<div class="main-title">Dashboard de visualisation des résultats d\'appontage autonome</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Simulation PPO–PyBullet, suivi de trajectoire 3D, analyse temporelle et export des résultats.</div>',
    unsafe_allow_html=True,
)

# ==================================================
# Barre latérale
# ==================================================
with st.sidebar:
    st.markdown("## Panneau de contrôle")

    page = st.radio(
        "Navigation",
        ["Simulation", "Analyse", "Comparaison", "Aide"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Modèle et scénario")

    model_choice = st.selectbox(
        "Architecture de commande",
        list(MODEL_OPTIONS.keys()),
        index=3,
        help="La quatrième version correspond à l'architecture PPO–SMC–PID.",
    )
    model_type = MODEL_OPTIONS[model_choice]

    scenario_mode = st.selectbox(
        "Mode de scénario",
        [1, 2, 3, 4, 5],
        index=4,
        help="Mode utilisé pour l'évaluation déterministe.",
    )

    wind_enabled = st.checkbox("Activer le vent Dryden", value=False)

    st.markdown("---")
    st.markdown("### Paramètres de simulation")

    max_steps = st.slider("Nombre maximal de pas", 100, 3000, 1000, 100)
    frame_interval = st.slider("Intervalle de mise à jour", 10, 100, 30, 10)
    run_button = st.button("Lancer la simulation", type="primary", use_container_width=True)

    st.markdown("---")
    with st.expander("Description du modèle", expanded=False):
        st.write(MODEL_DESCRIPTIONS[model_choice])

# ==================================================
# Métriques globales
# ==================================================
metric_cols = st.columns(5)
metric_success = metric_cols[0].empty()
metric_xy = metric_cols[1].empty()
metric_z = metric_cols[2].empty()
metric_reward = metric_cols[3].empty()
metric_duration = metric_cols[4].empty()

metric_success.metric("Succès", "---")
metric_xy.metric("Erreur XY", "---")
metric_z.metric("z relatif", "---")
metric_reward.metric("Reward totale", "---")
metric_duration.metric("Durée", "---")

# ==================================================
# Page Simulation
# ==================================================
if page == "Simulation":
    col_cfg, col_view, col_traj = st.columns([0.85, 1.45, 1.55])

    with col_cfg:
        with st.container(border=True):
            st.subheader("Configuration")
            st.dataframe(
                pd.DataFrame(
                    [
                        ["Architecture", model_choice],
                        ["Type interne", model_type],
                        ["Scénario", f"Mode {scenario_mode}"],
                        ["Vent", "Activé" if wind_enabled else "Désactivé"],
                        ["Pas maximal", max_steps],
                        ["Intervalle", f"{frame_interval} pas"],
                    ],
                    columns=["Paramètre", "Valeur"],
                ),
                hide_index=True,
                use_container_width=True,
            )
            status_placeholder = st.empty()
            status_placeholder.markdown('<div class="status-run">En attente</div>', unsafe_allow_html=True)
            progress_bar = st.progress(0)

    with col_view:
        with st.container(border=True):
            st.subheader("Visualisation PyBullet")
            image_placeholder = st.empty()
            image_placeholder.info("Cliquez sur **Lancer la simulation** pour afficher la vue PyBullet.")
            st.markdown('<div class="small-note">La caméra est générée à partir de PyBullet en mode DIRECT.</div>', unsafe_allow_html=True)

    with col_traj:
        with st.container(border=True):
            st.subheader("Trajectoire 3D")
            trajectory_placeholder = st.empty()
            trajectory_placeholder.plotly_chart(plot_3d_trajectory(None), use_container_width=True)

    st.markdown("### Courbes temporelles")
    tab_error, tab_reward, tab_actions, tab_distance, tab_attitude, tab_wind = st.tabs(
        ["Erreur", "Récompense", "Actions PPO", "Distance XY", "Attitude", "Vent"]
    )

    with tab_error:
        error_placeholder = st.empty()
        error_placeholder.plotly_chart(plot_error(None), use_container_width=True)
    with tab_reward:
        reward_placeholder = st.empty()
        reward_placeholder.plotly_chart(plot_reward(None), use_container_width=True)
    with tab_actions:
        actions_placeholder = st.empty()
        actions_placeholder.plotly_chart(plot_actions(None), use_container_width=True)
    with tab_distance:
        distance_placeholder = st.empty()
        distance_placeholder.plotly_chart(plot_distance(None), use_container_width=True)
    with tab_attitude:
        attitude_placeholder = st.empty()
        attitude_placeholder.plotly_chart(plot_attitude(None), use_container_width=True)
    with tab_wind:
        wind_placeholder = st.empty()
        wind_placeholder.plotly_chart(plot_wind(None), use_container_width=True)

    if run_button:
        live_data = []
        status_placeholder.markdown('<div class="status-run">Initialisation du modèle</div>', unsafe_allow_html=True)

        try:
            runner = SimulationRunner()
            runner.load_env_and_model(
                model_type=model_type,
                scenario_mode=scenario_mode,
                wind_enabled=wind_enabled,
            )
        except Exception as exc:
            st.error(f"Erreur lors du chargement du modèle ou de l'environnement : {exc}")
            st.stop()

        def update_metrics(row: dict, step: int):
            xy = row.get("xy_error", np.nan)
            z_rel = row.get("z_rel", np.nan)
            total_reward = row.get("total_reward", np.nan)
            success = bool(row.get("success", False))

            metric_success.metric("Succès", "Oui" if success else "Non")
            metric_xy.metric("Erreur XY", safe_metric_value(xy, " m", 3))
            metric_z.metric("z relatif", safe_metric_value(z_rel, " m", 3))
            metric_reward.metric("Reward totale", safe_metric_value(total_reward, "", 2))
            metric_duration.metric("Pas", int(step))

        def on_frame(frame, step):
            image_placeholder.image(frame, caption=f"Vue PyBullet — pas {step}", use_container_width=True)
            if live_data:
                trajectory_placeholder.plotly_chart(plot_3d_trajectory(pd.DataFrame(live_data)), use_container_width=True)

        def on_step(row, step):
            live_data.append(row)
            progress_bar.progress(min(1.0, step / max_steps))
            update_metrics(row, step)

            if step % frame_interval != 0:
                return

            df_live = pd.DataFrame(live_data)
            error_placeholder.plotly_chart(plot_error(df_live), use_container_width=True)
            reward_placeholder.plotly_chart(plot_reward(df_live), use_container_width=True)
            actions_placeholder.plotly_chart(plot_actions(df_live), use_container_width=True)
            distance_placeholder.plotly_chart(plot_distance(df_live), use_container_width=True)
            attitude_placeholder.plotly_chart(plot_attitude(df_live), use_container_width=True)
            wind_placeholder.plotly_chart(plot_wind(df_live), use_container_width=True)
            status_placeholder.markdown('<div class="status-run">Simulation en cours</div>', unsafe_allow_html=True)

        try:
            result = runner.run_episode(
                max_steps=max_steps,
                capture_frames=True,
                frame_interval=frame_interval,
                on_frame=on_frame,
                on_step=on_step,
            )
        except Exception as exc:
            st.error(f"Erreur pendant la simulation : {exc}")
            st.stop()

        if len(result) == 3:
            df, summary, frames = result
        else:
            df, summary = result
            frames = []

        st.session_state["df"] = df
        st.session_state["summary"] = summary
        st.session_state["model_choice"] = model_choice
        st.session_state["model_type"] = model_type
        st.session_state["scenario_mode"] = scenario_mode
        st.session_state["wind_enabled"] = wind_enabled

        progress_bar.progress(1.0)
        ok = bool(summary.get("success", False))
        status_placeholder.markdown(
            '<div class="status-ok">Simulation terminée : appontage réussi</div>' if ok else '<div class="status-ko">Simulation terminée : appontage non validé</div>',
            unsafe_allow_html=True,
        )

        metric_success.metric("Succès", "Oui" if ok else "Non")
        metric_xy.metric("Erreur XY finale", safe_metric_value(summary.get("final_xy_error"), " m", 3))
        metric_z.metric("z relatif final", safe_metric_value(summary.get("final_z_rel"), " m", 3))
        metric_reward.metric("Reward totale", safe_metric_value(summary.get("total_reward"), "", 2))
        metric_duration.metric("Durée", f"{summary.get('episode_length', '---')} pas")

        trajectory_placeholder.plotly_chart(plot_3d_trajectory(df), use_container_width=True)
        error_placeholder.plotly_chart(plot_error(df), use_container_width=True)
        reward_placeholder.plotly_chart(plot_reward(df), use_container_width=True)
        actions_placeholder.plotly_chart(plot_actions(df), use_container_width=True)
        distance_placeholder.plotly_chart(plot_distance(df), use_container_width=True)
        attitude_placeholder.plotly_chart(plot_attitude(df), use_container_width=True)
        wind_placeholder.plotly_chart(plot_wind(df), use_container_width=True)

# ==================================================
# Page Analyse
# ==================================================
elif page == "Analyse":
    st.subheader("Analyse de la dernière simulation")

    if "df" not in st.session_state:
        st.warning("Aucune simulation n'a encore été lancée dans cette session.")
    else:
        df = st.session_state["df"]
        summary = st.session_state["summary"]
        stored_model_choice = st.session_state.get("model_choice", model_choice)
        stored_scenario_mode = st.session_state.get("scenario_mode", scenario_mode)
        stored_wind_enabled = st.session_state.get("wind_enabled", wind_enabled)

        col_a, col_b = st.columns([1, 1.2])
        with col_a:
            st.markdown("### Résumé de l'épisode")
            st.dataframe(build_summary_table(summary, stored_model_choice, stored_scenario_mode, stored_wind_enabled), hide_index=True, use_container_width=True)
        with col_b:
            st.markdown("### Trajectoire finale")
            st.plotly_chart(plot_3d_trajectory(df), use_container_width=True)

        st.markdown("### Courbes principales")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_error(df), use_container_width=True)
            st.plotly_chart(plot_actions(df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_reward(df), use_container_width=True)
            st.plotly_chart(plot_distance(df), use_container_width=True)

        st.markdown("### Données brutes")
        st.dataframe(df, use_container_width=True, height=320)

        st.markdown("### Export")
        exp1, exp2, exp3 = st.columns(3)
        with exp1:
            st.download_button(
                "Télécharger CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"simulation_{st.session_state.get('model_type', model_type)}_mode{stored_scenario_mode}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp2:
            st.download_button(
                "Télécharger résumé JSON",
                data=json.dumps(summary, indent=2, default=str).encode("utf-8"),
                file_name=f"resume_{st.session_state.get('model_type', model_type)}_mode{stored_scenario_mode}.json",
                mime="application/json",
                use_container_width=True,
            )
        with exp3:
            st.download_button(
                "Télécharger statistiques",
                data=df.describe().to_csv().encode("utf-8"),
                file_name=f"stats_{st.session_state.get('model_type', model_type)}_mode{stored_scenario_mode}.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ==================================================
# Page Comparaison
# ==================================================
elif page == "Comparaison":
    st.subheader("Comparaison des architectures")
    st.markdown(
        "Importez un ou plusieurs fichiers CSV issus des simulations pour comparer les architectures "
        "PPO–PID, PPO–MPC–PID et PPO–SMC–PID."
    )

    uploaded_files = st.file_uploader(
        "Importer des fichiers CSV de résultats",
        type=["csv"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        summaries = []
        all_data = []

        for file in uploaded_files:
            temp_df = pd.read_csv(file)
            label = Path(file.name).stem
            temp_df["source"] = label
            all_data.append(temp_df)

            final = temp_df.iloc[-1]
            summaries.append(
                {
                    "Fichier": label,
                    "Durée": len(temp_df),
                    "Erreur XY finale": final.get("xy_error", np.nan),
                    "z relatif final": final.get("z_rel", np.nan),
                    "Reward totale": final.get("total_reward", np.nan),
                    "Succès": bool(final.get("success", False)),
                    "Contact stable max": temp_df.get("contact_stable_steps", pd.Series([0])).max(),
                }
            )

        df_summary = pd.DataFrame(summaries)
        st.markdown("### Tableau comparatif")
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        metric_to_plot = st.selectbox(
            "Indicateur à comparer",
            ["Erreur XY finale", "z relatif final", "Reward totale", "Durée", "Contact stable max"],
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df_summary["Fichier"],
                y=df_summary[metric_to_plot],
                name=metric_to_plot,
            )
        )
        fig.update_layout(
            title=f"Comparaison — {metric_to_plot}",
            xaxis_title="Simulation",
            yaxis_title=metric_to_plot,
            height=380,
            template=PLOTLY_TEMPLATE,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        combined = pd.concat(all_data, ignore_index=True)
        variable = st.selectbox(
            "Courbe temporelle à superposer",
            [col for col in ["xy_error", "z_rel", "total_reward", "reward", "action_1", "action_2", "action_3"] if col in combined.columns],
        )
        fig2 = go.Figure()
        for source, group in combined.groupby("source"):
            fig2.add_trace(go.Scatter(x=group["step"], y=group[variable], mode="lines", name=source))
        fig2.update_layout(
            title=f"Évolution temporelle — {variable}",
            xaxis_title="Pas",
            yaxis_title=variable,
            height=380,
            template=PLOTLY_TEMPLATE,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("Lancez plusieurs simulations et téléchargez les CSV, puis importez-les ici pour faire la comparaison.")

# ==================================================
# Page Aide
# ==================================================
elif page == "Aide":
    st.subheader("Guide d'utilisation")

    st.markdown(
        """
        ### Étapes principales
        1. Aller dans l'onglet **Simulation**.
        2. Choisir l'architecture de commande.
        3. Choisir le mode de scénario.
        4. Activer ou désactiver le vent Dryden.
        5. Lancer la simulation.
        6. Analyser les courbes puis exporter les résultats.

        ### Architectures disponibles
        - **PPO–PID : correction de position** : correction directe de la consigne de position.
        - **PPO–PID : correction de vitesse** : correction de vitesse produite par l'agent PPO.
        - **PPO–MPC–PID : correction de vitesse** : ajout d'un module MPC entre PPO et PID.
        - **PPO–SMC–PID : correction de vitesse** : ajout d'un correcteur SMC pour renforcer la robustesse.

        ### Grandeurs affichées
        - **Erreur XY** : distance horizontale entre le drone et la cible d'appontage.
        - **z relatif** : position verticale relative par rapport à la plateforme.
        - **Reward totale** : somme des récompenses obtenues pendant l'épisode.
        - **Actions PPO** : sorties de l'agent d'apprentissage.
        - **Distance XY** : distance horizontale drone-plateforme.
        - **Vent** : composantes du modèle Dryden lorsque disponibles.
        """
    )

    st.code("streamlit run dashboard_appontage_streamlit.py", language="powershell")
