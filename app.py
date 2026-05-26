import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

from simulation_runner import SimulationRunner


# ==================================================
# Configuration générale
# ==================================================

st.set_page_config(
    page_title="Appontage autonome - PPO et PyBullet",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# Style CSS académique
# ==================================================

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: "Arial", sans-serif;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1600px;
    }

    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        padding: 22px 30px;
        border-radius: 14px;
        color: white;
        margin-bottom: 18px;
        border: 1px solid rgba(148, 163, 184, 0.3);
        box-shadow: 0 6px 24px rgba(15, 23, 42, 0.22);
    }

    .hero-header h1 {
        margin: 0 0 6px 0;
        font-size: 30px;
        font-weight: 700;
    }

    .hero-header p {
        margin: 0;
        color: #dbeafe;
        font-size: 15px;
    }

    section[data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 700;
        font-size: 15px;
        padding: 10px;
        background: #1d4ed8;
        border: none;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 14px 16px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
        border: 1px solid #e2e8f0;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        border-color: #e2e8f0 !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }

    .status-idle {
        background: #f1f5f9;
        color: #475569;
    }

    .status-running {
        background: #dbeafe;
        color: #1d4ed8;
    }

    .status-success {
        background: #dcfce7;
        color: #15803d;
    }

    .section-title {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b;
        margin-bottom: 8px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 6px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f1f5f9;
        padding: 4px;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 6px 16px;
        font-weight: 600;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# En-tête
# ==================================================

st.markdown(
    """
    <div class="hero-header">
        <h1>Interface web d'appontage autonome</h1>
        <p>Simulation PPO - PyBullet avec visualisation, trajectoire 3D et courbes en temps réel</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# Barre latérale
# ==================================================

with st.sidebar:
    st.markdown("## Panneau de contrôle")
    st.markdown("---")

    st.markdown(
        '<div class="section-title">Modèle et scénario</div>',
        unsafe_allow_html=True
    )

    model_choice = st.selectbox(
        "Modèle PPO",
        [
            "PPO-PID : correction de position",
            "PPO-PID : correction de vitesse",
            "PPO-MPC-PID : correction de vitesse",
        ],
        help="Sélection du modèle de commande à évaluer."
    )

    scenario_mode = st.selectbox(
        "Scénario",
        [1, 2, 3, 4, 5],
        index=4,
        help="Choix du mode de scénario d'évaluation."
    )

    st.markdown("---")
    st.markdown(
        '<div class="section-title">Paramètres environnement</div>',
        unsafe_allow_html=True
    )

    wind_enabled = st.checkbox("Activer le vent", value=False)

    st.markdown("---")
    st.markdown(
        '<div class="section-title">Paramètres simulation</div>',
        unsafe_allow_html=True
    )

    max_steps = st.slider(
        "Nombre maximal de pas",
        min_value=100,
        max_value=3000,
        value=1000,
        step=100
    )

    frame_interval = st.slider(
        "Intervalle de mise à jour",
        min_value=10,
        max_value=100,
        value=30,
        step=10,
        help="Même intervalle pour la visualisation PyBullet, la trajectoire 3D et les courbes."
    )

    plot_interval = frame_interval

    st.caption(
        f"Visualisation, trajectoire et courbes synchronisées toutes les {frame_interval} étapes."
    )

    st.markdown("---")
    run_button = st.button("Lancer la simulation", type="primary")

    st.markdown("---")

    with st.expander("Aide", expanded=False):
        st.markdown(
            """
**Modèles disponibles :**

- **PPO-PID : correction de position**  
  Le modèle PPO génère une correction de position.

- **PPO-PID : correction de vitesse**  
  Le modèle PPO génère une correction de vitesse.

- **PPO-MPC-PID : correction de vitesse**  
  Le modèle PPO est associé à un module correctif de type MPC avant la commande PID.

**Sorties affichées :**

- visualisation PyBullet ;
- trajectoire 3D drone / plateforme ;
- erreur de position ;
- récompense ;
- actions PPO ;
- export CSV et JSON.
            """
        )


if model_choice == "PPO-PID : correction de position":
    model_type = "position"
elif model_choice == "PPO-PID : correction de vitesse":
    model_type = "vitesse"
else:
    model_type = "vitesse_mpc"


# ==================================================
# Métriques principales
# ==================================================

m1, m2, m3, m4, m5 = st.columns(5)

metric_success = m1.empty()
metric_xy = m2.empty()
metric_z = m3.empty()
metric_reward = m4.empty()
metric_step = m5.empty()

metric_success.metric("Succès", "---")
metric_xy.metric("Erreur XY", "---")
metric_z.metric("z relatif", "---")
metric_reward.metric("Reward totale", "---")
metric_step.metric("Durée épisode", "---")


# ==================================================
# Constantes graphiques
# ==================================================

PLOTLY_TEMPLATE = "plotly_white"
DRONE_COLOR = "#ef4444"
PLATFORM_COLOR = "#22c55e"
REWARD_COLOR = "#2563eb"
ACTION_COLORS = ["#f59e0b", "#7c3aed", "#0891b2"]


# ==================================================
# Fonctions de tracé
# ==================================================

def plot_3d_trajectory(df: pd.DataFrame) -> go.Figure:
    """
    Trace la trajectoire 3D drone / plateforme.
    Les axes X et Y sont inversés pour correspondre à l'affichage demandé.
    """

    fig = go.Figure()

    if df is None or len(df) == 0:
        fig.update_layout(
            height=500,
            template=PLOTLY_TEMPLATE,
            title=dict(text="Trajectoire 3D drone / plateforme", font=dict(size=13)),
            scene=dict(
                xaxis_title="Y-direction (m)",
                yaxis_title="X-direction (m)",
                zaxis_title="Z-direction (m)",
                camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.8)),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, t=45, b=0),
        )
        return fig

    has_drone = {"drone_x", "drone_y", "drone_z"}.issubset(df.columns)
    has_platform = {"platform_x", "platform_y", "platform_z"}.issubset(df.columns)

    # Sous-échantillonnage pour garder le tracé fluide
    max_points = 300
    if len(df) > max_points:
        df_plot = df.iloc[:: max(1, len(df) // max_points)].copy()
        if df_plot.index[-1] != df.index[-1]:
            df_plot = pd.concat([df_plot, df.iloc[[-1]]])
    else:
        df_plot = df

    if has_drone:
        fig.add_trace(go.Scatter3d(
            x=df_plot["drone_y"],
            y=df_plot["drone_x"],
            z=df_plot["drone_z"],
            mode="lines",
            name="Drone",
            line=dict(color=DRONE_COLOR, width=5),
            hovertemplate=(
                "Drone<br>"
                "Y: %{x:.3f} m<br>"
                "X: %{y:.3f} m<br>"
                "Z: %{z:.3f} m<extra></extra>"
            )
        ))

        fig.add_trace(go.Scatter3d(
            x=[df["drone_y"].iloc[-1]],
            y=[df["drone_x"].iloc[-1]],
            z=[df["drone_z"].iloc[-1]],
            mode="markers",
            name="Drone actuel",
            marker=dict(
                size=9,
                color="#1d4ed8",
                symbol="x",
                line=dict(color="white", width=2)
            )
        ))

    if has_platform:
        fig.add_trace(go.Scatter3d(
            x=df_plot["platform_y"],
            y=df_plot["platform_x"],
            z=df_plot["platform_z"],
            mode="lines",
            name="Plateforme",
            line=dict(color=PLATFORM_COLOR, width=5),
            hovertemplate=(
                "Plateforme<br>"
                "Y: %{x:.3f} m<br>"
                "X: %{y:.3f} m<br>"
                "Z: %{z:.3f} m<extra></extra>"
            )
        ))

        fig.add_trace(go.Scatter3d(
            x=[df["platform_y"].iloc[-1]],
            y=[df["platform_x"].iloc[-1]],
            z=[df["platform_z"].iloc[-1]],
            mode="markers",
            name="Plateforme actuelle",
            marker=dict(
                size=10,
                color="#7c3aed",
                symbol="diamond",
                line=dict(color="white", width=2)
            )
        ))

    if has_drone and has_platform:
        fig.add_trace(go.Scatter3d(
            x=[df["drone_y"].iloc[-1], df["platform_y"].iloc[-1]],
            y=[df["drone_x"].iloc[-1], df["platform_x"].iloc[-1]],
            z=[df["drone_z"].iloc[-1], df["platform_z"].iloc[-1]],
            mode="lines",
            name="Erreur instantanée",
            line=dict(color="#f59e0b", width=2, dash="dash"),
        ))

    all_x = []
    all_y = []
    all_z = []

    if has_drone:
        all_x += df["drone_y"].tolist()
        all_y += df["drone_x"].tolist()
        all_z += df["drone_z"].tolist()

    if has_platform:
        all_x += df["platform_y"].tolist()
        all_y += df["platform_x"].tolist()
        all_z += df["platform_z"].tolist()

    def safe_range(vals, pad=0.12):
        vals = [v for v in vals if not pd.isna(v)]
        if not vals:
            return None
        mn, mx = min(vals), max(vals)
        d = max(mx - mn, 0.1)
        return [mn - d * pad, mx + d * pad]

    fig.update_layout(
        title=dict(text="Trajectoire 3D drone / plateforme", font=dict(size=13)),
        scene=dict(
            xaxis=dict(
                title="Y-direction (m)",
                backgroundcolor="#f8fafc",
                gridcolor="#e2e8f0",
                showbackground=True,
                range=safe_range(all_x),
            ),
            yaxis=dict(
                title="X-direction (m)",
                backgroundcolor="#f1f5f9",
                gridcolor="#e2e8f0",
                showbackground=True,
                range=safe_range(all_y),
            ),
            zaxis=dict(
                title="Z-direction (m)",
                backgroundcolor="#eef2ff",
                gridcolor="#e2e8f0",
                showbackground=True,
                range=safe_range(all_z),
            ),
            camera=dict(
                eye=dict(x=-1.6, y=-1.6, z=0.8),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.1),
            ),
            aspectmode="cube",
        ),
        height=500,
        margin=dict(l=0, r=0, t=45, b=0),
        legend=dict(
            orientation="v",
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(size=11)
        ),
        template=PLOTLY_TEMPLATE,
    )

    return fig


def plot_error(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if df is not None and len(df) > 0:
        fig.add_trace(go.Scatter(
            x=df["step"],
            y=df["xy_error"],
            name="Erreur XY",
            line=dict(color=DRONE_COLOR, width=2),
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.08)"
        ))

        fig.add_trace(go.Scatter(
            x=df["step"],
            y=np.abs(df["z_rel"]),
            name="|z_rel|",
            line=dict(color=PLATFORM_COLOR, width=2),
            fill="tozeroy",
            fillcolor="rgba(34,197,94,0.08)"
        ))

        fig.add_hline(
            y=0.15,
            line_dash="dot",
            line_color="#94a3b8",
            annotation_text="seuil XY",
            annotation_position="top right"
        )

    fig.update_layout(
        title=dict(text="Erreur de position", font=dict(size=13)),
        xaxis_title="Pas",
        yaxis_title="Erreur (m)",
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        template=PLOTLY_TEMPLATE,
        legend=dict(font=dict(size=11))
    )

    return fig


def plot_reward(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if df is not None and len(df) > 0:
        fig.add_trace(go.Bar(
            x=df["step"],
            y=df["reward"],
            name="Reward instantanée",
            marker_color=REWARD_COLOR,
            opacity=0.45
        ))

        fig.add_trace(go.Scatter(
            x=df["step"],
            y=df["total_reward"],
            name="Reward cumulée",
            line=dict(color="#0f172a", width=2.5)
        ))

    fig.update_layout(
        title=dict(text="Récompense", font=dict(size=13)),
        xaxis_title="Pas",
        yaxis_title="Reward",
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        template=PLOTLY_TEMPLATE,
        legend=dict(font=dict(size=11)),
        barmode="overlay"
    )

    return fig


def plot_actions(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if df is not None and len(df) > 0:
        for i, col in enumerate(["action_1", "action_2", "action_3"]):
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["step"],
                    y=df[col],
                    name=f"Action {i + 1}",
                    line=dict(color=ACTION_COLORS[i], width=2)
                ))

    fig.update_layout(
        title=dict(text="Actions PPO", font=dict(size=13)),
        xaxis_title="Pas",
        yaxis_title="Valeur",
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        template=PLOTLY_TEMPLATE,
        legend=dict(font=dict(size=11))
    )

    return fig


def plot_xy_distance(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if df is not None and {"drone_x", "drone_y", "platform_x", "platform_y"}.issubset(df.columns):
        dist = np.sqrt(
            (df["drone_x"] - df["platform_x"]) ** 2
            + (df["drone_y"] - df["platform_y"]) ** 2
        )

        fig.add_trace(go.Scatter(
            x=df["step"],
            y=dist,
            line=dict(color="#f59e0b", width=2),
            fill="tozeroy",
            fillcolor="rgba(245,158,11,0.08)",
            name="Distance XY"
        ))

        fig.add_hline(
            y=0.15,
            line_dash="dot",
            line_color="#94a3b8",
            annotation_text="seuil",
            annotation_position="top right"
        )

    fig.update_layout(
        title=dict(text="Distance drone-plateforme", font=dict(size=13)),
        xaxis_title="Pas",
        yaxis_title="Distance (m)",
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        template=PLOTLY_TEMPLATE
    )

    return fig


# ==================================================
# Layout principal
# ==================================================

col_cfg, col_sim, col_traj = st.columns([0.85, 1.45, 1.50])

with col_cfg:
    with st.container(border=True):
        st.subheader("Configuration")

        st.markdown(
            f"""
| Paramètre | Valeur |
|---|---|
| Modèle | {model_choice} |
| Scénario | Mode {scenario_mode} |
| Vent | {'Activé' if wind_enabled else 'Désactivé'} |
| Pas maximal | {max_steps} |
| Intervalle | {frame_interval} pas |
"""
        )

        status_placeholder = st.empty()
        status_placeholder.markdown(
            '<span class="status-badge status-idle">En attente</span>',
            unsafe_allow_html=True
        )

        progress_bar = st.progress(0)

with col_sim:
    with st.container(border=True):
        st.subheader("Visualisation PyBullet")
        image_placeholder = st.empty()
        image_placeholder.info(
            "La simulation n'a pas encore démarré. Lancez-la depuis le panneau de gauche."
        )
        st.caption("Vue caméra générée par PyBullet en mode DIRECT.")

with col_traj:
    with st.container(border=True):
        st.subheader("Trajectoire 3D en temps réel")
        trajectory3d_placeholder = st.empty()
        trajectory3d_placeholder.plotly_chart(
            plot_3d_trajectory(None),
            use_container_width=True
        )


# ==================================================
# Courbes
# ==================================================

st.markdown("### Courbes de simulation")

tab_err, tab_rwd, tab_act, tab_dist = st.tabs([
    "Erreur de position",
    "Récompense",
    "Actions PPO",
    "Distance XY"
])

with tab_err:
    error_placeholder = st.empty()
    error_placeholder.plotly_chart(
        plot_error(None),
        use_container_width=True
    )

with tab_rwd:
    reward_placeholder = st.empty()
    reward_placeholder.plotly_chart(
        plot_reward(None),
        use_container_width=True
    )

with tab_act:
    actions_placeholder = st.empty()
    actions_placeholder.plotly_chart(
        plot_actions(None),
        use_container_width=True
    )

with tab_dist:
    dist_placeholder = st.empty()
    dist_placeholder.plotly_chart(
        plot_xy_distance(None),
        use_container_width=True
    )


# ==================================================
# Simulation
# ==================================================

if run_button:
    status_placeholder.markdown(
        '<span class="status-badge status-running">Initialisation</span>',
        unsafe_allow_html=True
    )

    image_placeholder.empty()

    try:
        runner = SimulationRunner()
        runner.load_env_and_model(
            model_type=model_type,
            scenario_mode=scenario_mode,
            wind_enabled=wind_enabled
        )
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        st.stop()

    live_data = []

    def update_frame(frame, step):
        image_placeholder.image(
            frame,
            caption=f"PyBullet - step {step}",
            use_container_width=True
        )

        # Synchronisation trajectoire 3D avec la visualisation PyBullet
        if len(live_data) > 0:
            df_live = pd.DataFrame(live_data)
            trajectory3d_placeholder.plotly_chart(
                plot_3d_trajectory(df_live),
                use_container_width=True
            )

    def update_step(row, step):
        live_data.append(row)

        progress_bar.progress(min(1.0, step / max_steps))

        if step % plot_interval != 0:
            return

        df_live = pd.DataFrame(live_data)

        xy = row.get("xy_error", float("nan"))
        z = row.get("z_rel", float("nan"))
        total = row.get("total_reward", float("nan"))
        ok = row.get("success", False)

        metric_success.metric("Succès", "Oui" if ok else "Non")
        metric_xy.metric(
            "Erreur XY",
            f"{xy:.3f} m",
            delta="Sous le seuil" if xy < 0.15 else "Hors seuil",
            delta_color="normal" if xy < 0.15 else "inverse"
        )
        metric_z.metric("z relatif", f"{z:.3f} m")
        metric_reward.metric("Reward totale", f"{total:.2f}")
        metric_step.metric("Pas", int(step))

        # Les courbes classiques sont mises à jour au même intervalle que PyBullet
        error_placeholder.plotly_chart(
            plot_error(df_live),
            use_container_width=True
        )

        reward_placeholder.plotly_chart(
            plot_reward(df_live),
            use_container_width=True
        )

        actions_placeholder.plotly_chart(
            plot_actions(df_live),
            use_container_width=True
        )

        dist_placeholder.plotly_chart(
            plot_xy_distance(df_live),
            use_container_width=True
        )

    status_placeholder.markdown(
        '<span class="status-badge status-running">Simulation en cours</span>',
        unsafe_allow_html=True
    )

    try:
        result = runner.run_episode(
            max_steps=max_steps,
            capture_frames=True,
            frame_interval=frame_interval,
            on_frame=update_frame,
            on_step=update_step
        )
    except Exception as e:
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

    status_placeholder.markdown(
        '<span class="status-badge status-success">Simulation terminée</span>',
        unsafe_allow_html=True
    )

    trajectory3d_placeholder.plotly_chart(
        plot_3d_trajectory(df),
        use_container_width=True
    )

    error_placeholder.plotly_chart(
        plot_error(df),
        use_container_width=True
    )

    reward_placeholder.plotly_chart(
        plot_reward(df),
        use_container_width=True
    )

    actions_placeholder.plotly_chart(
        plot_actions(df),
        use_container_width=True
    )

    dist_placeholder.plotly_chart(
        plot_xy_distance(df),
        use_container_width=True
    )

    ok = summary.get("success", False)

    metric_success.metric("Succès", "Oui" if ok else "Non")
    metric_xy.metric("Erreur XY finale", f"{summary.get('final_xy_error', 0):.3f} m")
    metric_z.metric("z relatif final", f"{summary.get('final_z_rel', 0):.3f} m")
    metric_reward.metric("Reward totale", f"{summary.get('total_reward', 0):.2f}")
    metric_step.metric("Durée épisode", summary.get("episode_length", "---"))


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
    st.markdown("### Résultats et export")

    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        with st.container(border=True):
            st.markdown("**Tableau des données brutes**")
            st.dataframe(df, use_container_width=True, height=300)

    with res_col2:
        with st.container(border=True):
            st.markdown("**Résumé de l'épisode**")

            ok = summary.get("success", False)
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
                unsafe_allow_html=True
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
        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Télécharger CSV",
            data=csv_data,
            file_name=f"simulation_appontage_mode{stored_scenario_mode}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with exp_col2:
        json_data = json.dumps(summary, indent=2, default=str).encode("utf-8")

        st.download_button(
            label="Télécharger résumé JSON",
            data=json_data,
            file_name=f"summary_appontage_mode{stored_scenario_mode}.json",
            mime="application/json",
            use_container_width=True
        )

    with exp_col3:
        stats_csv = df.describe().to_csv().encode("utf-8")

        st.download_button(
            label="Télécharger statistiques",
            data=stats_csv,
            file_name=f"stats_appontage_mode{stored_scenario_mode}.csv",
            mime="text/csv",
            use_container_width=True
        )