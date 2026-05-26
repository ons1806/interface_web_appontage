import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from simulation_runner import SimulationRunner


# ==================================================
# Configuration générale
# ==================================================

st.set_page_config(
    page_title="Interface Web - Appontage autonome",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# Style CSS
# ==================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        max-width: 1500px;
    }

    .title-box {
        background: linear-gradient(90deg, #0f172a, #1e3a8a);
        padding: 20px 26px;
        border-radius: 18px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0px 8px 24px rgba(15, 23, 42, 0.18);
    }

    .title-box h1 {
        margin-bottom: 6px;
        font-size: 34px;
        font-weight: 800;
    }

    .title-box p {
        margin: 0;
        color: #dbeafe;
        font-size: 16px;
    }

    section[data-testid="stSidebar"] {
        background-color: #f1f5f9;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 14px 16px;
        border-radius: 14px;
        box-shadow: 0px 2px 12px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
    }

    .small-text {
        color: #64748b;
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
    <div class="title-box">
        <h1>Interface Web d'Appontage Autonome</h1>
        <p>Simulation PPO + PyBullet avec visualisation, trajectoire 3D et courbes en temps réel</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# Barre latérale
# ==================================================

st.sidebar.title("Panneau de contrôle")

model_choice = st.sidebar.selectbox(
    "Modèle PPO",
    ["Correction de position", "Correction de vitesse"]
)

scenario_mode = st.sidebar.selectbox(
    "Scénario",
    [1, 2, 3, 4, 5],
    index=4
)

wind_enabled = st.sidebar.checkbox("Activer le vent", value=False)

max_steps = st.sidebar.slider(
    "Nombre maximal de pas",
    min_value=100,
    max_value=3000,
    value=1000,
    step=100
)

frame_interval = st.sidebar.slider(
    "Intervalle d'affichage image",
    min_value=5,
    max_value=100,
    value=20,
    step=5
)

plot_interval = st.sidebar.slider(
    "Intervalle de mise à jour des courbes",
    min_value=5,
    max_value=100,
    value=20,
    step=5
)

run_button = st.sidebar.button("Lancer la simulation", type="primary")


if model_choice == "Correction de position":
    model_type = "position"
else:
    model_type = "vitesse"


# ==================================================
# Fonctions de tracé
# ==================================================

def plot_3d_trajectory(df):
    fig = go.Figure()

    if len(df) == 0:
        fig.update_layout(
            height=520,
            margin=dict(l=0, r=0, t=40, b=0),
            title="Trajectoire 3D en temps réel"
        )
        return fig

    # Trajectoire du drone
    if {"drone_x", "drone_y", "drone_z"}.issubset(df.columns):
        fig.add_trace(go.Scatter3d(
            x=df["drone_y"],
            y=df["drone_x"],
            z=df["drone_z"],
            mode="lines+markers",
            name="Points of UAV",
            line=dict(color="red", width=5),
            marker=dict(size=3, color="red")
        ))

        fig.add_trace(go.Scatter3d(
            x=[df["drone_x"].iloc[-1]],
            y=[df["drone_y"].iloc[-1]],
            z=[df["drone_z"].iloc[-1]],
            mode="markers",
            name="Current UAV",
            marker=dict(size=7, color="blue", symbol="x")
        ))

    # Trajectoire de la plateforme
    if {"platform_x", "platform_y", "platform_z"}.issubset(df.columns):
        fig.add_trace(go.Scatter3d(
            x=df["platform_x"],
            y=df["platform_y"],
            z=df["platform_z"],
            mode="lines+markers",
            name="Points of platform",
            line=dict(color="green", width=5),
            marker=dict(size=3, color="green")
        ))

        fig.add_trace(go.Scatter3d(
            x=[df["platform_x"].iloc[-1]],
            y=[df["platform_y"].iloc[-1]],
            z=[df["platform_z"].iloc[-1]],
            mode="markers",
            name="Current platform",
            marker=dict(size=7, color="purple", symbol="diamond")
        ))

    fig.update_layout(
        title="Trajectoire 3D drone / plateforme",
        scene=dict(
            xaxis_title="X-direction (m)",
            yaxis_title="Y-direction (m)",
            zaxis_title="Z-direction (m)",
            aspectmode="cube",
        ),
        height=520,
        margin=dict(l=0, r=0, t=45, b=0),
        legend=dict(x=0.02, y=0.98),
    )

    return fig


def plot_error_reward(df):
    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    if len(df) > 0:
        ax.plot(df["step"], df["xy_error"], label="Erreur XY")
        ax.plot(df["step"], np.abs(df["z_rel"]), label="|z_rel|")

    ax.set_xlabel("Pas de simulation")
    ax.set_ylabel("Erreur (m)")
    ax.set_title("Erreur de position")
    ax.grid(True)
    ax.legend(loc="best")

    return fig


def plot_reward(df):
    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    if len(df) > 0:
        ax.plot(df["step"], df["reward"], label="Reward")
        ax.plot(df["step"], df["total_reward"], label="Reward cumulée")

    ax.set_xlabel("Pas de simulation")
    ax.set_ylabel("Récompense")
    ax.set_title("Évolution de la récompense")
    ax.grid(True)
    ax.legend(loc="best")

    return fig


def plot_actions(df):
    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    if len(df) > 0:
        ax.plot(df["step"], df["action_1"], label="action 1")
        ax.plot(df["step"], df["action_2"], label="action 2")
        ax.plot(df["step"], df["action_3"], label="action 3")

    ax.set_xlabel("Pas de simulation")
    ax.set_ylabel("Action PPO")
    ax.set_title("Actions générées par PPO")
    ax.grid(True)
    ax.legend(loc="best")

    return fig


# ==================================================
# Métriques principales
# ==================================================

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

metric_success = metric_col1.empty()
metric_xy = metric_col2.empty()
metric_z = metric_col3.empty()
metric_reward = metric_col4.empty()
metric_step = metric_col5.empty()

metric_success.metric("Succès", "---")
metric_xy.metric("Erreur XY", "---")
metric_z.metric("z relatif", "---")
metric_reward.metric("Reward totale", "---")
metric_step.metric("Step", "---")


# ==================================================
# Layout principal
# ==================================================

config_col, sim_col, traj_col = st.columns([0.85, 1.45, 1.45])

with config_col:
    with st.container(border=True):
        st.subheader("Configuration")
        st.write(f"**Modèle :** {model_choice}")
        st.write(f"**Scénario :** Mode {scenario_mode}")
        st.write(f"**Vent :** {'Activé' if wind_enabled else 'Désactivé'}")
        st.write(f"**Pas max :** {max_steps}")

        status_box = st.empty()
        progress_bar = st.progress(0)

with sim_col:
    with st.container(border=True):
        st.subheader("Visualisation PyBullet")
        image_placeholder = st.empty()
        st.caption("Vue caméra générée par PyBullet en mode DIRECT.")

with traj_col:
    with st.container(border=True):
        st.subheader("Trajectoire 3D en temps réel")
        trajectory3d_placeholder = st.empty()


curve_col1, curve_col2, curve_col3 = st.columns(3)

with curve_col1:
    with st.container(border=True):
        st.subheader("Erreur de position")
        error_placeholder = st.empty()

with curve_col2:
    with st.container(border=True):
        st.subheader("Reward")
        reward_placeholder = st.empty()

with curve_col3:
    with st.container(border=True):
        st.subheader("Actions PPO")
        actions_placeholder = st.empty()


# ==================================================
# Simulation
# ==================================================

if run_button:
    status_box.info("Initialisation de la simulation...")

    runner = SimulationRunner()
    runner.load_env_and_model(
        model_type=model_type,
        scenario_mode=scenario_mode,
        wind_enabled=wind_enabled
    )

    live_data = []

    def update_frame(frame, step):
        image_placeholder.image(
            frame,
            caption=f"Visualisation PyBullet - step {step}",
            use_container_width=True
        )

    def update_step(row, step):
        live_data.append(row)

        if step % plot_interval != 0:
            return

        df_live = pd.DataFrame(live_data)

        progress_bar.progress(min(1.0, step / max_steps))

        current_xy = row.get("xy_error", np.nan)
        current_z = row.get("z_rel", np.nan)
        current_reward = row.get("total_reward", np.nan)
        current_success = row.get("success", False)

        metric_success.metric("Succès", "Oui" if current_success else "Non")
        metric_xy.metric("Erreur XY", f"{current_xy:.3f} m")
        metric_z.metric("z relatif", f"{current_z:.3f} m")
        metric_reward.metric("Reward totale", f"{current_reward:.2f}")
        metric_step.metric("Step", int(step))

        trajectory3d_placeholder.plotly_chart(
            plot_3d_trajectory(df_live),
            use_container_width=True,
            key=f"traj3d_{step}"
        )

        error_placeholder.pyplot(plot_error_reward(df_live), clear_figure=True)
        reward_placeholder.pyplot(plot_reward(df_live), clear_figure=True)
        actions_placeholder.pyplot(plot_actions(df_live), clear_figure=True)

    status_box.info("Simulation en cours...")

    result = runner.run_episode(
        max_steps=max_steps,
        capture_frames=True,
        frame_interval=frame_interval,
        on_frame=update_frame,
        on_step=update_step
    )

    if len(result) == 3:
        df, summary, frames = result
    else:
        df, summary = result
        frames = []

    st.session_state["df"] = df
    st.session_state["summary"] = summary

    progress_bar.progress(1.0)
    status_box.success("Simulation terminée.")

    trajectory3d_placeholder.plotly_chart(
        plot_3d_trajectory(df),
        use_container_width=True,
        key="traj3d_final"
    )

    error_placeholder.pyplot(plot_error_reward(df), clear_figure=True)
    reward_placeholder.pyplot(plot_reward(df), clear_figure=True)
    actions_placeholder.pyplot(plot_actions(df), clear_figure=True)

    metric_success.metric("Succès", "Oui" if summary["success"] else "Non")
    metric_xy.metric("Erreur XY finale", f"{summary['final_xy_error']:.3f} m")
    metric_z.metric("z relatif final", f"{summary['final_z_rel']:.3f} m")
    metric_reward.metric("Reward totale", f"{summary['total_reward']:.2f}")
    metric_step.metric("Durée épisode", summary["episode_length"])


# ==================================================
# Résultats finaux + téléchargement
# ==================================================

if "df" in st.session_state:
    df = st.session_state["df"]
    summary = st.session_state["summary"]

    st.markdown("---")
    st.subheader("Résultats enregistrés")

    st.dataframe(df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Télécharger les résultats CSV",
        data=csv_data,
        file_name="resultats_simulation_appontage.csv",
        mime="text/csv"
    )