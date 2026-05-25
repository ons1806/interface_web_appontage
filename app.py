import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    .title-box {
        background: linear-gradient(90deg, #0f172a, #1e3a8a);
        padding: 22px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0px 4px 18px rgba(0,0,0,0.15);
    }

    .title-box h1 {
        margin-bottom: 4px;
        font-size: 32px;
    }

    .title-box p {
        margin: 0;
        color: #dbeafe;
        font-size: 16px;
    }

    .metric-card {
        background-color: white;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0px 2px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #2563eb;
        margin-bottom: 12px;
    }

    .section-card {
        background-color: white;
        padding: 16px;
        border-radius: 14px;
        box-shadow: 0px 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 16px;
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
        <p>Simulation PPO + PyBullet avec visualisation, trajectoire et courbes en temps réel</p>
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

def plot_xy_trajectory(df):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    if len(df) > 0:
        ax.plot(df["x_rel"], df["y_rel"], linewidth=2, label="Trajectoire relative")
        ax.scatter(df["x_rel"].iloc[0], df["y_rel"].iloc[0], marker="o", label="Départ")
        ax.scatter(df["x_rel"].iloc[-1], df["y_rel"].iloc[-1], marker="x", s=80, label="Position actuelle")

    ax.scatter(0, 0, marker="+", s=120, label="Cible")
    ax.set_xlabel("x relatif (m)")
    ax.set_ylabel("y relatif (m)")
    ax.set_title("Trajectoire horizontale relative")
    ax.grid(True)
    ax.axis("equal")
    ax.legend(loc="best")

    return fig


def plot_error_reward(df):
    fig, ax = plt.subplots(figsize=(5.2, 3.5))

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
    fig, ax = plt.subplots(figsize=(5.2, 3.5))

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
    fig, ax = plt.subplots(figsize=(5.2, 3.5))

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
# Layout principal
# ==================================================

left_col, center_col, right_col = st.columns([1.0, 1.8, 1.2])

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Configuration")
    st.write(f"**Modèle :** {model_choice}")
    st.write(f"**Scénario :** Mode {scenario_mode}")
    st.write(f"**Vent :** {'Activé' if wind_enabled else 'Désactivé'}")
    st.write(f"**Pas max :** {max_steps}")
    st.markdown('</div>', unsafe_allow_html=True)

    status_box = st.empty()
    progress_bar = st.progress(0)

with center_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Visualisation PyBullet")
    image_placeholder = st.empty()
    st.markdown('<p class="small-text">Vue caméra générée par PyBullet en mode DIRECT.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Trajectoire en temps réel")
    trajectory_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Indicateurs")
    metric_success = st.empty()
    metric_xy = st.empty()
    metric_z = st.empty()
    metric_reward = st.empty()
    metric_step = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Erreur de position")
    error_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)


curve_col1, curve_col2 = st.columns(2)

with curve_col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Reward")
    reward_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with curve_col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Actions PPO")
    actions_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)


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

        trajectory_placeholder.pyplot(plot_xy_trajectory(df_live), clear_figure=True)
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

    # Affichage final
    trajectory_placeholder.pyplot(plot_xy_trajectory(df), clear_figure=True)
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