import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from simulation_runner import SimulationRunner


st.set_page_config(
    page_title="Interface Web - Appontage autonome",
    layout="wide"
)

st.title("Interface Web de Simulation - Appontage Autonome d'un Drone")

st.markdown("""
Cette interface permet de lancer une simulation d'appontage autonome avec deux espaces d'actions :
correction de position et correction de vitesse.
""")

# ==================================================
# Barre latérale de contrôle
# ==================================================

st.sidebar.header("Panneau de contrôle")

model_choice = st.sidebar.selectbox(
    "Choisir le modèle PPO",
    ["Correction de position", "Correction de vitesse"]
)

scenario_mode = st.sidebar.selectbox(
    "Choisir le scénario",
    [1, 2, 3, 4, 5]
)

wind_enabled = st.sidebar.checkbox("Activer le vent", value=False)

render_enabled = st.sidebar.checkbox("Afficher la simulation PyBullet", value=True)

frame_interval = st.sidebar.slider(
    "Frequence d'affichage PyBullet",
    min_value=5,
    max_value=100,
    value=20,
    step=5
)

max_steps = st.sidebar.slider(
    "Nombre maximal de pas",
    min_value=100,
    max_value=3000,
    value=1000,
    step=100
)

run_button = st.sidebar.button("Lancer la simulation")


# ==================================================
# Conversion choix utilisateur
# ==================================================

if model_choice == "Correction de position":
    model_type = "position"
else:
    model_type = "vitesse"


# ==================================================
# Lancement simulation
# ==================================================

if run_button:
    st.info("Simulation en cours...")
    render_placeholder = st.empty()

    runner = SimulationRunner()
    runner.load_env_and_model(
        model_type=model_type,
        scenario_mode=scenario_mode,
        wind_enabled=wind_enabled
    )

    def show_frame(frame, step):
        render_placeholder.image(
            frame,
            caption=f"Vue PyBullet - pas {step}",
            use_container_width=True
        )

    if render_enabled:
        df, summary, frames = runner.run_episode(
            max_steps=max_steps,
            capture_frames=True,
            frame_interval=frame_interval,
            on_frame=show_frame
        )
    else:
        df, summary = runner.run_episode(max_steps=max_steps)
        frames = []

    st.success("Simulation terminée.")

    # Sauvegarde dans la session Streamlit
    st.session_state["df"] = df
    st.session_state["summary"] = summary
    st.session_state["model_type"] = model_type
    st.session_state["frames"] = frames


# ==================================================
# Affichage résultats
# ==================================================

if "df" in st.session_state:
    df = st.session_state["df"]
    summary = st.session_state["summary"]

    st.subheader("Résumé de l'épisode")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Récompense totale", f"{summary['total_reward']:.2f}")
    col2.metric("Durée épisode", summary["episode_length"])
    col3.metric("Succès", "Oui" if summary["success"] else "Non")
    col4.metric("Erreur XY finale", f"{summary['final_xy_error']:.3f} m")
    col5.metric("z relatif final", f"{summary['final_z_rel']:.3f} m")

    if st.session_state.get("frames"):
        st.subheader("Vue PyBullet")
        frames = st.session_state["frames"]
        frame_index = st.slider(
            "Image de la simulation",
            min_value=0,
            max_value=len(frames) - 1,
            value=len(frames) - 1
        )
        st.image(
            frames[frame_index],
            caption=f"Frame {frame_index + 1}/{len(frames)}",
            use_container_width=True
        )

    st.subheader("Données enregistrées")
    st.dataframe(df)

    # ==================================================
    # Courbe erreur horizontale
    # ==================================================

    st.subheader("Évolution de l'erreur horizontale")

    fig1, ax1 = plt.subplots()
    ax1.plot(df["step"], df["xy_error"])
    ax1.set_xlabel("Pas de simulation")
    ax1.set_ylabel("Erreur XY relative (m)")
    ax1.grid(True)
    st.pyplot(fig1)

    # ==================================================
    # Courbe z relatif
    # ==================================================

    st.subheader("Évolution de la hauteur relative")

    fig2, ax2 = plt.subplots()
    ax2.plot(df["step"], df["z_rel"])
    ax2.set_xlabel("Pas de simulation")
    ax2.set_ylabel("z relatif (m)")
    ax2.grid(True)
    st.pyplot(fig2)

    # ==================================================
    # Courbe récompense
    # ==================================================

    st.subheader("Évolution de la récompense")

    fig3, ax3 = plt.subplots()
    ax3.plot(df["step"], df["reward"])
    ax3.set_xlabel("Pas de simulation")
    ax3.set_ylabel("Reward")
    ax3.grid(True)
    st.pyplot(fig3)

    # ==================================================
    # Courbes actions
    # ==================================================

    st.subheader("Actions générées par le modèle PPO")

    fig4, ax4 = plt.subplots()
    ax4.plot(df["step"], df["action_1"], label="action 1")
    ax4.plot(df["step"], df["action_2"], label="action 2")
    ax4.plot(df["step"], df["action_3"], label="action 3")
    ax4.set_xlabel("Pas de simulation")
    ax4.set_ylabel("Action")
    ax4.legend()
    ax4.grid(True)
    st.pyplot(fig4)

    # ==================================================
    # Export CSV
    # ==================================================

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Télécharger les résultats CSV",
        data=csv_data,
        file_name="resultats_simulation_appontage.csv",
        mime="text/csv"
    )
