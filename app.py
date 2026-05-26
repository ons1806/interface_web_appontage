import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from simulation_runner import SimulationRunner

# ==================================================
# Configuration générale
# ==================================================
st.set_page_config(
    page_title="Appontage Autonome — PPO+PyBullet",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# Style CSS amélioré
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 1600px;
}

/* Header */
.hero-header {
    background: linear-gradient(135deg, #050d1a 0%, #0f2355 50%, #1a3a7a 100%);
    padding: 22px 30px;
    border-radius: 16px;
    color: white;
    margin-bottom: 20px;
    border: 1px solid rgba(59,130,246,0.3);
    box-shadow: 0 8px 32px rgba(15,23,42,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: "";
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-header h1 {
    margin: 0 0 6px 0;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.hero-header .subtitle {
    margin: 0;
    color: #93c5fd;
    font-size: 14px;
    font-family: 'JetBrains Mono', monospace;
}
.hero-header .badge {
    display: inline-block;
    background: rgba(59,130,246,0.2);
    border: 1px solid rgba(59,130,246,0.4);
    color: #93c5fd;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    margin-right: 6px;
    font-family: 'JetBrains Mono', monospace;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    border-right: 1px solid #e2e8f0;
}
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 10px;
    font-weight: 700;
    font-size: 15px;
    padding: 12px;
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    border: none;
    box-shadow: 0 4px 12px rgba(37,99,235,0.35);
    transition: all 0.2s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(37,99,235,0.45);
}

/* Métriques */
div[data-testid="stMetric"] {
    background: white;
    padding: 16px 18px;
    border-radius: 14px;
    box-shadow: 0 2px 12px rgba(15,23,42,0.07);
    border: 1px solid #e2e8f0;
    transition: box-shadow 0.2s;
}
div[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 20px rgba(15,23,42,0.12);
}

/* Metric success/failure color */
.metric-success div[data-testid="stMetricValue"] { color: #16a34a !important; }
.metric-fail    div[data-testid="stMetricValue"] { color: #dc2626 !important; }

/* Containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    border-color: #e2e8f0 !important;
    box-shadow: 0 2px 10px rgba(15,23,42,0.05);
}

/* Status boxes */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
}
.status-running  { background:#dbeafe; color:#1d4ed8; }
.status-success  { background:#dcfce7; color:#15803d; }
.status-idle     { background:#f1f5f9; color:#475569; }

/* Section titles */
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

/* Résumé stats */
.summary-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 10px;
}
.summary-item {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 14px;
}
.summary-item .label { font-size: 11px; color: #64748b; margin-bottom: 2px; }
.summary-item .value { font-size: 18px; font-weight: 700; color: #0f172a; font-family: 'JetBrains Mono', monospace; }

/* Tabs */
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

/* Table */
.dataframe { font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# En-tête
# ==================================================
st.markdown("""
<div class="hero-header">
    <h1>🚁 Interface Web d'Appontage Autonome</h1>
    <p class="subtitle">
        <span class="badge">PPO</span>
        <span class="badge">PyBullet</span>
        <span class="badge">Trajectoire 3D</span>
        Simulation temps réel · Courbes dynamiques · Export CSV/JSON
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# Barre latérale
# ==================================================
with st.sidebar:
    st.markdown("## 🎛️ Panneau de contrôle")
    st.markdown("---")

    st.markdown('<div class="section-title">Modèle & Scénario</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Modèle PPO",
        ["Correction de position", "Correction de vitesse"],
        help="Sélectionnez la stratégie de contrôle PPO utilisée."
    )
    scenario_mode = st.selectbox(
        "Scénario",
        [1, 2, 3, 4, 5],
        index=4,
        help="Mode 1–4 : mer calme → agitée. Mode 5 : conditions maximales."
    )

    st.markdown("---")
    st.markdown('<div class="section-title">Paramètres environnement</div>', unsafe_allow_html=True)
    wind_enabled = st.checkbox("💨 Activer le vent", value=False)

    st.markdown("---")
    st.markdown('<div class="section-title">Paramètres simulation</div>', unsafe_allow_html=True)
    max_steps = st.slider("Nombre maximal de pas", 100, 3000, 1000, 100)
    frame_interval = st.slider("Intervalle d'affichage image", 5, 100, 20, 5)
    plot_interval  = st.slider("Intervalle mise à jour courbes", 5, 100, 20, 5)

    st.markdown("---")
    run_button = st.button("▶ Lancer la simulation", type="primary")

    st.markdown("---")
    with st.expander("ℹ️ Aide", expanded=False):
        st.markdown("""
**Modèles disponibles :**
- *Correction de position* : le drone corrige directement sa position XYZ
- *Correction de vitesse* : le drone ajuste sa vitesse (plus réaliste)

**Scénarios :**
- Modes 1–4 : intensité croissante de houle
- Mode 5 : conditions les plus difficiles

**Indicateurs de succès :**
- Erreur XY finale < 0.15 m
- z relatif final < 0.10 m
        """)

model_type = "position" if model_choice == "Correction de position" else "vitesse"

# ==================================================
# Métriques — rangée principale
# ==================================================
m1, m2, m3, m4, m5 = st.columns(5)
metric_success = m1.empty()
metric_xy      = m2.empty()
metric_z       = m3.empty()
metric_reward  = m4.empty()
metric_step    = m5.empty()

metric_success.metric("✅ Succès",        "—")
metric_xy.metric(     "📏 Erreur XY",     "—")
metric_z.metric(      "⬆️ z relatif",     "—")
metric_reward.metric( "🏆 Reward totale", "—")
metric_step.metric(   "🔢 Durée épisode", "—")

# ==================================================
# Fonctions de tracé (toutes Plotly)
# ==================================================
PLOTLY_TEMPLATE = "plotly_white"
DRONE_COLOR    = "#ef4444"
PLATFORM_COLOR = "#22c55e"
REWARD_COLOR   = "#3b82f6"
ACTION_COLORS  = ["#f59e0b", "#8b5cf6", "#06b6d4"]


def plot_3d_trajectory(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df is None or len(df) == 0:
        fig.update_layout(height=480, template=PLOTLY_TEMPLATE,
                          title="Trajectoire 3D en temps réel")
        return fig

    if {"drone_x", "drone_y", "drone_z"}.issubset(df.columns):
        fig.add_trace(go.Scatter3d(
            x=df["drone_y"], y=df["drone_x"], z=df["drone_z"],
            mode="lines+markers", name="UAV",
            line=dict(color=DRONE_COLOR, width=4),
            marker=dict(size=2, color=DRONE_COLOR)
        ))
        fig.add_trace(go.Scatter3d(
            x=[df["drone_y"].iloc[-1]],
            y=[df["drone_x"].iloc[-1]],
            z=[df["drone_z"].iloc[-1]],
            mode="markers", name="UAV actuel",
            marker=dict(size=8, color="#1d4ed8", symbol="x")
        ))

    if {"platform_x", "platform_y", "platform_z"}.issubset(df.columns):
        fig.add_trace(go.Scatter3d(
            x=df["platform_y"], y=df["platform_x"], z=df["platform_z"],
            mode="lines+markers", name="Plateforme",
            line=dict(color=PLATFORM_COLOR, width=4),
            marker=dict(size=2, color=PLATFORM_COLOR)
        ))
        fig.add_trace(go.Scatter3d(
            x=[df["platform_y"].iloc[-1]],
            y=[df["platform_x"].iloc[-1]],
            z=[df["platform_z"].iloc[-1]],
            mode="markers", name="Plateforme actuelle",
            marker=dict(size=9, color="#7c3aed", symbol="diamond")
        ))

    fig.update_layout(
        title=dict(text="Trajectoire 3D — drone & plateforme", font=dict(size=13)),
        scene=dict(
            xaxis_title="Y (m)", yaxis_title="X (m)", zaxis_title="Z (m)",
            aspectmode="cube",
            xaxis=dict(backgroundcolor="#f8fafc", gridcolor="#e2e8f0"),
            yaxis=dict(backgroundcolor="#f8fafc", gridcolor="#e2e8f0"),
            zaxis=dict(backgroundcolor="#f8fafc", gridcolor="#e2e8f0"),
        ),
        height=480,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="v", x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#e2e8f0", borderwidth=1, font=dict(size=11)),
        template=PLOTLY_TEMPLATE,
    )
    return fig


def plot_error(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df is not None and len(df) > 0:
        fig.add_trace(go.Scatter(
            x=df["step"], y=df["xy_error"],
            name="Erreur XY", line=dict(color=DRONE_COLOR, width=2),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.08)"
        ))
        fig.add_trace(go.Scatter(
            x=df["step"], y=np.abs(df["z_rel"]),
            name="|z_rel|", line=dict(color=PLATFORM_COLOR, width=2),
            fill="tozeroy", fillcolor="rgba(34,197,94,0.08)"
        ))
        # Seuil de succès
        fig.add_hline(y=0.15, line_dash="dot", line_color="#94a3b8",
                      annotation_text="seuil XY", annotation_position="top right")
    fig.update_layout(
        title=dict(text="Erreur de position", font=dict(size=13)),
        xaxis_title="Pas", yaxis_title="Erreur (m)",
        height=280, margin=dict(l=10, r=10, t=40, b=10),
        template=PLOTLY_TEMPLATE, legend=dict(font=dict(size=11)),
    )
    return fig


def plot_reward(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df is not None and len(df) > 0:
        fig.add_trace(go.Bar(
            x=df["step"], y=df["reward"],
            name="Reward instantanée",
            marker_color=REWARD_COLOR, opacity=0.5
        ))
        fig.add_trace(go.Scatter(
            x=df["step"], y=df["total_reward"],
            name="Reward cumulée",
            line=dict(color="#0f172a", width=2.5)
        ))
    fig.update_layout(
        title=dict(text="Récompense", font=dict(size=13)),
        xaxis_title="Pas", yaxis_title="Reward",
        height=280, margin=dict(l=10, r=10, t=40, b=10),
        template=PLOTLY_TEMPLATE, legend=dict(font=dict(size=11)),
        barmode="overlay"
    )
    return fig


def plot_actions(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df is not None and len(df) > 0:
        for i, col in enumerate(["action_1", "action_2", "action_3"]):
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["step"], y=df[col],
                    name=f"Action {i+1}",
                    line=dict(color=ACTION_COLORS[i], width=2)
                ))
    fig.update_layout(
        title=dict(text="Actions PPO", font=dict(size=13)),
        xaxis_title="Pas", yaxis_title="Valeur",
        height=280, margin=dict(l=10, r=10, t=40, b=10),
        template=PLOTLY_TEMPLATE, legend=dict(font=dict(size=11)),
    )
    return fig


def plot_xy_distance(df: pd.DataFrame) -> go.Figure:
    """Distance drone–plateforme au fil du temps."""
    fig = go.Figure()
    if df is not None and {"drone_x", "drone_y", "platform_x", "platform_y"}.issubset(df.columns):
        dist = np.sqrt((df["drone_x"] - df["platform_x"])**2 +
                       (df["drone_y"] - df["platform_y"])**2)
        fig.add_trace(go.Scatter(
            x=df["step"], y=dist,
            line=dict(color="#f59e0b", width=2),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
            name="Distance XY"
        ))
        fig.add_hline(y=0.15, line_dash="dot", line_color="#94a3b8",
                      annotation_text="seuil", annotation_position="top right")
    fig.update_layout(
        title=dict(text="Distance drone–plateforme", font=dict(size=13)),
        xaxis_title="Pas", yaxis_title="Distance (m)",
        height=280, margin=dict(l=10, r=10, t=40, b=10),
        template=PLOTLY_TEMPLATE
    )
    return fig


# ==================================================
# Layout principal
# ==================================================
col_cfg, col_sim, col_traj = st.columns([0.85, 1.45, 1.5])

with col_cfg:
    with st.container(border=True):
        st.subheader("⚙️ Configuration")
        st.markdown(f"""
| Paramètre | Valeur |
|---|---|
| Modèle | {model_choice} |
| Scénario | Mode {scenario_mode} |
| Vent | {'✅ Activé' if wind_enabled else '❌ Désactivé'} |
| Pas max | {max_steps} |
""")
        status_placeholder = st.empty()
        status_placeholder.markdown(
            '<span class="status-badge status-idle">⏸ En attente</span>',
            unsafe_allow_html=True
        )
        progress_bar = st.progress(0)

with col_sim:
    with st.container(border=True):
        st.subheader("📷 Visualisation PyBullet")
        image_placeholder = st.empty()
        image_placeholder.info("La simulation n'a pas encore démarré. Lancez-la depuis le panneau de gauche.")
        st.caption("Vue caméra générée par PyBullet en mode DIRECT.")

with col_traj:
    with st.container(border=True):
        st.subheader("🌐 Trajectoire 3D en temps réel")
        trajectory3d_placeholder = st.empty()
        trajectory3d_placeholder.plotly_chart(plot_3d_trajectory(None), use_container_width=True, key="traj3d_init")

# ==================================================
# Courbes — onglets
# ==================================================
st.markdown("### 📊 Courbes de simulation")
tab_err, tab_rwd, tab_act, tab_dist = st.tabs([
    "📏 Erreur de position",
    "🏆 Récompense",
    "🕹️ Actions PPO",
    "📐 Distance XY"
])

with tab_err:
    error_placeholder = st.empty()
    error_placeholder.plotly_chart(plot_error(None), use_container_width=True, key="err_init")

with tab_rwd:
    reward_placeholder = st.empty()
    reward_placeholder.plotly_chart(plot_reward(None), use_container_width=True, key="rwd_init")

with tab_act:
    actions_placeholder = st.empty()
    actions_placeholder.plotly_chart(plot_actions(None), use_container_width=True, key="act_init")

with tab_dist:
    dist_placeholder = st.empty()
    dist_placeholder.plotly_chart(plot_xy_distance(None), use_container_width=True, key="dist_init")

# ==================================================
# Simulation
# ==================================================
if run_button:
    status_placeholder.markdown(
        '<span class="status-badge status-running">⚙️ Initialisation...</span>',
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
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        st.stop()

    live_data = []

    def update_frame(frame, step):
        image_placeholder.image(
            frame,
            caption=f"PyBullet — step {step}",
            use_container_width=True
        )

    def update_step(row, step):
        live_data.append(row)

        # Mise à jour de la barre de progression
        progress_bar.progress(min(1.0, step / max_steps))

        if step % plot_interval != 0:
            return

        df_live = pd.DataFrame(live_data)
        xy    = row.get("xy_error", float("nan"))
        z     = row.get("z_rel", float("nan"))
        total = row.get("total_reward", float("nan"))
        ok    = row.get("success", False)

        metric_success.metric("✅ Succès",        "Oui ✅" if ok else "Non ❌")
        metric_xy.metric(     "📏 Erreur XY",     f"{xy:.3f} m",
                              delta=f"{'↓ bon' if xy < 0.15 else '↑ hors seuil'}",
                              delta_color="normal" if xy < 0.15 else "inverse")
        metric_z.metric(      "⬆️ z relatif",     f"{z:.3f} m")
        metric_reward.metric( "🏆 Reward totale", f"{total:.2f}")
        metric_step.metric(   "🔢 Pas",           int(step))

        trajectory3d_placeholder.plotly_chart(
            plot_3d_trajectory(df_live),
            use_container_width=True, key=f"traj3d_{step}"
        )
        error_placeholder.plotly_chart(plot_error(df_live),   use_container_width=True, key=f"err_{step}")
        reward_placeholder.plotly_chart(plot_reward(df_live), use_container_width=True, key=f"rwd_{step}")
        actions_placeholder.plotly_chart(plot_actions(df_live), use_container_width=True, key=f"act_{step}")
        dist_placeholder.plotly_chart(plot_xy_distance(df_live), use_container_width=True, key=f"dist_{step}")

    status_placeholder.markdown(
        '<span class="status-badge status-running">🔄 Simulation en cours...</span>',
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
        st.error(f"❌ Erreur pendant la simulation : {e}")
        st.stop()

    df      = result[0]
    summary = result[1]

    st.session_state["df"]      = df
    st.session_state["summary"] = summary

    progress_bar.progress(1.0)
    status_placeholder.markdown(
        '<span class="status-badge status-success">✅ Simulation terminée</span>',
        unsafe_allow_html=True
    )

    # Mise à jour finale des graphiques
    trajectory3d_placeholder.plotly_chart(plot_3d_trajectory(df), use_container_width=True, key="traj3d_final")
    error_placeholder.plotly_chart(plot_error(df),     use_container_width=True, key="err_final")
    reward_placeholder.plotly_chart(plot_reward(df),   use_container_width=True, key="rwd_final")
    actions_placeholder.plotly_chart(plot_actions(df), use_container_width=True, key="act_final")
    dist_placeholder.plotly_chart(plot_xy_distance(df), use_container_width=True, key="dist_final")

    # Métriques finales
    ok = summary.get("success", False)
    metric_success.metric("✅ Succès",        "Oui ✅" if ok else "Non ❌")
    metric_xy.metric(     "📏 Erreur XY finale",  f"{summary.get('final_xy_error', 0):.3f} m")
    metric_z.metric(      "⬆️ z relatif final",   f"{summary.get('final_z_rel', 0):.3f} m")
    metric_reward.metric( "🏆 Reward totale",     f"{summary.get('total_reward', 0):.2f}")
    metric_step.metric(   "🔢 Durée épisode",     summary.get("episode_length", "—"))

# ==================================================
# Section Résultats & Export
# ==================================================
if "df" in st.session_state:
    df      = st.session_state["df"]
    summary = st.session_state["summary"]

    st.markdown("---")
    st.markdown("### 💾 Résultats & Export")

    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        with st.container(border=True):
            st.markdown("**Tableau des données brutes**")
            st.dataframe(df, use_container_width=True, height=300)

    with res_col2:
        with st.container(border=True):
            st.markdown("**Résumé de l'épisode**")
            ok = summary.get("success", False)
            color = "#16a34a" if ok else "#dc2626"
            st.markdown(f"""
<div style="text-align:center; padding:12px; background:{'#dcfce7' if ok else '#fee2e2'};
     border-radius:10px; margin-bottom:12px;">
    <div style="font-size:28px">{'✅' if ok else '❌'}</div>
    <div style="font-size:16px; font-weight:700; color:{color}">
        {'Atterrissage réussi' if ok else 'Atterrissage échoué'}
    </div>
</div>
""", unsafe_allow_html=True)
            st.markdown(f"""
| Métrique | Valeur |
|---|---|
| Erreur XY finale | `{summary.get('final_xy_error', 0):.4f} m` |
| z relatif final | `{summary.get('final_z_rel', 0):.4f} m` |
| Reward totale | `{summary.get('total_reward', 0):.2f}` |
| Durée épisode | `{summary.get('episode_length', '—')} pas` |
| Modèle | `{model_choice}` |
| Scénario | `Mode {scenario_mode}` |
| Vent | `{'Activé' if wind_enabled else 'Désactivé'}` |
""")

    # Boutons d'export
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Télécharger CSV",
            data=csv_data,
            file_name=f"simulation_appontage_mode{scenario_mode}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with exp_col2:
        json_data = json.dumps(summary, indent=2, default=str).encode("utf-8")
        st.download_button(
            label="⬇️ Télécharger résumé JSON",
            data=json_data,
            file_name=f"summary_appontage_mode{scenario_mode}.json",
            mime="application/json",
            use_container_width=True
        )
    with exp_col3:
        # Export statistiques descriptives
        stats_csv = df.describe().to_csv().encode("utf-8")
        st.download_button(
            label="⬇️ Statistiques descriptives",
            data=stats_csv,
            file_name=f"stats_appontage_mode{scenario_mode}.csv",
            mime="text/csv",
            use_container_width=True
        )
