import numpy as np
import pandas as pd
import pybullet as p
from stable_baselines3 import PPO


class SimulationRunner:
    def __init__(self):
        self.env = None
        self.model = None
        self.model_type = None
        self.scenario_mode = None
        self.wind_enabled = False
        self.seed = 42

    def load_env_and_model(self, model_type, scenario_mode=1, wind_enabled=False):
        """
        Charge le vrai environnement PyBullet et le vrai modèle PPO.

        Important pour Streamlit Cloud :
        - gui=False
        - PyBullet fonctionne en DIRECT
        - le scénario est forcé pour limiter les différences avec VS Code
        """

        self.model_type = model_type
        self.scenario_mode = int(scenario_mode)
        self.wind_enabled = bool(wind_enabled)

        # ==================================================
        # Chargement de l'environnement et du modèle PPO
        # ==================================================

        if model_type == "position":
            from envs.drone_rl_env_2_obs import DroneLandingRLEnv

            self.env = DroneLandingRLEnv(gui=False)
            self.model = PPO.load(
                "models/ppo_drone_meta_quadricopter_mode5_hard_obs.zip"
            )

        elif model_type == "vitesse":
            from envs.drone_rl_env_2_action import DroneLandingRLEnv

            self.env = DroneLandingRLEnv(gui=False)
            self.model = PPO.load(
                "models/ppo_drone_meta_quadricopter_mode5_hard_action_5modes_2.zip"
            )

        else:
            raise ValueError(
                "Type de modèle inconnu : choisir 'position' ou 'vitesse'."
            )

        # ==================================================
        # Stabilisation du scénario
        # ==================================================

        # Désactiver le choix aléatoire du mode du méta-modèle
        if hasattr(self.env, "use_random_meta_mode"):
            self.env.use_random_meta_mode = False

        # Forcer le mode du méta-modèle choisi dans l'interface
        if hasattr(self.env, "meta_model"):
            self.env.meta_model.set_mode(self.scenario_mode)

        # Compatibilité avec différents noms éventuels
        if hasattr(self.env, "scenario_mode"):
            self.env.scenario_mode = self.scenario_mode

        if hasattr(self.env, "mode"):
            self.env.mode = self.scenario_mode

        # Forcer le mode de mouvement de plateforme si ton environnement l'utilise
        if hasattr(self.env, "platform_motion_mode"):
            self.env.platform_motion_mode = self.scenario_mode

        # Forcer l'activation/désactivation du vent
        if hasattr(self.env, "wind_enabled"):
            self.env.wind_enabled = self.wind_enabled

        # Fixer les seeds côté numpy
        np.random.seed(self.seed)

    def _capture_frame(self, width=640, height=360):
        """
        Capture une image PyBullet en mode DIRECT pour affichage dans Streamlit.
        """

        if self.env is None or getattr(self.env, "pyb_client", None) is None:
            return None

        target = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        if getattr(self.env, "platform_position", None) is not None:
            target = np.array(self.env.platform_position, dtype=np.float32)

        if getattr(self.env, "drone_id", None) is not None:
            drone_pos, _ = p.getBasePositionAndOrientation(
                self.env.drone_id,
                physicsClientId=self.env.pyb_client,
            )
            target = 0.5 * target + 0.5 * np.array(drone_pos, dtype=np.float32)

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target.tolist(),
            distance=2.4,
            yaw=45,
            pitch=-35,
            roll=0,
            upAxisIndex=2,
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width / height,
            nearVal=0.05,
            farVal=20.0,
        )

        _, _, rgba, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.env.pyb_client,
        )

        frame = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)
        return frame[:, :, :3]

    def run_episode(
        self,
        max_steps=1000,
        capture_frames=False,
        frame_interval=20,
        on_frame=None,
    ):
        """
        Lance un épisode complet avec le vrai modèle PPO.

        Retourne :
        - df : données temporelles
        - summary : résumé numérique
        - frames : images PyBullet si capture_frames=True
        """

        if self.env is None or self.model is None:
            raise RuntimeError(
                "L'environnement ou le modèle PPO n'est pas chargé."
            )

        # ==================================================
        # Reset déterministe
        # ==================================================

        np.random.seed(self.seed)

        obs, reset_info = self.env.reset(seed=self.seed)

        # Après reset, on force encore les paramètres car certains environnements
        # peuvent les modifier dans reset()
        if hasattr(self.env, "use_random_meta_mode"):
            self.env.use_random_meta_mode = False

        if hasattr(self.env, "meta_model"):
            self.env.meta_model.set_mode(self.scenario_mode)

        if hasattr(self.env, "wind_enabled"):
            self.env.wind_enabled = self.wind_enabled

        data = []
        frames = []
        total_reward = 0.0
        success = False

        # ==================================================
        # Capture initiale
        # ==================================================

        if capture_frames:
            frame = self._capture_frame()
            if frame is not None:
                frames.append(frame)
                if on_frame is not None:
                    on_frame(frame, 0)

        # ==================================================
        # Boucle de simulation
        # ==================================================

        for step in range(max_steps):
            action, _ = self.model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = self.env.step(action)

            done = bool(terminated or truncated)
            total_reward += float(reward)

            # ==================================================
            # Grandeurs principales
            # ==================================================

            x_rel = float(info.get("x_rel", np.nan))
            y_rel = float(info.get("y_rel", np.nan))
            z_rel = float(info.get("z_rel", np.nan))

            if "xy_error" in info:
                xy_error = float(info["xy_error"])
            else:
                if not np.isnan(x_rel) and not np.isnan(y_rel):
                    xy_error = float(np.sqrt(x_rel**2 + y_rel**2))
                else:
                    xy_error = np.nan

            # Détection du succès
            if info.get("success", False):
                success = True

            if info.get("contact_stable_steps", 0) >= 5:
                success = True

            action = np.array(action).flatten()

            # ==================================================
            # Enregistrement des données
            # ==================================================

            data.append({
                "step": int(step),
                "reward": float(reward),
                "total_reward": float(total_reward),

                "x_rel": x_rel,
                "y_rel": y_rel,
                "z_rel": z_rel,
                "xy_error": xy_error,

                "vx": float(info.get("vx", np.nan)),
                "vy": float(info.get("vy", np.nan)),
                "vz": float(info.get("vz", np.nan)),

                "roll": float(info.get("roll", np.nan)),
                "pitch": float(info.get("pitch", np.nan)),
                "yaw": float(info.get("yaw", np.nan)),

                "action_1": float(action[0]) if len(action) > 0 else np.nan,
                "action_2": float(action[1]) if len(action) > 1 else np.nan,
                "action_3": float(action[2]) if len(action) > 2 else np.nan,

                "done": done,
                "success": bool(success),

                # Informations utiles pour comparer VS Code et Streamlit
                "meta_mode": info.get("meta_mode", reset_info.get("meta_mode", np.nan)),
                "platform_motion_mode": info.get(
                    "platform_motion_mode",
                    reset_info.get("platform_motion_mode", np.nan),
                ),
                "wind_enabled": info.get(
                    "wind_enabled",
                    reset_info.get("wind_enabled", self.wind_enabled),
                ),

                "init_x": info.get("init_x", reset_info.get("init_x", np.nan)),
                "init_y": info.get("init_y", reset_info.get("init_y", np.nan)),
                "init_z": info.get("init_z", reset_info.get("init_z", np.nan)),

                "wind_x": float(info.get("wind_x", np.nan)),
                "wind_y": float(info.get("wind_y", np.nan)),
                "wind_z": float(info.get("wind_z", np.nan)),

                "has_contact": info.get("has_contact", False),
                "contact_stable_steps": info.get("contact_stable_steps", 0),
            })

            # ==================================================
            # Capture image PyBullet
            # ==================================================

            if capture_frames and step % frame_interval == 0:
                frame = self._capture_frame()
                if frame is not None:
                    frames.append(frame)
                    if on_frame is not None:
                        on_frame(frame, step)

            if done:
                break

        # ==================================================
        # Création du DataFrame
        # ==================================================

        df = pd.DataFrame(data)

        # ==================================================
        # Fermeture de l'environnement
        # ==================================================

        if self.env is not None:
            self.env.close()

        # ==================================================
        # Résumé numérique
        # ==================================================

        if len(df) > 0:
            final_xy_error = float(df["xy_error"].iloc[-1])
            final_z_rel = float(df["z_rel"].iloc[-1])
        else:
            final_xy_error = np.nan
            final_z_rel = np.nan

        summary = {
            "model_type": self.model_type,
            "scenario_mode": self.scenario_mode,
            "wind_enabled": self.wind_enabled,
            "total_reward": float(total_reward),
            "episode_length": int(len(df)),
            "success": bool(success),
            "final_xy_error": final_xy_error,
            "final_z_rel": final_z_rel,
        }

        if capture_frames:
            return df, summary, frames

        return df, summary