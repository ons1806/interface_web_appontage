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

    def load_env_and_model(self, model_type, scenario_mode=1, wind_enabled=False):
        """
        Charge le vrai environnement PyBullet et le vrai modèle PPO.
        Important :
        - gui=False pour Streamlit Cloud
        - PyBullet doit fonctionner en DIRECT, pas en GUI
        """

        self.model_type = model_type
        self.scenario_mode = scenario_mode
        self.wind_enabled = wind_enabled

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
            raise ValueError("Type de modèle inconnu : choisir 'position' ou 'vitesse'.")

        # Paramètres optionnels selon ton environnement
        if hasattr(self.env, "scenario_mode"):
            self.env.scenario_mode = scenario_mode

        if hasattr(self.env, "mode"):
            self.env.mode = scenario_mode

        if hasattr(self.env, "wind_enabled"):
            self.env.wind_enabled = wind_enabled

    def _capture_frame(self, width=640, height=360):
        if self.env is None or self.env.pyb_client is None:
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
        return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    def run_episode(self, max_steps=1000, capture_frames=False, frame_interval=20, on_frame=None):
        """
        Lance un épisode complet avec le vrai modèle PPO.
        Retourne :
        - df : données temporelles
        - summary : résumé numérique
        """

        if self.env is None or self.model is None:
            raise RuntimeError("L'environnement ou le modèle PPO n'est pas chargé.")

        obs, _ = self.env.reset()

        data = []
        frames = []
        total_reward = 0.0
        success = False

        if capture_frames:
            frame = self._capture_frame()
            if frame is not None:
                frames.append(frame)
                if on_frame is not None:
                    on_frame(frame, 0)

        for step in range(max_steps):
            action, _ = self.model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = self.env.step(action)

            done = terminated or truncated
            total_reward += float(reward)

            # Récupération des grandeurs depuis info si disponibles
            x_rel = info.get("x_rel", np.nan)
            y_rel = info.get("y_rel", np.nan)
            z_rel = info.get("z_rel", np.nan)

            # Calcul de sécurité si xy_error n'existe pas dans info
            if "xy_error" in info:
                xy_error = info["xy_error"]
            else:
                if not np.isnan(x_rel) and not np.isnan(y_rel):
                    xy_error = np.sqrt(x_rel**2 + y_rel**2)
                else:
                    xy_error = np.nan

            if info.get("success", False):
                success = True

            action = np.array(action).flatten()

            data.append({
                "step": step,
                "reward": float(reward),
                "total_reward": float(total_reward),
                "x_rel": x_rel,
                "y_rel": y_rel,
                "z_rel": z_rel,
                "xy_error": xy_error,
                "action_1": action[0] if len(action) > 0 else np.nan,
                "action_2": action[1] if len(action) > 1 else np.nan,
                "action_3": action[2] if len(action) > 2 else np.nan,
                "done": done,
                "success": success
            })

            if capture_frames and step % frame_interval == 0:
                frame = self._capture_frame()
                if frame is not None:
                    frames.append(frame)
                    if on_frame is not None:
                        on_frame(frame, step)

            if done:
                break

        df = pd.DataFrame(data)

        if self.env is not None:
            self.env.close()

        summary = {
            "total_reward": total_reward,
            "episode_length": len(df),
            "success": success,
            "final_xy_error": df["xy_error"].iloc[-1] if len(df) > 0 else np.nan,
            "final_z_rel": df["z_rel"].iloc[-1] if len(df) > 0 else np.nan
        }

        if capture_frames:
            return df, summary, frames

        return df, summary
