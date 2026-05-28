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

        # Même seed de base que dans le script de test VS Code
        self.base_seed = 12345

    def load_env_and_model(self, model_type, scenario_mode=1, wind_enabled=False):
        """
        Charge l'environnement PyBullet et le modèle PPO.
        """

        self.model_type = model_type
        self.scenario_mode = int(scenario_mode)
        self.wind_enabled = bool(wind_enabled)

        if model_type == "position":
            from envs.drone_rl_env_2_obs import DroneLandingRLEnv

            self.env = DroneLandingRLEnv(
                gui=False,
                episode_len_sec=50,
                normalized_action_input=True,
            )

            self.model = PPO.load(
                "models/ppo_drone_meta_quadricopter_mode5_hard_obs.zip"
            )

        elif model_type == "vitesse":
            from envs.drone_rl_env_2_action import DroneLandingRLEnv

            self.env = DroneLandingRLEnv(
                gui=False,
                episode_len_sec=50,
                normalized_action_input=True,
            )

            # Architecture PPO–PID classique : le filtre MPC est désactivé.
            if hasattr(self.env, "use_mpc_target_filter"):
                self.env.use_mpc_target_filter = False

            self.model = PPO.load(
                "models/ppo_drone_meta_quadricopter_mode5_hard_action_5modes_2.zip"
            )

        elif model_type == "vitesse_mpc":
            from envs.drone_rl_env_2_action import DroneLandingRLEnv

            self.env = DroneLandingRLEnv(
                gui=False,
                episode_len_sec=50,
                normalized_action_input=True,
            )

            # Architecture PPO–MPC–PID : le module MPC de l'environnement est activé.
            if hasattr(self.env, "use_mpc_target_filter"):
                self.env.use_mpc_target_filter = True

            self.model = PPO.load(
                "models/ppo_drone_meta_quadricopter_ppo_mpc_action_5modes_2.zip"
            )
        elif model_type == "vitesse_smc":
            from envs.drone_rl_env_2_action import DroneLandingRLEnv

            self.env = DroneLandingRLEnv(
              gui=False,
              episode_len_sec=50,
              normalized_action_input=True,
            )

             # Activation éventuelle du mode SMC si ton environnement possède ces variables
            if hasattr(self.env, "use_smc_controller"):
              self.env.use_smc_controller = True

            if hasattr(self.env, "use_mpc_target_filter"):
              self.env.use_mpc_target_filter = False

            if hasattr(self.env, "controller_type"):
              self.env.controller_type = "smc"

            if hasattr(self.env, "architecture"):
              self.env.architecture = "ppo_smc_pid"

            self.model = PPO.load(
              "models/ppo_drone_meta_quadricopter_mode5_hard_smc_action_5modes_2.zip"
            )
        else:
            raise ValueError(
                "Type de modèle inconnu : choisir 'position', 'vitesse' ou 'vitesse_mpc'."
            )

        self._apply_deterministic_eval_config()

    def _get_mode_seed(self):
        """
        Reproduit la logique du script VS Code :
        mode_seed = seed + mode
        """
        return int(self.base_seed + self.scenario_mode)

    def _apply_deterministic_eval_config(self):
        """
        Force un scénario déterministe pour Streamlit.
        """

        if self.env is None:
            return

        mode_seed = self._get_mode_seed()

        np.random.seed(mode_seed)

        # Désactiver le choix aléatoire du mode
        if hasattr(self.env, "use_random_meta_mode"):
            self.env.use_random_meta_mode = False

        # Forcer le mode du méta-modèle
        if hasattr(self.env, "meta_model"):
            self.env.meta_model.set_mode(int(self.scenario_mode))

            if hasattr(self.env.meta_model, "rng"):
                self.env.meta_model.rng = np.random.default_rng(mode_seed)

        # Forcer le générateur du vent Dryden
        if hasattr(self.env, "wind_model"):
            if hasattr(self.env.wind_model, "rng"):
                self.env.wind_model.rng = np.random.default_rng(mode_seed + 100_000)

        # Forcer l'activation ou non du vent depuis l'interface
        if hasattr(self.env, "wind_enabled"):
            self.env.wind_enabled = bool(self.wind_enabled)

        # Activer/désactiver explicitement le filtre MPC selon le modèle sélectionné.
        if hasattr(self.env, "use_mpc_target_filter"):
            self.env.use_mpc_target_filter = (self.model_type == "vitesse_mpc")

        # Compatibilité éventuelle avec d'autres noms de variables
        if hasattr(self.env, "scenario_mode"):
            self.env.scenario_mode = int(self.scenario_mode)

        if hasattr(self.env, "mode"):
            self.env.mode = int(self.scenario_mode)

    def _get_drone_position(self):
        """
        Récupère la position absolue du drone dans PyBullet.
        """
        if (
            self.env is None
            or getattr(self.env, "pyb_client", None) is None
            or getattr(self.env, "drone_id", None) is None
        ):
            return np.nan, np.nan, np.nan

        pos, _ = p.getBasePositionAndOrientation(
            self.env.drone_id,
            physicsClientId=self.env.pyb_client,
        )

        return float(pos[0]), float(pos[1]), float(pos[2])

    def _get_platform_position(self):
        """
        Récupère la position absolue de la plateforme.
        """

        if self.env is None:
            return np.nan, np.nan, np.nan

        if getattr(self.env, "platform_position", None) is not None:
            pos = self.env.platform_position
            return float(pos[0]), float(pos[1]), float(pos[2])

        if (
            getattr(self.env, "platform_id", None) is not None
            and getattr(self.env, "pyb_client", None) is not None
        ):
            pos, _ = p.getBasePositionAndOrientation(
                self.env.platform_id,
                physicsClientId=self.env.pyb_client,
            )
            return float(pos[0]), float(pos[1]), float(pos[2])

        return np.nan, np.nan, np.nan

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
        on_step=None,
    ):
        """
        Lance un épisode complet avec le vrai modèle PPO.

        Paramètres :
        - capture_frames : active la capture PyBullet
        - frame_interval : intervalle de capture des images
        - on_frame : fonction appelée à chaque nouvelle image
        - on_step : fonction appelée à chaque pas de simulation pour courbes temps réel
        """

        if self.env is None or self.model is None:
            raise RuntimeError("L'environnement ou le modèle PPO n'est pas chargé.")

        mode_seed = self._get_mode_seed()

        # Reforcer la configuration juste avant reset
        self._apply_deterministic_eval_config()

        obs, reset_info = self.env.reset(seed=mode_seed)

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

            done = bool(terminated or truncated)
            total_reward += float(reward)

            x_rel = float(info.get("x_rel", np.nan))
            y_rel = float(info.get("y_rel", np.nan))
            z_rel = float(info.get("z_rel", np.nan))

            if "xy_error" in info:
                xy_error = float(info["xy_error"])
            elif not np.isnan(x_rel) and not np.isnan(y_rel):
                xy_error = float(np.sqrt(x_rel**2 + y_rel**2))
            else:
                xy_error = np.nan

            if info.get("success", False):
                success = True

            if info.get("contact_stable_steps", 0) >= 5:
                success = True

            action = np.array(action).flatten()

            drone_x, drone_y, drone_z = self._get_drone_position()
            platform_x, platform_y, platform_z = self._get_platform_position()

            row = {
                "step": int(step),
                "t": float(step / getattr(self.env, "control_freq_hz", 24)),
                "reward": float(reward),
                "total_reward": float(total_reward),

                "x_rel": x_rel,
                "y_rel": y_rel,
                "z_rel": z_rel,
                "xy_error": xy_error,

                "drone_x": drone_x,
                "drone_y": drone_y,
                "drone_z": drone_z,

                "platform_x": platform_x,
                "platform_y": platform_y,
                "platform_z": platform_z,

                "vx": float(info.get("vx", np.nan)),
                "vy": float(info.get("vy", np.nan)),
                "vz": float(info.get("vz", np.nan)),

                "roll": float(info.get("roll", np.nan)),
                "pitch": float(info.get("pitch", np.nan)),
                "yaw": float(info.get("yaw", np.nan)),

                "action_1": float(action[0]) if len(action) > 0 else np.nan,
                "action_2": float(action[1]) if len(action) > 1 else np.nan,
                "action_3": float(action[2]) if len(action) > 2 else np.nan,

                "policy_action_dvx": float(
                    info.get("policy_action_dvx", action[0] if len(action) > 0 else np.nan)
                ),
                "policy_action_dvy": float(
                    info.get("policy_action_dvy", action[1] if len(action) > 1 else np.nan)
                ),
                "policy_action_dvz": float(
                    info.get("policy_action_dvz", action[2] if len(action) > 2 else np.nan)
                ),

                "command_dvx": float(info.get("command_dvx", np.nan)),
                "command_dvy": float(info.get("command_dvy", np.nan)),
                "command_dvz": float(info.get("command_dvz", np.nan)),

                "target_vx": float(info.get("target_vx", np.nan)),
                "target_vy": float(info.get("target_vy", np.nan)),
                "target_vz": float(info.get("target_vz", np.nan)),

                "done": done,
                "success": bool(success),

                "meta_mode": info.get(
                    "meta_mode",
                    reset_info.get("meta_mode", np.nan),
                ),
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

                "has_contact": bool(info.get("has_contact", False)),
                "contact_stable_steps": int(info.get("contact_stable_steps", 0)),
            }

            data.append(row)

            # Mise à jour temps réel des courbes Streamlit
            if on_step is not None:
                on_step(row, step)

            # Capture de l'image PyBullet
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

        if len(df) > 0:
            final_xy_error = float(df["xy_error"].iloc[-1])
            final_z_rel = float(df["z_rel"].iloc[-1])
            max_contact_stable_steps = int(df["contact_stable_steps"].max())
        else:
            final_xy_error = np.nan
            final_z_rel = np.nan
            max_contact_stable_steps = 0

        summary = {
            "model_type": self.model_type,
            "use_mpc_target_filter": bool(
                getattr(self.env, "use_mpc_target_filter", False)
            ) if self.env is not None else False,
            "scenario_mode": self.scenario_mode,
            "mode_seed": mode_seed,
            "wind_enabled": self.wind_enabled,
            "total_reward": float(total_reward),
            "episode_length": int(len(df)),
            "success": bool(success),
            "final_xy_error": final_xy_error,
            "final_z_rel": final_z_rel,
            "max_contact_stable_steps": max_contact_stable_steps,
        }

        if capture_frames:
            return df, summary, frames

        return df, summary