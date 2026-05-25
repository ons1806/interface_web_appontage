import numpy as np
import pandas as pd


class SimulationRunner:
    def __init__(self):
        self.model_type = None
        self.scenario_mode = None
        self.wind_enabled = False

    def load_env_and_model(self, model_type, scenario_mode=1, wind_enabled=False):
        """
        Version temporaire pour tester l'interface Streamlit.
        Elle ne charge pas encore PyBullet ni PPO.
        """
        self.model_type = model_type
        self.scenario_mode = scenario_mode
        self.wind_enabled = wind_enabled

    def run_episode(self, max_steps=1000):
        """
        Génère des données fictives pour vérifier que l'interface web fonctionne.
        """

        steps = np.arange(max_steps)

        # Simulation fictive d'une convergence vers la plateforme
        x_rel = 0.6 * np.exp(-steps / 250) * np.cos(steps / 60)
        y_rel = 0.5 * np.exp(-steps / 250) * np.sin(steps / 70)
        z_rel = 1.2 * (1 - steps / max_steps)

        xy_error = np.sqrt(x_rel**2 + y_rel**2)

        reward = -xy_error - 0.1 * np.abs(z_rel)
        total_reward = np.cumsum(reward)

        if self.model_type == "position":
            action_1 = 0.08 * np.sin(steps / 50)
            action_2 = 0.08 * np.cos(steps / 60)
            action_3 = -0.04 * np.ones_like(steps)
        else:
            action_1 = 0.12 * np.sin(steps / 45)
            action_2 = 0.12 * np.cos(steps / 55)
            action_3 = -0.06 * np.ones_like(steps)

        df = pd.DataFrame({
            "step": steps,
            "reward": reward,
            "total_reward": total_reward,
            "x_rel": x_rel,
            "y_rel": y_rel,
            "z_rel": z_rel,
            "xy_error": xy_error,
            "action_1": action_1,
            "action_2": action_2,
            "action_3": action_3,
            "done": False,
            "success": False
        })

        df.loc[df.index[-1], "done"] = True
        df.loc[df.index[-1], "success"] = True

        summary = {
            "total_reward": float(total_reward[-1]),
            "episode_length": len(df),
            "success": True,
            "final_xy_error": float(xy_error[-1]),
            "final_z_rel": float(z_rel[-1])
        }

        return df, summary