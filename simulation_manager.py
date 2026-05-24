import time
import pybullet as p


class SimulationManager:
    def __init__(self):
        self.env = None
        self.model = None
        self.obs = None
        self.running = False
        self.paused = False
        self.model_type = "position"

    def load_simulation(self, model_type):
        """
        Charge l'environnement et le modÃ¨le PPO selon le choix :
        - position
        - vitesse
        """

        self.model_type = model_type

        if model_type == "position":
            print("Chargement du modÃ¨le avec correction de position...")

            # Ã€ adapter selon le nom rÃ©el de ton environnement
            from stable_baselines3 import PPO
            from envs.drone_rl_env_2_obs import DroneLandingRLEnv

            self.env = DroneLandingRLEnv(gui=True)
            self.model = PPO.load("models/ppo_drone_meta_quadricopter_mode5_hard_obs.zip")

        elif model_type == "vitesse":
            print("Chargement du modÃ¨le avec correction de vitesse...")

            # Ã€ adapter selon le nom rÃ©el de ton environnement
            from stable_baselines3 import PPO
            from envs.drone_rl_env_2_action import DroneLandingRLEnv

            self.env = DroneLandingRLEnv(gui=True)
            self.model = PPO.load("models/ppo_drone_meta_quadricopter_mode5_hard_action_5modes_2.zip")

        else:
            raise ValueError("Type de modÃ¨le inconnu.")

        self.obs, _ = self.env.reset()
        self.running = True
        self.paused = False

    def step_simulation(self):
        """
        Avance la simulation d'un pas.
        """

        if self.env is None or self.model is None:
            return None

        if not self.running or self.paused:
            return None

        action, _ = self.model.predict(self.obs, deterministic=True)

        self.obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        if done:
            print("Ã‰pisode terminÃ©.")
            self.running = False

        return {
            "reward": reward,
            "done": done,
            "info": info
        }

    def pause_simulation(self):
        self.paused = True
        print("Simulation en pause.")

    def resume_simulation(self):
        self.paused = False
        print("Simulation reprise.")

    def reset_simulation(self):
        if self.env is not None:
            self.obs, _ = self.env.reset()
            self.running = True
            self.paused = False
            print("Simulation rÃ©initialisÃ©e.")

    def stop_simulation(self):
        self.running = False
        self.paused = False

        if self.env is not None:
            self.env.close()
            self.env = None

        self.model = None
        self.obs = None

        print("Simulation arrÃªtÃ©e.")

