import numpy as np


class LocalLandingMPC:
    """MPC local rapide pour filtrer la cible PPO avant le PID.

    Cette version évite une optimisation numérique coûteuse à chaque pas.
    Elle applique un prédicteur court horizon avec contraintes de vitesse et
    d'accélération, ce qui donne un comportement de type MPC en temps réel.
    """

    def __init__(
        self,
        horizon=8,
        dt=1.0 / 24.0,
        max_xy_speed=0.35,
        max_descent_speed=0.09,
        max_climb_speed=0.10,
        max_accel=0.35,
    ):
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.max_xy_speed = float(max_xy_speed)
        self.max_descent_speed = float(max_descent_speed)
        self.max_climb_speed = float(max_climb_speed)
        self.max_accel = float(max_accel)
        self.prev_velocity = np.zeros(3, dtype=np.float32)

    def reset(self):
        self.prev_velocity = np.zeros(3, dtype=np.float32)

    def compute_target(self, current_pos, desired_pos, platform_velocity):
        current_pos = np.asarray(current_pos, dtype=np.float32)
        desired_pos = np.asarray(desired_pos, dtype=np.float32)
        platform_velocity = np.asarray(platform_velocity, dtype=np.float32)

        lookahead = self.horizon * self.dt
        predicted_desired = desired_pos + platform_velocity * lookahead

        requested_velocity = (predicted_desired - current_pos) / max(lookahead, 1e-6)
        requested_velocity[:2] = np.clip(
            requested_velocity[:2],
            -self.max_xy_speed,
            self.max_xy_speed,
        )
        requested_velocity[2] = np.clip(
            requested_velocity[2],
            -self.max_descent_speed,
            self.max_climb_speed,
        )

        max_delta = self.max_accel * self.dt
        velocity = self.prev_velocity + np.clip(
            requested_velocity - self.prev_velocity,
            -max_delta,
            max_delta,
        )
        velocity[:2] = np.clip(velocity[:2], -self.max_xy_speed, self.max_xy_speed)
        velocity[2] = np.clip(velocity[2], -self.max_descent_speed, self.max_climb_speed)

        self.prev_velocity = velocity.astype(np.float32)
        return (current_pos + self.prev_velocity * self.dt).astype(np.float32)
