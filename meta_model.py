import numpy as np


class MetaModel:
    """
    Meta-model pour générer automatiquement les paramètres
    d'un épisode d'apprentissage DRL.

    Il ne contrôle pas le drone.
    Il configure seulement les conditions de simulation :
    - position initiale du drone
    - orientation initiale
    - mouvement de la plateforme
    - paramètres du vent Dryden
    """

    def __init__(self, mode=1, seed=None):
        """
        mode = 1 : scénario facile
        mode = 2 : scénario moyen
        mode = 3 : scénario difficile
        mode = 4 : scenario tres difficile
        mode = 5 : scenario extreme
        """
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def set_mode(self, mode):
        """
        Permet de changer le niveau de difficulté.
        """
        self.mode = mode

    def get_ranges(self):
        """
        Définit les plages de valeurs selon le mode choisi.
        """

        if self.mode == 1:
            # Mode facile : drone proche, plateforme simple, vent faible ou nul
            return {
                "x0_range": (-0.05, 0.05),
                "y0_range": (-0.05, 0.05),
                "z0_range": (1.35, 1.50),

                "roll0_range": (-0.03, 0.03),
                "pitch0_range": (-0.03, 0.03),
                "yaw0_range": (-0.05, 0.05),

                "platform_motion_enabled": True,
            
                "platform_motion_modes": [0, 1],

                "wind_enabled": False,
                "wx_mean_range": (-0.0, 0.0),
                "wy_mean_range": (-0.0, 0.0),
                "wz_mean_range": (-0.0, 0.0),
                "wx_sigma_range": (0.0, 0.0),
                "wy_sigma_range": (0.0, 0.0),
                "wz_sigma_range": (0.0, 0.0),
            }

        elif self.mode == 2:
            # Mode moyen : variation de la position initiale du drone+ plateforme simple + vent Dryden modéré
            return {
                "x0_range": (-0.15, 0.15),
                "y0_range": (-0.15, 0.15),
                "z0_range": (1.35, 1.75),

                "roll0_range": (-0.08, 0.08),
                "pitch0_range": (-0.08, 0.08),
                "yaw0_range": (-0.10, 0.10),

                "platform_motion_enabled": True,
                "platform_motion_modes": [0, 1],

                "wind_enabled": True,
                "wx_mean_range": (-0.15, 0.15),
                "wy_mean_range": (-0.15, 0.15),
                "wz_mean_range": (-0.03, 0.03),
                "wx_sigma_range": (0.05, 0.15),
                "wy_sigma_range": (0.05, 0.15),
                "wz_sigma_range": (0.02, 0.08),
            }

        elif self.mode == 3:
            # Mode difficile : plateforme plus complexe + vent plus fort
            return {
                "x0_range": (-0.15, 0.15),
                "y0_range": (-0.15, 0.15),
                "z0_range": (1.30, 1.70),

                "roll0_range": (-0.08, 0.08),
                "pitch0_range": (-0.08, 0.08),
                "yaw0_range": (-0.15, 0.15),

                "platform_motion_enabled": True,
                "platform_motion_modes": [1, 2],

                "wind_enabled": True,
                "wx_mean_range": (-1.0, 1.0),
                "wy_mean_range": (-1.0, 1.0),
                "wz_mean_range": (-0.5, 0.5),
                "wx_sigma_range": (0.5, 0.8),
                "wy_sigma_range": (0.5, 0.8),
                "wz_sigma_range": (0.15, 0.30),
            }


        elif self.mode == 4:
            # Mode tres difficile : perturbations initiales plus larges + vent fort
            return {
                "x0_range": (-0.25, 0.25),
                "y0_range": (-0.25, 0.25),
                "z0_range": (1.25, 1.80),

                "roll0_range": (-0.12, 0.12),
                "pitch0_range": (-0.12, 0.12),
                "yaw0_range": (-0.20, 0.20),

                "platform_motion_enabled": True,
                "platform_motion_modes": [1, 2],

                "wind_enabled": True,
                "wx_mean_range": (-1.5, 1.5),
                "wy_mean_range": (-1.5, 1.5),
                "wz_mean_range": (-0.75, 0.75),
                "wx_sigma_range": (0.8, 1.2),
                "wy_sigma_range": (0.8, 1.2),
                "wz_sigma_range": (0.30, 0.50),
            }

        elif self.mode == 5:
            # Mode extreme : conditions severes + vent tres fort
            return {
                "x0_range": (-0.35, 0.35),
                "y0_range": (-0.35, 0.35),
                "z0_range": (1.20, 1.90),

                "roll0_range": (-0.16, 0.16),
                "pitch0_range": (-0.16, 0.16),
                "yaw0_range": (-0.30, 0.30),

                "platform_motion_enabled": True,
                "platform_motion_modes": [1, 2],

                "wind_enabled": True,
                "wx_mean_range": (-2.0, 2.0),
                "wy_mean_range": (-2.0, 2.0),
                "wz_mean_range": (-1.0, 1.0),
                "wx_sigma_range": (1.2, 1.8),
                "wy_sigma_range": (1.2, 1.8),
                "wz_sigma_range": (0.50, 0.80),
            }

        else:
            raise ValueError(
                f"Mode inconnu: {self.mode}. Utilisez un mode entre 1 et 5."
            )

    def sample_episode_config(self):
        """
        Génère une configuration complète pour un épisode.
        Cette configuration sera utilisée dans reset().
        """

        ranges = self.get_ranges()

        init_xyzs = np.array(
            [[
                self.rng.uniform(*ranges["x0_range"]),
                self.rng.uniform(*ranges["y0_range"]),
                self.rng.uniform(*ranges["z0_range"]),
            ]],
            dtype=np.float32,
        )

        init_rpys = np.array(
            [[
                self.rng.uniform(*ranges["roll0_range"]),
                self.rng.uniform(*ranges["pitch0_range"]),
                self.rng.uniform(*ranges["yaw0_range"]),
            ]],
            dtype=np.float32,
        )

        platform_motion_mode = int(
            self.rng.choice(ranges["platform_motion_modes"])
        )

        wind_config = {
            "wind_enabled": ranges["wind_enabled"],

            "wx_mean": self.rng.uniform(*ranges["wx_mean_range"]),
            "wy_mean": self.rng.uniform(*ranges["wy_mean_range"]),
            "wz_mean": self.rng.uniform(*ranges["wz_mean_range"]),

            "wx_sigma": self.rng.uniform(*ranges["wx_sigma_range"]),
            "wy_sigma": self.rng.uniform(*ranges["wy_sigma_range"]),
            "wz_sigma": self.rng.uniform(*ranges["wz_sigma_range"]),

            "altitude": float(init_xyzs[0, 2]),
        }

        episode_config = {
            "mode": self.mode,
            "init_xyzs": init_xyzs,
            "init_rpys": init_rpys,
            "platform_motion_enabled": ranges["platform_motion_enabled"],
            "platform_motion_mode": platform_motion_mode,
            "wind_config": wind_config,
        }

        return episode_config
