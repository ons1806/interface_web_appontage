import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CustomDrydenWind:
    """
    Modèle de vent Dryden-like en Python pur.

    API inspirée de goromal/wind-dynamics :
        - initialize(...)
        - getWind(dt)

    Idée :
        - axe x (u) : filtre du 1er ordre
        - axes y,z (v,w) : filtres du 2e ordre
        - vent total = vent moyen + composante turbulente
    """

    seed: Optional[int] = None
    default_V: float = 1.0       # vitesse de traversée du champ turbulent
    max_sub_dt: float = 0.01     # sous-pas pour stabilité numérique

    rng: np.random.Generator = field(init=False)

    mean_wind: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3, dtype=float))
    sigma: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3, dtype=float))
    L: np.ndarray = field(init=False, default_factory=lambda: np.ones(3, dtype=float))

    # Etats internes
    x_u: float = field(init=False, default=0.0)                 # état 1er ordre pour u
    x_v: np.ndarray = field(init=False, default_factory=lambda: np.zeros(2, dtype=float))
    x_w: np.ndarray = field(init=False, default_factory=lambda: np.zeros(2, dtype=float))

    initialized: bool = field(init=False, default=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def _length_scales_from_altitude(self, altitude_m: float) -> np.ndarray:
        """
        Loi inspirée du repo goromal/wind-dynamics.
        Le repo convertit l'altitude en pieds, calcule Lz, puis Lx=Ly
        avant de reconvertir en mètres.
        """
        altitude_m = max(0.1, float(altitude_m))
        Lz_ft = 3.281 * altitude_m
        Lx_ft = Lz_ft / ((0.177 + 0.000823 * Lz_ft) ** 1.2)
        Ly_ft = Lx_ft
        return np.array([Lx_ft / 3.281, Ly_ft / 3.281, Lz_ft / 3.281], dtype=float)

    def initialize(
        self,
        wx_mean: float,
        wy_mean: float,
        wz_mean: float,
        wx_sigma: float,
        wy_sigma: float,
        wz_sigma: float,
        altitude: float = 2.0,
        Lx: Optional[float] = None,
        Ly: Optional[float] = None,
        Lz: Optional[float] = None,
    ) -> None:
        """
        Initialise le vent moyen, les intensités turbulentes et les longueurs
        de turbulence.
        """
        self.mean_wind = np.array([wx_mean, wy_mean, wz_mean], dtype=float)
        self.sigma = np.array([wx_sigma, wy_sigma, wz_sigma], dtype=float)

        if Lx is None or Ly is None or Lz is None:
            self.L = self._length_scales_from_altitude(altitude)
        else:
            self.L = np.array([Lx, Ly, Lz], dtype=float)

        self.reset_states()
        self.initialized = True

    def reset_states(self) -> None:
        self.x_u = 0.0
        self.x_v[:] = 0.0
        self.x_w[:] = 0.0

    def _step_u(self, dt: float, V: float) -> float:
        """
        Filtre Dryden longitudinal du 1er ordre :
            G_u(s) = sigma_u * sqrt(2L_u/(pi V)) / (1 + (L_u/V)s)
        Implémentation Euler-Maruyama.
        """
        Lu = max(1e-6, self.L[0])
        sigma_u = self.sigma[0]

        tau = Lu / V
        a = 1.0 / tau
        k = sigma_u * np.sqrt(2.0 * Lu / (np.pi * V))
        b = k / tau

        n = self.rng.standard_normal()
        self.x_u += (-a * self.x_u) * dt + b * np.sqrt(dt) * n
        return self.x_u

    def _step_vw(self, state: np.ndarray, L: float, sigma: float, dt: float, V: float) -> tuple[np.ndarray, float]:
        """
        Filtre Dryden latéral/vertical du 2e ordre :
            G(s) = sigma * sqrt(2L/(pi V)) * (1 + (2sqrt(3)L/V)s) / (1 + (2L/V)s)^2

        Réalisation d'état simple :
            xdot = A x + B w
            y    = C x
        """
        L = max(1e-6, L)
        sigma = float(sigma)

        a = 2.0 * L / V
        b = 2.0 * np.sqrt(3.0) * L / V
        k = sigma * np.sqrt(2.0 * L / (np.pi * V))

        # Denominateur (1 + a s)^2 = a^2 s^2 + 2a s + 1
        A = np.array([
            [0.0, 1.0],
            [-1.0 / (a * a), -2.0 / a]
        ], dtype=float)

        B = np.array([0.0, 1.0], dtype=float)

        # Choix de C pour obtenir le bon numérateur
        C = np.array([
            k / (a * a),
            k * b / (a * a)
        ], dtype=float)

        n = self.rng.standard_normal()
        state = state + (A @ state) * dt + B * np.sqrt(dt) * n
        y = float(C @ state)
        return state, y

    def getWind(self, dt: float, V: Optional[float] = None) -> np.ndarray:
        """
        Renvoie le vecteur vent [wx, wy, wz] en m/s.
        V est la vitesse de traversée du champ turbulent.
        """
        if not self.initialized:
            return np.zeros(3, dtype=float)

        dt = float(dt)
        if dt <= 0.0:
            return self.mean_wind.copy()

        Veff = self.default_V if V is None else max(0.1, float(V))

        gust = np.zeros(3, dtype=float)

        t = 0.0
        while t < dt:
            h = min(self.max_sub_dt, dt - t)

            gust[0] = self._step_u(h, Veff)
            self.x_v, gust[1] = self._step_vw(self.x_v, self.L[1], self.sigma[1], h, Veff)
            self.x_w, gust[2] = self._step_vw(self.x_w, self.L[2], self.sigma[2], h, Veff)

            t += h

        return self.mean_wind + gust