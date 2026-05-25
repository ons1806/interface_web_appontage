import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Paramètres du système ---
# M_tot : masse généralisée (masse + masse ajoutée)
M_tot = np.eye(6) * 1000000.0

# C : matrice d'amortissement hydrodynamique
C = np.eye(6) * 500.0

# K : matrice de raideur hydrostatique
# La raideur n'agit pas sur x, y et lacet (indices 0, 1 et 5) car // à la surface de l'eau => pas de modif du volume immergé => pas de rappel hydrostatique
K = np.zeros(6)
K[2] = 20000.0  # Raideur en pilonnement (z)
K[3] = 2000.0  # Raideur en roulis
K[4] = 2000.0  # Raideur en tangage

def force_houle(t):
    """
    Définit le vecteur des efforts hydrodynamiques d'excitation F(t).
    Ici, on simule une houle régulière agissant uniquement sur le pilonnement (z, indice 2).
    """
    F = np.zeros(6)
    omega = 0.4  # Pulsation de la houle en rad/s
    F[2] = 5000.0 * np.cos(omega * t)  # Force appliquée sur le pilonnement (z)
    return F

def equations_mouvement(t, y):
    """
    Fonction d'état pour le solveur ODE.
    y est un vecteur de dimension 12 contenant [S (6 positions), dS/dt (6 vitesses)]
    """
    S = y[0:6]
    dSdt = y[6:12]

    # Calcul des forces d'excitation à l'instant t
    F = force_houle(t)

    # Résolution de l'accélération : d2S/dt2 = M_tot^-1 * (F - C*dS/dt - K*S)
    forces_restantes = F - np.dot(C, dSdt) - np.dot(K, S)
    d2Sdt2 = np.linalg.solve(M_tot, forces_restantes)

    # On retourne le dérivé du vecteur d'état : [vitesses, accélérations]
    return np.concatenate((dSdt, d2Sdt2))

# --- Conditions initiales et résolution ---
# Le navire part de sa position de repos (S=0) avec une vitesse nulle (dSdt=0)

def get_DOF(affichage = True):
    y0 = np.zeros(12)

    # Intervalle de temps pour la simulation
    t_span = (0, 100)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # Résolution numérique du système
    solution = solve_ivp(equations_mouvement, t_span, y0, t_eval=t_eval, method='RK45')

    # --- Affichage des résultats ---
    temps = solution.t
    cavalement_x = solution.y[0]  # Indice 0 correspond à la translation x (cavèlement)
    embardee_y = solution.y[1]  # Indice 1 correspond à la translation y (embardée)
    pilonnement_z = solution.y[2]  # Indice 2 correspond à la translation z (pilonnement)
    Roulis_Rx = solution.y[3]  # Indice 3 correspond au roulis (Rx)
    Tangage_Ry = solution.y[4]  # Indice 4 correspond au tangage (Ry)
    Lacet_Rz = solution.y[5]  # Indice 5 correspond au lacet (Rz)

    if affichage:
        fig, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True)

        # Titres et labels pour chaque sous-graphique
        titles = [
            "Cavèlement (x)",
            "Embardée (y)",
            "Pilonnement (z)",
            "Roulis (Rx)",
            "Tangage (Ry)",
            "Lacet (Rz)"
        ]
        colors = ["blue", "orange", "green", "red", "purple", "brown"]

        # Tracer chaque mouvement
        for i, ax in enumerate(axes):
            ax.plot(temps, solution.y[i], label=titles[i], color=colors[i])
            ax.set_ylabel("Déplacement (m ou rad)")
            ax.set_title(titles[i])
            ax.grid(True)
            ax.legend()

        # Ajouter un label commun pour l'axe x
        axes[-1].set_xlabel("Temps (s)")

        # Ajustement de l'espacement entre les sous-graphiques
        plt.tight_layout()
        plt.show()

    return temps, cavalement_x, embardee_y, pilonnement_z, Roulis_Rx, Tangage_Ry, Lacet_Rz

