import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox
)
from PyQt5.QtCore import QTimer

from simulation_manager import SimulationManager


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Interface d'appontage autonome - PyBullet")
        self.setGeometry(200, 100, 500, 350)

        self.sim_manager = SimulationManager()

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)

    def init_ui(self):
        main_layout = QVBoxLayout()

        title = QLabel("Interface de contrôle - Appontage autonome")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(title)

        # ===============================
        # Choix du modèle
        # ===============================
        model_group = QGroupBox("Choix du modèle PPO")
        model_layout = QVBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.addItem("Correction de position")
        self.model_combo.addItem("Correction de vitesse")

        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)

        main_layout.addWidget(model_group)

        # ===============================
        # Boutons de contrôle
        # ===============================
        control_group = QGroupBox("Contrôle de la simulation")
        control_layout = QHBoxLayout()

        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Reprendre")
        self.reset_button = QPushButton("Reset")
        self.stop_button = QPushButton("Stop")

        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.resume_button)
        control_layout.addWidget(self.reset_button)
        control_layout.addWidget(self.stop_button)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        # ===============================
        # Informations
        # ===============================
        info_group = QGroupBox("État de la simulation")
        info_layout = QVBoxLayout()

        self.status_label = QLabel("État : en attente")
        self.reward_label = QLabel("Reward : ---")
        self.done_label = QLabel("Épisode terminé : ---")

        info_layout.addWidget(self.status_label)
        info_layout.addWidget(self.reward_label)
        info_layout.addWidget(self.done_label)

        info_group.setLayout(info_layout)
        main_layout.addWidget(info_group)

        self.setLayout(main_layout)

        # Connexion des boutons
        self.start_button.clicked.connect(self.start_simulation)
        self.pause_button.clicked.connect(self.pause_simulation)
        self.resume_button.clicked.connect(self.resume_simulation)
        self.reset_button.clicked.connect(self.reset_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)

    def start_simulation(self):
        selected_model = self.model_combo.currentText()

        if selected_model == "Correction de position":
            model_type = "position"
        else:
            model_type = "vitesse"

        self.status_label.setText(f"État : lancement du modèle {model_type}")

        self.sim_manager.load_simulation(model_type)

        self.timer.start(30)  # mise à jour toutes les 30 ms

        self.status_label.setText(f"État : simulation lancée ({model_type})")

    def pause_simulation(self):
        self.sim_manager.pause_simulation()
        self.status_label.setText("État : simulation en pause")

    def resume_simulation(self):
        self.sim_manager.resume_simulation()
        self.status_label.setText("État : simulation reprise")

    def reset_simulation(self):
        self.sim_manager.reset_simulation()
        self.status_label.setText("État : simulation réinitialisée")
        self.reward_label.setText("Reward : ---")
        self.done_label.setText("Épisode terminé : ---")

    def stop_simulation(self):
        self.timer.stop()
        self.sim_manager.stop_simulation()
        self.status_label.setText("État : simulation arrêtée")
        self.reward_label.setText("Reward : ---")
        self.done_label.setText("Épisode terminé : ---")

    def update_simulation(self):
        result = self.sim_manager.step_simulation()

        if result is None:
            return

        reward = result["reward"]
        done = result["done"]

        self.reward_label.setText(f"Reward : {reward:.3f}")
        self.done_label.setText(f"Épisode terminé : {done}")

        if done:
            self.timer.stop()
            self.status_label.setText("État : épisode terminé")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())