import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gymnasium as gym
import numpy as np
import pybullet as p
from SimulateurHoule import get_DOF
from typing import Optional
from gymnasium import spaces
from meta_model import MetaModel
from mon_modele_dryden import CustomDrydenWind
from mpc_controller import LocalLandingMPC

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class DroneLandingRLEnv(gym.Env):
    # Métadonnées standard Gymnasium indiquant que cet environnement supporte le rendu "humain"
    metadata = {"render_modes": ["human"], "render_fps": 48}

    def __init__(
        self,
        gui: bool = True,
        episode_len_sec: float = 10.0,
        normalized_action_input: bool = False,
    ):
        super().__init__()

        # Active ou non l'affichage PyBullet pendant la simulation
        self.gui = gui
        self.normalized_action_input = normalized_action_input

        # Durée maximale d’un épisode en secondes (ensuite l'épisode est tronqué)
        self.episode_len_sec = episode_len_sec

        # Modele de drone utilise pour la simulation PyBullet.
        self.drone_model = DroneModel("quadricopter")

        # On travaille avec un seul drone
        self.num_drones = 1

        # Moteur physique PyBullet
        self.physics = Physics("pyb")

        # Fréquence de simulation physique
        self.simulation_freq_hz = 240

        # Fréquence de contrôle (à quelle fréquence PPO envoie une action)
        self.control_freq_hz = 24

        # État initial par défaut du drone (position et orientation)
        # Il sera remplacé par une valeur aléatoire à chaque reset() pour rendre l'apprentissage plus robuste
        self.init_xyzs = np.array([[0.0, 0.0, 1.5]], dtype=np.float32)
        self.init_rpys = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        # Position de base de la plateforme
        self.platform_base_position = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Point cible d'atterrissage relatif à la plateforme
        # à ajuster selon la vraie géométrie du mesh
        self.landing_target_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Position absolue initiale de la cible
        self.landing_target = self.platform_base_position + self.landing_target_offset

        # Chemin du mesh .obj
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.platform_obj_path = os.path.join(
            current_dir,
            "asset",
            "plateforme_obj",
            "STD04_0100_Plateaux_M600 Mobile Aluminium.obj"
        )

        # IDs PyBullet de la plateforme
        self.platform_id = None
        self.platform_visual_id = None
        self.platform_collision_id = None
        self.gui_rotor_marker_ids = []
        self.gui_line_ids = {}

        # Mouvement de plateforme
        self.platform_motion_enabled = True
        self.platform_motion_mode = 2   # 0 statique, 1, 2, 3 comme dans ton script
        self.platform_speed_factor = 1.0
        self.motion_index = 0
        self.motion_t = None
        self.motion_x = None
        self.motion_y = None
        self.motion_z = None
        self.motion_Rx = None
        self.motion_Ry = None
        self.motion_Rz = None
        self.prev_landing_target = None
        self.platform_position = self.platform_base_position.copy()
        self.platform_rpy = np.zeros(3, dtype=np.float32)
        self.platform_target_velocity = np.zeros(3, dtype=np.float32)

        # Espace des observations
        # Observation = [
        #   x_rel, y_rel, z_rel, vx, vy, vz, roll, pitch, yaw, wx, wy, wz,
        #   wind_x, wind_y, wind_z,
        #   platform_vx, platform_vy, platform_vz,
        #   platform_roll, platform_pitch, platform_yaw
        # ]
        obs_low = np.array(
            [
                -2.0, -2.0, -1.0,
                -5.0, -5.0, -5.0,
                -np.pi, -np.pi, -np.pi,
                -20.0, -20.0, -20.0,
                -5.0, -5.0, -5.0,
                -5.0, -5.0, -5.0,
                -np.pi, -np.pi, -np.pi,
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                2.0, 2.0, 2.0,
                5.0, 5.0, 5.0,
                np.pi, np.pi, np.pi,
                20.0, 20.0, 20.0,
                5.0, 5.0, 5.0,
                5.0, 5.0, 5.0,
                np.pi, np.pi, np.pi,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Espace des actions
        # Action = [dvx_cmd, dvy_cmd, dvz_cmd]
        # dvx_cmd, dvy_cmd = corrections de vitesse horizontale autour de la vitesse plateforme
        # dvz_cmd = correction de vitesse verticale autour de la vitesse plateforme
        self.command_low = np.array([-0.30, -0.30, -0.20], dtype=np.float32)
        self.command_high = np.array([0.30, 0.30, 0.10], dtype=np.float32)

        if self.normalized_action_input:
            # PPO travaille dans [-1, 1], puis l'environnement convertit en m/s.
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            # Compatibilite avec les anciens modeles deja entraines.
            self.action_space = spaces.Box(
                low=self.command_low,
                high=self.command_high,
                dtype=np.float32,
            )

        # Variables internes pour gérer l'environnement et la simulation
        self.env: Optional[CtrlAviary] = None       # environnement PyBullet drone
        self.ctrl: Optional[DSLPIDControl] = None   # contrôleur PID utilisé temporairement en simulation
        self.mpc: Optional[LocalLandingMPC] = None
        self.pyb_client: Optional[int] = None       # identifiant client PyBullet
        self.drone_id: Optional[int] = None         # identifiant du drone
    
        self.hover_near_target_steps = 0
        self.current_step = 0
        self.max_steps = int(self.episode_len_sec * self.control_freq_hz)

        # Dernière action RL brute envoyée à l’environnement
        self.last_action = np.zeros(3, dtype=np.float32)
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.last_policy_action = np.zeros(3, dtype=np.float32)

        self.contact_stable_steps = 0
        self.required_contact_steps = 5
        self.prev_landing_error = None
        self.stable_xy_threshold = 0.06
        self.precise_xy_threshold = 0.035
        self.initial_descent_rate = 0.35
        self.nominal_descent_rate = 0.65
        self.descent_ramp_duration_sec = 0.80
        self.max_safe_descent_speed = 0.10
        self.min_target_z_margin = -0.01
        self.use_mpc_target_filter = False
        
        # Meta-model
        self.meta_model = MetaModel(mode=1)
        self.current_scenario = None
        self.episode_count = 0
        # Variation automatique du mode du meta-model
        self.use_random_meta_mode = True
        self.available_meta_modes = [1, 2, 3, 4, 5]

        # Modèle de vent
        self.wind_model = CustomDrydenWind(seed=None)
        self.wind_enabled = False
        self.current_wind = np.zeros(3, dtype=np.float32)
        self.wind_force_scale = 0.30

    def _build_env(self):
        """
        Construit l’environnement PyBullet et ajoute la plateforme.
        """
        self.env = CtrlAviary(
            drone_model=self.drone_model,
            num_drones=self.num_drones,
            initial_xyzs=self.init_xyzs,
            initial_rpys=self.init_rpys,
            physics=self.physics,
            neighbourhood_radius=10,
            pyb_freq=self.simulation_freq_hz,
            ctrl_freq=self.control_freq_hz,
            gui=self.gui,
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )

        # Création du contrôleur PID
        # Ici, le PPO ne commande plus directement les moteurs.
        # Il génère une cible haut niveau, puis le PID calcule les RPM moteurs.
        self.ctrl = DSLPIDControl(drone_model=self.drone_model)
        self.mpc = LocalLandingMPC(
            horizon=8,
            dt=1.0 / self.control_freq_hz,
            max_xy_speed=0.35,
            max_descent_speed=0.14,
            max_climb_speed=0.10,
            max_accel=0.70,
        )

        # Récupération des infos utiles depuis l’environnement
        self.pyb_client = self.env.getPyBulletClient()
        self.drone_id = self.env.DRONE_IDS[0]

        # Création du mesh visuel
        self.platform_visual_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=self.platform_obj_path,
            meshScale=[1, 1, 1],
            rgbaColor=[0.82, 0.86, 0.88, 1.0],
            physicsClientId=self.pyb_client,
        )

        # Création du mesh de collision
        self.platform_collision_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=self.platform_obj_path,
            meshScale=[1, 1, 1],
            physicsClientId=self.pyb_client,
        )

        # Création du corps rigide de la plateforme
        self.platform_id = p.createMultiBody(
            baseMass=0,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=self.platform_collision_id,
            baseVisualShapeIndex=self.platform_visual_id,
            basePosition=self.platform_base_position.tolist(),
            physicsClientId=self.pyb_client,
        )

        p.changeDynamics(
            self.platform_id,
            -1,
            lateralFriction=2.2,
            spinningFriction=0.08,
            rollingFriction=0.02,
            restitution=0.0,
            physicsClientId=self.pyb_client,
        )
        p.changeDynamics(
            self.drone_id,
            -1,
            lateralFriction=1.6,
            spinningFriction=0.05,
            rollingFriction=0.02,
            restitution=0.0,
            physicsClientId=self.pyb_client,
        )

        p.changeVisualShape(
            self.drone_id,
            -1,
            rgbaColor=[0.08, 0.18, 0.95, 1.0],
            physicsClientId=self.pyb_client,
        )

        if self.gui:
            self._setup_gui_visual_helpers()

        p.resetDebugVisualizerCamera(
            cameraDistance=1.35,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=(self.platform_base_position + np.array([0.0, 0.0, 0.22])).tolist(),
            physicsClientId=self.pyb_client,
        )

    def _setup_gui_visual_helpers(self):
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.pyb_client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.pyb_client)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self.pyb_client)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self.pyb_client)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self.pyb_client)

        colors = [
            [1.0, 0.12, 0.12, 1.0],
            [0.1, 0.35, 1.0, 1.0],
            [0.1, 0.8, 0.25, 1.0],
            [1.0, 0.85, 0.05, 1.0],
        ]
        self.gui_rotor_marker_ids = []
        for color in colors:
            marker_visual = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.030,
                rgbaColor=color,
                physicsClientId=self.pyb_client,
            )
            marker_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=marker_visual,
                basePosition=[0.0, 0.0, 0.0],
                physicsClientId=self.pyb_client,
            )
            self.gui_rotor_marker_ids.append(marker_id)

        target_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.11,
            length=0.01,
            rgbaColor=[0.0, 0.65, 1.0, 0.85],
            physicsClientId=self.pyb_client,
        )
        self.gui_target_marker_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=target_visual,
            basePosition=self.landing_target.tolist(),
            physicsClientId=self.pyb_client,
        )

    def _update_gui_visual_helpers(self):
        if not self.gui or self.pyb_client is None or self.drone_id is None:
            return

        pos, quat = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.pyb_client)
        pos = np.array(pos, dtype=np.float32)
        rot = np.array(p.getMatrixFromQuaternion(quat), dtype=np.float32).reshape(3, 3)

        rotor_offsets = [
            np.array([0.12, 0.12, 0.0], dtype=np.float32),
            np.array([0.12, -0.12, 0.0], dtype=np.float32),
            np.array([-0.12, -0.12, 0.0], dtype=np.float32),
            np.array([-0.12, 0.12, 0.0], dtype=np.float32),
        ]
        rotor_positions = [pos + rot @ offset for offset in rotor_offsets]

        for marker_id, marker_pos in zip(self.gui_rotor_marker_ids, rotor_positions):
            p.resetBasePositionAndOrientation(
                marker_id,
                marker_pos.tolist(),
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.pyb_client,
            )

        line_specs = [
            ("arm_a", rotor_positions[0], rotor_positions[2], [1.0, 1.0, 1.0]),
            ("arm_b", rotor_positions[1], rotor_positions[3], [1.0, 1.0, 1.0]),
            ("altitude", pos, [pos[0], pos[1], self.landing_target[2]], [0.0, 0.65, 1.0]),
        ]
        for name, start, end, color in line_specs:
            line_id = self.gui_line_ids.get(name, -1)
            self.gui_line_ids[name] = p.addUserDebugLine(
                start,
                end,
                lineColorRGB=color,
                lineWidth=2.0,
                lifeTime=0,
                replaceItemUniqueId=line_id,
                physicsClientId=self.pyb_client,
            )

        if hasattr(self, "gui_target_marker_id"):
            p.resetBasePositionAndOrientation(
                self.gui_target_marker_id,
                self.landing_target.tolist(),
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.pyb_client,
            )

        camera_target = (0.65 * self.landing_target + 0.35 * pos + np.array([0.0, 0.0, 0.18])).tolist()
        p.resetDebugVisualizerCamera(
            cameraDistance=1.10,
            cameraYaw=42,
            cameraPitch=-24,
            cameraTargetPosition=camera_target,
            physicsClientId=self.pyb_client,
        )

    def _set_platform_pose(self, x, y, z, Rx, Ry, Rz):
        quat = p.getQuaternionFromEuler([Rx, Ry, Rz])
        self.platform_position = np.array([x, y, z], dtype=np.float32)
        self.platform_rpy = np.array([Rx, Ry, Rz], dtype=np.float32)
        p.resetBasePositionAndOrientation(
            self.platform_id,
            [x, y, z],
            quat,
            physicsClientId=self.pyb_client,
        )

        # Met à jour la cible d'atterrissage absolue
        rot_matrix = np.array(p.getMatrixFromQuaternion(quat), dtype=np.float32).reshape(3, 3)
        new_landing_target = np.array([x, y, z], dtype=np.float32) + rot_matrix @ self.landing_target_offset
        if self.prev_landing_target is not None:
            dt = 1.0 / self.control_freq_hz
            self.platform_target_velocity = (
                (new_landing_target - self.prev_landing_target) / dt
            ).astype(np.float32)
        else:
            self.platform_target_velocity = np.zeros(3, dtype=np.float32)
        self.landing_target = new_landing_target
        self.prev_landing_target = new_landing_target.copy()

    def _init_platform_motion(self):
        if not self.platform_motion_enabled:
            return

        if self.platform_motion_mode == 0:
            self.motion_t = None
            return

        elif self.platform_motion_mode == 1:
            tmax = 100
            dt = 1.0 / self.control_freq_hz
            self.motion_t = np.arange(0, tmax, dt)
            angle_max_Rx = np.deg2rad(30)

            self.motion_x = np.zeros_like(self.motion_t)
            self.motion_y = np.zeros_like(self.motion_t)
            self.motion_z = np.zeros_like(self.motion_t)

            self.motion_Rx = (angle_max_Rx / 2.0) * np.sin(
                0.05 * self.platform_speed_factor * self.motion_t
            )
            self.motion_Ry = np.zeros_like(self.motion_t)
            self.motion_Rz = np.zeros_like(self.motion_t)

        elif self.platform_motion_mode == 2:
            tmax = 100
            dt = 1.0 / self.control_freq_hz
            self.motion_t = np.arange(0, tmax, dt)
            angle_max_Rx = np.deg2rad(30)

            self.motion_x = np.zeros_like(self.motion_t)
            self.motion_y = np.zeros_like(self.motion_t)
            self.motion_z = 0.02 * np.sin(0.2 * self.platform_speed_factor * self.motion_t)
            self.motion_Rx = (angle_max_Rx / 2.0) * np.sin(
                0.2 * self.platform_speed_factor * self.motion_t
            )
            self.motion_Ry = np.zeros_like(self.motion_t)
            self.motion_Rz = np.zeros_like(self.motion_t)

        elif self.platform_motion_mode == 3:
            (
                self.motion_t,
                self.motion_x,
                self.motion_y,
                self.motion_z,
                self.motion_Rx,
                self.motion_Ry,
                self.motion_Rz,
            ) = get_DOF(affichage=False)
            if self.platform_speed_factor != 1.0:
                scaled_t = np.minimum(
                    self.motion_t * self.platform_speed_factor,
                    self.motion_t[-1],
                )
                self.motion_x = np.interp(scaled_t, self.motion_t, self.motion_x)
                self.motion_y = np.interp(scaled_t, self.motion_t, self.motion_y)
                self.motion_z = np.interp(scaled_t, self.motion_t, self.motion_z)
                self.motion_Rx = np.interp(scaled_t, self.motion_t, self.motion_Rx)
                self.motion_Ry = np.interp(scaled_t, self.motion_t, self.motion_Ry)
                self.motion_Rz = np.interp(scaled_t, self.motion_t, self.motion_Rz)

        self.motion_index = 0

    def _get_obs_from_state(self, state: np.ndarray) -> np.ndarray:
        """
        Transforme l’état brut PyBullet en observation RL compacte.
        """
        state = np.asarray(state, dtype=np.float32)

        # Position du drone
        drone_pos = state[0:3]

        # Angles roll, pitch, yaw
        drone_rpy = state[7:10] if state.shape[0] >= 10 else np.zeros(3, dtype=np.float32)

        # Vitesses linéaires vx, vy, vz
        drone_vel = state[10:13] if state.shape[0] >= 13 else np.zeros(3, dtype=np.float32)

        # Vitesses angulaires wx, wy, wz
        drone_ang_vel = state[13:16] if state.shape[0] >= 16 else np.zeros(3, dtype=np.float32)
        
        # Vitesse actuelle du vent
        wind_vel = self.current_wind.astype(np.float32)

        # Position relative par rapport au point cible d'atterrissage
        relative_pos = drone_pos - self.landing_target

        # Donnees de la plateforme donnees a l'agent
        platform_vel = self.platform_target_velocity.astype(np.float32)
        platform_rpy = self.platform_rpy.astype(np.float32)

        # Observation finale donnée à l’agent PPO
        obs = np.array(
            [
                relative_pos[0],
                relative_pos[1],
                relative_pos[2],
                drone_vel[0],
                drone_vel[1],
                drone_vel[2],
                drone_rpy[0],
                drone_rpy[1],
                drone_rpy[2],
                drone_ang_vel[0],
                drone_ang_vel[1],
                drone_ang_vel[2],
                wind_vel[0],
                wind_vel[1],
                wind_vel[2],
                platform_vel[0],
                platform_vel[1],
                platform_vel[2],
                platform_rpy[0],
                platform_rpy[1],
                platform_rpy[2],
            ],
            dtype=np.float32,
        )
        return obs
    def _update_meta_mode(self):
       """
       Choisit aléatoirement le mode du meta-model
       à chaque nouvel épisode.
       """

       if self.use_random_meta_mode:
         new_mode = np.random.choice(self.available_meta_modes)
         self.meta_model.set_mode(int(new_mode))
         
    def _generate_and_apply_scenario(self):
       """
       Génère un nouveau scénario avec le meta-model
       et applique ses paramètres dans l'environnement.
       """

       # Génération d'un nouveau scénario
       self.current_scenario = self.meta_model.sample_episode_config()

       # Application de l'état initial du drone
       self.init_xyzs = self.current_scenario["init_xyzs"]
       self.init_rpys = self.current_scenario["init_rpys"]

       # Application du mouvement de la plateforme
       self.platform_motion_enabled = self.current_scenario["platform_motion_enabled"]
       self.platform_motion_mode = self.current_scenario["platform_motion_mode"]

       # Application du vent
       wind_config = self.current_scenario["wind_config"]
       self.wind_enabled = wind_config["wind_enabled"]


       if self.wind_enabled:
          self.wind_model.initialize(
            wx_mean=wind_config["wx_mean"],
            wy_mean=wind_config["wy_mean"],
            wz_mean=wind_config["wz_mean"],
            wx_sigma=wind_config["wx_sigma"],
            wy_sigma=wind_config["wy_sigma"],
            wz_sigma=wind_config["wz_sigma"],
            altitude=wind_config["altitude"],
          )
          
        # Génère une première valeur de vent dès le reset()
          dt = 1.0 / self.control_freq_hz
          self.current_wind = self.wind_model.getWind(dt).astype(np.float32)
       else:
          self.current_wind = np.zeros(3, dtype=np.float32)

    def reset(self, seed=None, options=None):
      """
      Réinitialise l’environnement au début d’un épisode
      sans reconstruire tout PyBullet à chaque fois.
      """
      super().reset(seed=seed)

      if seed is not None:
        np.random.seed(seed)

      self.episode_count += 1

      # Choix du mode avant de générer le scénario
      self._update_meta_mode()

      # Génération du scénario dans le mode choisi
      self._generate_and_apply_scenario()

      # Construction de l’environnement une seule fois
      if self.env is None:
        self._build_env()

      # Réinitialisation du mouvement de plateforme
      self._init_platform_motion()

      # Remise de la plateforme à sa pose initiale
      self.prev_landing_target = None
      self.platform_target_velocity = np.zeros(3, dtype=np.float32)
      self._set_platform_pose(
        self.platform_base_position[0],
        self.platform_base_position[1],
        self.platform_base_position[2],
        0.0, 0.0, 0.0
      )

      # Remise du drone à son état initial
      init_pos = self.init_xyzs[0]
      init_rpy = self.init_rpys[0]
      init_quat = p.getQuaternionFromEuler(init_rpy.tolist())

      p.resetBasePositionAndOrientation(
        self.drone_id,
        init_pos.tolist(),
        init_quat,
        physicsClientId=self.pyb_client,
      )

      p.resetBaseVelocity(
        self.drone_id,
        linearVelocity=[0.0, 0.0, 0.0],
        angularVelocity=[0.0, 0.0, 0.0],
        physicsClientId=self.pyb_client,
      )

      # Réinitialisation des variables internes
      self.current_step = 0
      self.last_action = np.zeros(3, dtype=np.float32)
      self.prev_action = np.zeros(3, dtype=np.float32)
      self.last_policy_action = np.zeros(3, dtype=np.float32)
      self.contact_stable_steps = 0
      self.hover_near_target_steps = 0
      self.prev_landing_error = None
      if self.mpc is not None:
        self.mpc.reset()

      # Lecture directe de l'etat initial pour eviter un pas avec moteurs a zero,
      # qui provoquait une petite chute artificielle au debut.
      state = self.env._getDroneStateVector(0)
      rl_obs = self._get_obs_from_state(state)
      self._update_gui_visual_helpers()
      x_rel, y_rel, z_rel = rl_obs[0:3]
      self.prev_landing_error = float(np.sqrt(x_rel**2 + y_rel**2 + 0.50 * z_rel**2))

      info = {
             "episode_count": int(self.episode_count),
            "meta_mode": int(self.meta_model.mode),
            "platform_motion_mode": int(self.platform_motion_mode),
            "wind_enabled": bool(self.wind_enabled),
            "init_x": float(self.init_xyzs[0, 0]),
            "init_y": float(self.init_xyzs[0, 1]),
            "init_z": float(self.init_xyzs[0, 2]),
     }
      return rl_obs, info

    def step(self, action: np.ndarray):
        """
        Applique une action, fait avancer la simulation, calcule reward et conditions d’arrêt.
        """
        # Convertit l'action PPO en correction de vitesse envoyee au PID.
        policy_action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        self.last_policy_action = policy_action.copy()
        if self.normalized_action_input:
            raw_action = self.command_low + 0.5 * (policy_action + 1.0) * (
                self.command_high - self.command_low
            )
        else:
            raw_action = policy_action.copy()

        # Lissage de l'action pour eviter les changements brusques
        alpha = 0.3
        action = alpha * self.prev_action + (1.0 - alpha) * raw_action
        
        # L'action du PPO represente une correction de vitesse autour de la plateforme.
        # [dvx_cmd, dvy_cmd, dvz_cmd] en m/s
        dvx_cmd, dvy_cmd, dvz_cmd = action
        self.last_action = action.copy()

        action_delta = action - self.prev_action
        action_smooth_penalty = float(np.sum(action_delta ** 2))
        self.prev_action = action.copy()

        if self.platform_motion_enabled and self.motion_t is not None:
            idx = min(self.motion_index, len(self.motion_t) - 1)

            x = self.platform_base_position[0] + self.motion_x[idx]
            y = self.platform_base_position[1] + self.motion_y[idx]
            z = self.platform_base_position[2] + self.motion_z[idx]

            Rx = self.motion_Rx[idx]
            Ry = self.motion_Ry[idx]
            Rz = self.motion_Rz[idx]

            self._set_platform_pose(x, y, z, Rx, Ry, Rz)
            self.motion_index += 1

        # Yaw imposé pour éviter l’agitation inutile
        state = self.env._getDroneStateVector(0)
        current_yaw = float(state[9]) if state.shape[0] >= 10 else 0.0
        target_yaw = 0.0
        yaw_error = target_yaw - current_yaw
        dt = 1.0 / self.control_freq_hz

        # Construction de la cible absolue :
        # la position cible reste centree sur la plateforme, tandis que PPO corrige
        # la vitesse de poursuite autour de la vitesse de la plateforme.
        current_z = float(state[2])
        current_vz = float(state[12])
        ramp_steps = max(1, int(self.descent_ramp_duration_sec * self.control_freq_hz))
        ramp_ratio = min(1.0, self.current_step / ramp_steps)
        descent_rate = (
            self.initial_descent_rate
            + (self.nominal_descent_rate - self.initial_descent_rate) * ramp_ratio
        )
        limited_target_z = max(
            float(self.landing_target[2] + self.min_target_z_margin),
            current_z - descent_rate * dt,
        )
        target_vel = self.platform_target_velocity.copy()
        target_vel += np.array([dvx_cmd, dvy_cmd, dvz_cmd], dtype=np.float32)
        if current_vz < -self.max_safe_descent_speed:
            limited_target_z = max(limited_target_z, current_z)
            target_vel[2] = max(0.0, float(target_vel[2]))
        target_pos = np.array([
            self.landing_target[0],
            self.landing_target[1],
            limited_target_z,
        ], dtype=np.float32)
        if self.use_mpc_target_filter and self.mpc is not None:
            mpc_target_pos = self.mpc.compute_target(
                current_pos=state[0:3],
                desired_pos=target_pos,
                platform_velocity=target_vel,
            )
            z_to_deck = max(0.0, current_z - float(self.landing_target[2]))
            xy_mpc_blend = float(np.clip((0.16 - z_to_deck) / 0.12, 0.0, 0.18))
            target_pos = np.array(
                [
                    (1.0 - xy_mpc_blend) * target_pos[0] + xy_mpc_blend * mpc_target_pos[0],
                    (1.0 - xy_mpc_blend) * target_pos[1] + xy_mpc_blend * mpc_target_pos[1],
                    mpc_target_pos[2],
                ],
                dtype=np.float32,
            )
            target_vel[:2] = (
                (1.0 - xy_mpc_blend) * target_vel[:2]
                + xy_mpc_blend * self.mpc.prev_velocity[:2]
            )
            target_vel[2] = float(self.mpc.prev_velocity[2])

        # Le contrôleur PID transforme la cible haut niveau en RPM moteurs
        rpm, _, _ = self.ctrl.computeControl(
            control_timestep=1.0 / self.control_freq_hz,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=target_pos,
            target_rpy=np.array([0.0, 0.0, target_yaw], dtype=np.float32),
            target_vel=target_vel,
        )

        # Le simulateur attend une commande moteur de forme (1,4)
        motor_action = rpm.reshape(1, 4)

        if self.wind_enabled:
            self.current_wind = self.wind_model.getWind(dt).astype(np.float32)

            wind_force = self.wind_force_scale * self.current_wind
            
            drone_pos = state[0:3]
            
            p.applyExternalForce(
               objectUniqueId=self.drone_id,
               linkIndex=-1,
               forceObj=wind_force.tolist(),
               posObj=drone_pos.tolist(),
               flags=p.WORLD_FRAME,
               physicsClientId=self.pyb_client,
            ) 
        else:
            self.current_wind = np.zeros(3, dtype=np.float32)
        # Avance la simulation avec la commande moteur calculée par le PID
        obs, _, _, _, _ = self.env.step(motor_action)
        self._update_gui_visual_helpers()

        # Conversion en observation RL
        rl_obs = self._get_obs_from_state(obs[0])

        # Décomposition de l’observation pour calculer la récompense
        x_rel, y_rel, z_rel, vx, vy, vz, roll, pitch, yaw, wx, wy, wz, wind_x, wind_y, wind_z = rl_obs[:15]

        contact_points = p.getContactPoints(
            bodyA=self.drone_id,
            bodyB=self.platform_id,
            physicsClientId=self.pyb_client,
        )
        has_contact = len(contact_points) > 0

        # Contact stable = contact réel + faibles erreurs/vitesses
        stable_contact_condition = (
            abs(x_rel) < self.stable_xy_threshold
            and abs(y_rel) < self.stable_xy_threshold
            and -0.03 <= z_rel < 0.08
            and abs(vx) < 0.20
            and abs(vy) < 0.20
            and abs(vz) < 0.18
            and abs(roll) < 0.28
            and abs(pitch) < 0.28
            and abs(yaw_error) < 0.30
        )

        if has_contact and stable_contact_condition:
            self.contact_stable_steps += 1
        else:
            self.contact_stable_steps = 0
        # Reward :
        # on pénalise :
        # - l’erreur horizontale x,y
        # - l’altitude z
        # - les vitesses
        # - l’inclinaison
        # - les vitesses angulaires
        # - les actions trop fortes
        # - les changements brusques d’action
        horizontal_error = float(np.sqrt(x_rel**2 + y_rel**2))
        vertical_error = float(abs(z_rel))
        landing_error = float(np.sqrt(x_rel**2 + y_rel**2 + 0.50 * z_rel**2))
        xy_speed = float(np.sqrt(vx**2 + vy**2))
        angular_speed = float(np.sqrt(wx**2 + wy**2 + wz**2))
        tilt = float(np.sqrt(roll**2 + pitch**2))
        action_scale = np.maximum(np.abs(self.command_low), np.abs(self.command_high))
        action_norm = float(np.linalg.norm(action / action_scale))
        action_smooth_norm = float(np.linalg.norm(action_delta / action_scale))

        progress_reward = 0.0
        if self.prev_landing_error is not None:
            progress_reward = 2.0 * (self.prev_landing_error - landing_error)
        self.prev_landing_error = landing_error

        reward = (
                2.5 * progress_reward
                + 2.10 * np.exp(-14.0 * horizontal_error)
                + 0.80 * np.exp(-6.0 * vertical_error)
                - 3.20 * horizontal_error
                - 0.90 * vertical_error
                - 0.25 * xy_speed
                - 0.40 * abs(vz)
                - 0.80 * tilt
                - 0.03 * angular_speed
                - 0.35 * action_norm
                - 0.60 * max(0.0, float(dvz_cmd))
                - 0.45 * action_smooth_norm
                - 0.10 * action_smooth_penalty
            )
        terminated = False

        if abs(x_rel) < 0.10 and abs(y_rel) < 0.10 and 0.0 <= z_rel < 0.20:
            reward += 0.50

        if abs(x_rel) < 0.05 and abs(y_rel) < 0.05 and 0.0 <= z_rel < 0.05:
            reward += 1.50

        if horizontal_error < self.precise_xy_threshold and 0.0 <= z_rel < 0.06:
            reward += 1.20

        if has_contact and stable_contact_condition:
            reward += 3.50
        elif has_contact:
            reward -= 3.00

        if abs(x_rel) < 0.06 and abs(y_rel) < 0.06 and 0.0 <= z_rel < 0.08 and not has_contact:
            reward -= 1.40

        sat_penalty = (
           float(abs(dvx_cmd) > 0.24)
           + float(abs(dvy_cmd) > 0.24)
           + float(dvz_cmd > 0.08)
           + float(dvz_cmd < -0.16)
        )
        reward -= 1.20 * sat_penalty
        if self.contact_stable_steps >= self.required_contact_steps:
           terminated = True
           reward += 10.0

        if abs(x_rel) > 1.0 or abs(y_rel) > 1.0:
           terminated = True
           reward -= 8.0

        if z_rel < -0.05:
           terminated = True
           reward -= 8.0

        if abs(roll) > 1.2 or abs(pitch) > 1.2:
           terminated = True
           reward -= 8.0

        reward = float(np.clip(reward, -10.0, 15.0))

        # Incrémente le nombre d’étapes
        self.current_step += 1

        # Fin d’épisode si le nombre max de pas est atteint
        truncated = self.current_step >= self.max_steps

        # Infos utiles pour debug et affichage dans le test PPO
        info = {
            "x_rel": float(x_rel),
            "y_rel": float(y_rel),
            "z_rel": float(z_rel),
            "vx": float(vx),
            "vy": float(vy),
            "vz": float(vz),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "wx": float(wx),
            "wy": float(wy),
            "wz": float(wz),
            "target_x": float(target_pos[0]),
            "target_y": float(target_pos[1]),
            "target_z": float(target_pos[2]),
            "target_x_rel": float(target_pos[0] - self.landing_target[0]),
            "target_y_rel": float(target_pos[1] - self.landing_target[1]),
            "target_z_rel": float(target_pos[2] - self.landing_target[2]),
            "target_yaw": float(target_yaw),
            "policy_action_dvx": float(self.last_policy_action[0]),
            "policy_action_dvy": float(self.last_policy_action[1]),
            "policy_action_dvz": float(self.last_policy_action[2]),
            "command_dvx": float(dvx_cmd),
            "command_dvy": float(dvy_cmd),
            "command_dvz": float(dvz_cmd),
            "target_vx": float(target_vel[0]),
            "target_vy": float(target_vel[1]),
            "target_vz": float(target_vel[2]),
            "has_contact": bool(has_contact),
            "contact_stable_steps": int(self.contact_stable_steps),
            "yaw_error": float(yaw_error),
            "episode_count": int(self.episode_count),
            "meta_mode": int(self.meta_model.mode),
            "platform_motion_mode": int(self.platform_motion_mode),
            "wind_enabled": bool(self.wind_enabled),
            "wind_x": float(self.current_wind[0]),
            "wind_y": float(self.current_wind[1]),
            "wind_z": float(self.current_wind[2]),
            "init_x": float(self.init_xyzs[0, 0]),
            "init_y": float(self.init_xyzs[0, 1]),
            "init_z": float(self.init_xyzs[0, 2]),
        }

        return rl_obs, float(reward), terminated, truncated, info

    def render(self):
        # Non utilisé ici, car PyBullet GUI gère déjà l’affichage
        pass

    def close(self):
        # Ferme proprement l’environnement
        if self.env is not None:
            self.env.close()
            self.env = None
