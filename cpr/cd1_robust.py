"""CD1: robuste, blinde G1-Lokomotion mit Treppen und Domain Randomization.

Baut auf Unitrees G1-Velocity-Task (Lauf A) und den bodenrelativen Termen aus
cpr/unitree_g1_perceptive.py (E-008, E-010) auf. Aenderungen gegenueber Lauf A:

1. Gelaende mit Treppen auf/ab, Kisten, rauem Boden und Rampen (10 Stufen,
   Gelaende-Curriculum). Stufenhoehe bis 18 cm (G1-tauglich, ROUGH_TERRAINS_CFG
   geht bis 23 cm).
2. Domain Randomization nach Vorlesung (RL-Folien S. 11):
   Reibung 0,2-1,25 | Rumpfmasse -1..+3 kg | Schwerpunkt +-5 cm |
   PD-Gains +-20 % | Stoesse bis 1 m/s alle 5-15 s | Sensorrauschen (wie A).
   Noch NICHT: Motorstaerke +-10 % und Aktionslatenz 0-20 ms (braucht eigenen
   Action-Term bzw. DelayedPD-Aktoren).
3. Actor bleibt BLIND (nur Propriozeption) -> genau diese Policy soll laut CD2
   auf den echten G1. Der Critic bekommt das Hoehenraster als privilegierte
   Information (Asymmetric Actor-Critic, Folie S. 12).
4. Befehls-Zeitplan (E-010) statt Unitrees leistungsabhaengigem Curriculum,
   damit die Roboter schnell genug laufen, um im Gelaende-Curriculum
   aufzusteigen (Aufstieg erst ab 4 m pro Episode).
5. Koerperhoehe, Abbruch und Fussfreiheit relativ zum Boden (E-008/E-010),
   sonst werden aufrechte Roboter auf Treppen bestraft bzw. abgebrochen.

Play-Konfiguration fuer das Video: nur Treppen (mittlere bis hohe Stufe),
konstanter Vorwaertsbefehl, keine Stoesse, Kamera folgt dem Roboter.
"""

import importlib

import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from cpr.unitree_g1_perceptive import (
    CriticCfg,
    _cmd_schedule,
    _history_per_term,
    foot_clearance_reward_rel,
    root_height_below_minimum_rel,
)

_unitree = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg")

STEP_HEIGHT = (0.02, 0.18)  # Stufe 0 -> Stufe max

STAIRS_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,  # Schwierigkeitsstufen
    num_cols=20,  # Gelaendetypen nebeneinander
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
        # Pyramide: Mitte oben -> Roboter laeuft vom Start aus Treppen RUNTER
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=STEP_HEIGHT,
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # Grube: Mitte unten -> Roboter laeuft vom Start aus Treppen HOCH
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=STEP_HEIGHT,
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.10, grid_width=0.45, grid_height_range=(0.02, 0.10), platform_width=2.0
        ),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.075, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.075, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
    },
)


def _terrain(cfg, generator=STAIRS_TERRAINS_CFG, max_init_level=3):
    cfg.scene.terrain.terrain_generator = generator.copy()
    cfg.scene.terrain.max_init_terrain_level = max_init_level


def _domain_randomization(cfg):
    ev = cfg.events
    # Dynamik
    ev.physics_material.params["static_friction_range"] = (0.2, 1.25)
    ev.physics_material.params["dynamic_friction_range"] = (0.2, 1.25)
    ev.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
    ev.base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    # Aktorik: PD-Gains +-20 % (G1 nutzt implizite Aktoren -> wirkt direkt in PhysX)
    ev.actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # Stoerungen: Stoesse bis 1 m/s, zufaellig alle 5-15 s
    ev.push_robot.interval_range_s = (5.0, 15.0)
    ev.push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
    # Sensorrauschen: bereits in Unitrees PolicyCfg (enable_corruption=True)


def _ground_relative(cfg):
    cfg.rewards.base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
    cfg.rewards.feet_clearance.func = foot_clearance_reward_rel
    cfg.terminations.base_height.func = root_height_below_minimum_rel


@configclass
class CD1RobustEnvCfg(_unitree.RobotEnvCfg):
    def __post_init__(self):
        _terrain(self)
        # Critic sieht das Hoehenraster (privilegiert), Actor bleibt blind
        self.observations.critic = CriticCfg()
        super().__post_init__()  # schaltet das Gelaende-Curriculum ein
        _history_per_term(self.observations.critic)
        _domain_randomization(self)
        _ground_relative(self)
        _cmd_schedule(self)


# --- Video ---------------------------------------------------------------------

VIDEO_TERRAINS_CFG = STAIRS_TERRAINS_CFG.replace(
    num_rows=3,
    num_cols=4,
    difficulty_range=(0.5, 0.9),  # Stufen ca. 10-16 cm
    sub_terrains={
        "stairs_down": STAIRS_TERRAINS_CFG.sub_terrains["stairs_down"],
        "stairs_up": STAIRS_TERRAINS_CFG.sub_terrains["stairs_up"],
    },
)


@configclass
class CD1RobustPlayEnvCfg(CD1RobustEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        _terrain(self, VIDEO_TERRAINS_CFG, max_init_level=None)
        self.scene.terrain.terrain_generator.curriculum = False
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        # Konstant geradeaus, kein Stehen, keine Stoesse
        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (0.6, 0.6)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.rel_standing_envs = 0.0
        cmd.resampling_time_range = (1e6, 1e6)
        self.events.push_robot = None
        # Kamera folgt Roboter 0. Direkt in der Klasse gesetzt (nicht per Hydra),
        # daher greift die Typpruefung aus Job 8380328 hier nicht.
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (2.5, 2.5, 1.6)
        self.viewer.lookat = (0.0, 0.0, 0.4)
