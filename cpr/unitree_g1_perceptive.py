"""T-004 Lauf B: Unitrees G1-Velocity-Beispiel mit Hoehenraster und unebenem Gelaende.

Abgeleitet von Unitrees `RobotEnvCfg` (unitree_rl_lab 4960b84,
tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py). Genau zwei Aenderungen
gegenueber Lauf A, entschieden in E-006 und T-004:

1. Gelaende: `ROUGH_TERRAINS_CFG` aus Isaac Lab statt Unitrees Ebene
   (Unitree :24-36, der Tausch steht dort als Kommentar an :48).
2. Beobachtung: das Hoehenraster (Unitrees Scanner, :68-75, 17 x 11 = 187 Punkte)
   fuer Policy und Critic, OHNE Historie ("1. keine Historie", T-004). Die
   uebrigen Terme behalten ihre 5 Zeitschritte. Isaac Lab ueberschreibt jede
   Term-Historie, sobald die Gruppe eine hat (observation_manager.py:550-552) -
   deshalb steht die Historie hier pro Term, die Gruppe hat keine.

Rewards bleiben unveraendert ("2. Rewards bleiben unverändert", T-004).

Zwei Abspiel-Konfigurationen fuer den Vergleich A gegen B im selben Gelaende:
`PerceptivePlayEnvCfg` (B) und `BlindRoughPlayEnvCfg` (A-Policy, B-Gelaende,
A-Beobachtungen - die Policy aus A bekommt genau die Eingaben, mit denen sie
trainiert wurde).

Erst nach dem App-Start importieren; registriert werden die Tasks in
unitree_train.py ueber Zeichenketten-Einstiegspunkte.
"""

import copy
import importlib

import torch

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp

# "29dof" ist kein gueltiger Python-Bezeichner, daher import_module
_unitree = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg")

HISTORY = 5  # Unitrees Gruppen-Historie (velocity_env_cfg.py:206, :233)
SCAN = {"sensor_cfg": SceneEntityCfg("height_scanner")}


# Das Raster als echtes Feld (haengt sich hinten an die Terme aus A an), damit es
# die dict-Umwandlung durch Hydra ueberlebt. Rauschen und Clip der Policy wie bei
# NVIDIA (velocity_env_cfg.py:134-139), der Critic wie Unitrees auskommentierter
# Term (:225-228).
@configclass
class PolicyCfg(_unitree.ObservationsCfg.PolicyCfg):
    height_scan = ObsTerm(
        func=mdp.height_scan, params=SCAN, noise=Unoise(n_min=-0.1, n_max=0.1), clip=(-1.0, 1.0)
    )


@configclass
class CriticCfg(_unitree.ObservationsCfg.CriticCfg):
    height_scan = ObsTerm(func=mdp.height_scan, params=SCAN, clip=(-1.0, 5.0))


def _history_per_term(group, skip=("height_scan",)):
    """Gruppen-Historie in die einzelnen Terme verschieben (Ergebnis identisch zu A)."""
    group.history_length = None
    for name, term in vars(group).items():
        if isinstance(term, ObsTerm) and name not in skip:
            term.history_length = HISTORY
            term.flatten_history_dim = True


def _rough_terrain(cfg):
    terrain = cfg.scene.terrain
    terrain.terrain_generator = copy.deepcopy(ROUGH_TERRAINS_CFG)
    # Unitree startet auf der hoechsten Stufe (:49), was auf der Ebene folgenlos
    # war. Mit echtem Curriculum nehmen wir NVIDIAs Wert
    # (isaaclab_tasks/.../velocity/velocity_env_cfg.py:48).
    terrain.max_init_terrain_level = 5


@configclass
class PerceptiveEnvCfg(_unitree.RobotEnvCfg):
    def __post_init__(self):
        _rough_terrain(self)
        self.observations.policy = PolicyCfg()
        self.observations.critic = CriticCfg()
        # Elternklasse schaltet das Gelaende-Curriculum ein, sobald der Generator steht
        super().__post_init__()
        _history_per_term(self.observations.policy)
        _history_per_term(self.observations.critic)


def _play(cfg):
    # wie Unitrees RobotPlayEnvCfg (:400-406), aber zufaellige Startstufe statt Curriculum
    cfg.scene.num_envs = 32
    gen = cfg.scene.terrain.terrain_generator
    gen.num_rows = 5
    gen.num_cols = 5
    gen.curriculum = False
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.commands.base_velocity.ranges = cfg.commands.base_velocity.limit_ranges


@configclass
class PerceptivePlayEnvCfg(PerceptiveEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)


# Lauf B mit bodenrelativer Koerperhoehe (E-008): B scheiterte (Job 9000707), weil
# Unitrees base_height (:296) gegen z = 0 der Welt misst - auf den Treppenfeldern
# steht ein richtig stehender Roboter 0,35 m und mehr darueber. Mit sensor_cfg
# misst Isaac Lab gegen den Mittelwert der Hoehenraster-Treffer
# (isaaclab/envs/mdp/rewards.py:115-118). Gewicht und Ziel 0,78 m bleiben;
# feet_clearance (absolute Fusshoehe) bleibt bewusst unveraendert - eine Aenderung.
@configclass
class PerceptiveRelHeightEnvCfg(PerceptiveEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")


@configclass
class PerceptiveRelHeightPlayEnvCfg(PerceptiveRelHeightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)


# --- E-010: Befehls-Zeitplan und bodenrelative Hoehen ---------------------------
#
# Unitrees Befehls-Curriculum erweitert nur, wenn der Geschwindigkeits-Reward ueber
# 80 % liegt (mdp/curriculums.py:20-30) - in 3000 Iterationen kamen A und B so nur
# auf 0,3 m/s, B blieb auf Gelaendestufe 0 (Aufstieg erst ab 4 m pro Episode,
# isaaclab_tasks .../velocity/mdp/curriculums.py:46-52). Stattdessen ein fester
# Zeitplan: linear von Unitrees Startspanne zur Endspanne (velocity_env_cfg.py:170-175).
# Wie bei Unitree nur vorwaerts/seitlich; die Drehspanne bleibt.

CMD_START = (-0.1, 0.1)  # Unitrees Startspanne fuer lin_vel_x und lin_vel_y
STEPS_PER_ITER = 24  # num_steps_per_env, agents/rsl_rl_ppo_cfg.py:12
CMD_FULL_AT_ITER = 1000  # E-010: volle Spanne ab Iteration 1000


def lin_vel_cmd_schedule(env, env_ids, full_at_step: int, start=CMD_START):
    """Befehlsspanne nach Schrittzahl statt nach Leistung (Ersatz fuer lin_vel_cmd_levels)."""
    cfg = env.command_manager.get_term("base_velocity").cfg
    frac = min(env.common_step_counter / full_at_step, 1.0)
    for axis in ("lin_vel_x", "lin_vel_y"):
        lo, hi = getattr(cfg.limit_ranges, axis)
        setattr(cfg.ranges, axis, (start[0] + frac * (lo - start[0]), start[1] + frac * (hi - start[1])))
    # gleicher Log-Name wie bei Unitree (Curriculum/lin_vel_cmd_levels): obere Grenze vorwaerts
    return torch.tensor(cfg.ranges.lin_vel_x[1], device=env.device)


def _cmd_schedule(cfg):
    cfg.curriculum.lin_vel_cmd_levels = CurrTerm(
        func=lin_vel_cmd_schedule, params={"full_at_step": CMD_FULL_AT_ITER * STEPS_PER_ITER}
    )


def _ground_z(env, sensor_cfg):
    """Mittlere Bodenhoehe unter dem Hoehenraster, wie Isaac Labs base_height_l2 (rewards.py:115-118)."""
    return torch.mean(env.scene[sensor_cfg.name].data.ray_hits_w[..., 2], dim=1)


def root_height_below_minimum_rel(env, minimum_height: float, asset_cfg=SceneEntityCfg("robot"), sensor_cfg=SCAN["sensor_cfg"]):
    """Unitrees Abbruch (velocity_env_cfg.py:345) gegen den Boden statt gegen z = 0.

    Absolut wurden ab Stufe 2 aufrechte Roboter in Treppengruben sofort abgebrochen
    (Grubenmitte bei -7 x Stufenhoehe, mesh_terrains.py:246; Log 9000707: 15-23 %).
    """
    root_z = env.scene[asset_cfg.name].data.root_pos_w[:, 2]
    return root_z - _ground_z(env, sensor_cfg) < minimum_height


def foot_clearance_reward_rel(env, asset_cfg, target_height: float, std: float, tanh_mult: float, sensor_cfg=SCAN["sensor_cfg"]):
    """Unitrees foot_clearance_reward (mdp/rewards.py:120-128), Fusshoehe ueber dem Boden direkt darunter.

    Bodenhoehe je Fuss = naechster Rasterpunkt (0,1 m Abstand) in der Ebene. Auf
    ebenem Boden bei z = 0 identisch zu Unitree.
    """
    asset = env.scene[asset_cfg.name]
    hits = torch.nan_to_num(env.scene[sensor_cfg.name].data.ray_hits_w, posinf=1e6, neginf=-1e6)
    feet = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    nearest = torch.cdist(feet[..., :2], hits[..., :2]).argmin(dim=2)
    ground = torch.gather(hits[..., 2], 1, nearest)
    foot_z_target_error = torch.square(feet[..., 2] - ground - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    return torch.exp(-torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1) / std)


# Lauf A mit Zeitplan E (auf der Ebene aendern bodenrelative Hoehen nichts)
@configclass
class CmdScheduleEnvCfg(_unitree.RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _cmd_schedule(self)


# Lauf B mit Zeitplan E, Abbruch und Fussfreiheit bodenrelativ (base_height schon seit E-008)
@configclass
class PerceptiveCmdScheduleEnvCfg(PerceptiveRelHeightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _cmd_schedule(self)
        self.terminations.base_height.func = root_height_below_minimum_rel
        self.rewards.feet_clearance.func = foot_clearance_reward_rel


@configclass
class PerceptiveCmdSchedulePlayEnvCfg(PerceptiveCmdScheduleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
        # der Zeitplan setzte die Befehle sonst ab Schritt 0 wieder auf die Startspanne
        self.curriculum.lin_vel_cmd_levels = None


@configclass
class BlindRoughPlayEnvCfg(_unitree.RobotEnvCfg):
    def __post_init__(self):
        _rough_terrain(self)
        super().__post_init__()
        _play(self)
