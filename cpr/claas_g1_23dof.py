"""Unitree G1 mit 23 DoF (Roboter der RobotChallenge) - Claas, 18.11.

Hinweis Coach (17.11.): In der RC wird der G1 mit 23 DoF genutzt, nicht 29.
23 DoF = 6 pro Bein, 1 Taille (waist_yaw), 5 pro Arm (Schulter pitch/roll/yaw, Ellbogen,
wrist_roll). Gegenueber 29 DoF fehlen waist_roll/pitch und wrist_pitch/yaw.

Modell: offizielle URDF g1_23dof_rev_1_0.urdf aus unitree_ros (claas/ext/unitree_ros,
geklont 18.11.). Isaac Lab wandelt sie beim ersten Start in USD um und legt das Ergebnis
in claas/ext/usd_g1_23dof ab (spaetere Jobs nutzen es wieder).

Aktoren: wie Unitrees UNITREE_G1_29DOF_CFG (assets/robots/unitree.py), das unsere
29-DoF-Baseline erfolgreich trainiert hat - nur die Gelenke, die es bei 23 DoF gibt.
Gleiche Motortypen: 7520-14 (Huefte pitch/yaw, Taille yaw), 7520-22 (Huefte roll, Knie),
5020 (Schultern, Ellbogen, Handgelenk roll, Knoechel).

Trainings-Konfigurationen nutzen die Erkenntnisse aus dem 29-DoF-Meilenstein
(cpr/claas_milestone.py): adaptive DR ueber die Sturzrate (V2), adaptive Befehle (70 %),
milder Gelaende-Abstieg; Tracking-Gewicht 2,0 (Reward-Fix aus B2/B5).
"""

import importlib

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UnitreeArticulationCfg, UnitreeUrdfFileCfg

from cpr import claas_milestone as ms

def _urdf_importer_offline():
    """Isaac Lab 2.3 verlangt exakt isaacsim.asset.importer.urdf-2.4.31 und will ihn aus dem
    Internet laden; im Container (offline) liegt 2.4.30. Wir aktivieren die lokale Version
    und ueberspringen den Versions-Wechsel im UrdfConverter (Fehler Smoke 9378180, 18.11.)."""
    try:
        import omni.kit.app
        import isaaclab.sim.converters.urdf_converter as uc

        mgr = omni.kit.app.get_app().get_extension_manager()
        if not mgr.is_extension_enabled("isaacsim.asset.importer.urdf"):
            mgr.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
        uc.get_isaac_sim_version = lambda: uc.Version("5.0")

        # 2.4.30 kennt nicht alle Setter, die Isaac Lab 2.3 aufruft (z. B.
        # set_merge_fixed_ignore_inertia, Fehler Smoke 9378612). Fehlende Setter ueberspringen.
        import inspect
        import re
        import textwrap

        src = textwrap.dedent(inspect.getsource(uc.UrdfConverter._get_urdf_import_config))
        src = re.sub(r"\)\s*->[^\n]*:\n", "):\n", src, count=1)
        src = re.sub(r"import_config\.(set_\w+)\(", r'_claas_setter(import_config, "\1")(', src)

        def _claas_setter(obj, name):
            if hasattr(obj, name):
                return getattr(obj, name)
            print(f"[claas_g1_23dof] {name} fehlt im URDF-Importer 2.4.30 - uebersprungen")
            return lambda *a, **k: None

        glb = dict(uc.__dict__)
        glb["_claas_setter"] = _claas_setter
        ns = {}
        exec(src, glb, ns)
        uc.UrdfConverter._get_urdf_import_config = ns["_get_urdf_import_config"]
        print("[claas_g1_23dof] lokaler URDF-Importer aktiv:", mgr.is_extension_enabled("isaacsim.asset.importer.urdf"))
    except Exception as e:  # noqa: BLE001
        print("[claas_g1_23dof] URDF-Importer konnte nicht aktiviert werden:", repr(e))


_urdf_importer_offline()


def _std_absichern():
    """rsl_rl (noise_std_type="scalar"): Die Rausch-Standardabweichung ist ein freier Parameter und
    kann nach einem Optimierungsschritt negativ/NaN werden -> "normal expects all elements of std
    >= 0.0" (Abstuerze B2, B5, A3, T1 am 18./19.11.). Vor jeder Verteilungsberechnung auf >= 0,02
    begrenzen; Checkpoint-Format bleibt gleich (Weitertrainieren und Export unveraendert)."""
    try:
        import torch
        from rsl_rl.modules import actor_critic as ac

        orig = ac.ActorCritic._update_distribution
        if getattr(orig, "_claas_safe", False):
            return

        def _safe(self, obs):
            if getattr(self, "noise_std_type", "scalar") == "scalar" and not getattr(self, "state_dependent_std", False):
                std = getattr(self, "std", None)
                if std is not None:
                    with torch.no_grad():
                        std.data = torch.nan_to_num(std.data, nan=0.1, posinf=1.0, neginf=0.02).clamp_(0.02, 2.0)
            return orig(self, obs)

        _safe._claas_safe = True
        ac.ActorCritic._update_distribution = _safe
        print("[claas_g1_23dof] Policy-Std wird auf [0,02; 2,0] begrenzt")
    except Exception as e:  # noqa: BLE001
        print("[claas_g1_23dof] Std-Absicherung nicht aktiv:", repr(e))


_std_absichern()

_unitree = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg")

EXT = "/bigwork/nhkbrc2a/isaac/claas/ext"
URDF = f"{EXT}/unitree_ros/robots/g1_description/g1_23dof_rev_1_0.urdf"

G1_23DOF_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=URDF,
        usd_dir=f"{EXT}/usd_g1_23dof",
        usd_file_name="g1_23dof_rev_1_0.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            stiffness={".*_hip_.*": 100.0, "waist_yaw_joint": 200.0},
            damping={".*_hip_.*": 2.0, "waist_yaw_joint": 5.0},
            armature=0.01,
        ),
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
            effort_limit_sim=139,
            velocity_limit_sim=20.0,
            stiffness={".*_hip_roll_.*": 100.0, ".*_knee_.*": 150.0},
            damping={".*_hip_roll_.*": 2.0, ".*_knee_.*": 4.0},
            armature=0.01,
        ),
        "N5020-16": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_roll.*", ".*_ankle_.*"],
            effort_limit_sim=25,
            velocity_limit_sim=37,
            stiffness=40.0,
            damping={
                ".*_shoulder_.*": 1.0,
                ".*_elbow_.*": 1.0,
                ".*_wrist_roll.*": 1.0,
                ".*_ankle_.*": 2.0,
            },
            armature=0.01,
        ),
    },
    # Motor-Reihenfolge im Unitree-SDK (29 Plaetze, beim 23-DoF-Modell bleiben
    # waist_roll/pitch und wrist_pitch/yaw leer) - wird fuer den Export/Deploy gebraucht.
    joint_sdk_names=[
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "", "",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "", "",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "", "",
    ],
)


def _robot23(cfg):
    cfg.scene.robot = G1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _scratch_cmd(cfg):
    """Training ab null: Befehle starten klein wie bei Unitree (+-0,1 m/s) und wachsen adaptiv."""
    cmd = cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (-0.1, 0.1)
    cmd.ranges.lin_vel_y = (-0.1, 0.1)


# --- Training ----------------------------------------------------------------------


@configclass
class G23FlatEnvCfg(_unitree.RobotEnvCfg):
    """A1: eben, ab null, adaptive DR/Befehle V2, Unitree-Rewards."""

    def __post_init__(self):
        super().__post_init__()
        _robot23(self)
        ms._common(self)
        ms._v2(self)
        _scratch_cmd(self)


@configclass
class G23FlatTrackEnvCfg(G23FlatEnvCfg):
    """A2: wie A1, Tracking-Gewicht 2,0."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy.weight = 2.0


@configclass
class G23StairsTrackEnvCfg(ms.MsStairsV2TrackEnvCfg):
    """A3: Treppen-Mix ab null (Stufe 0), Curricula V2, Tracking-Gewicht 2,0."""

    def __post_init__(self):
        super().__post_init__()
        _robot23(self)
        _scratch_cmd(self)


@configclass
class G23StairsEnvCfg(ms.MsStairsV2EnvCfg):
    """Treppen, Curricula V2, Unitree-Rewards (fuer das Weitertrainieren ab A1)."""

    def __post_init__(self):
        super().__post_init__()
        _robot23(self)


# --- Videos ------------------------------------------------------------------------


@configclass
class G23PlayStairsEasyCfg(ms.MsPlayStairsEasyCfg):
    def __post_init__(self):
        super().__post_init__()
        _robot23(self)


@configclass
class G23PlayStairsMidCfg(ms.MsPlayStairsMidCfg):
    def __post_init__(self):
        super().__post_init__()
        _robot23(self)


@configclass
class G23PlayFlatPushCfg(ms.MsPlayFlatPushCfg):
    def __post_init__(self):
        super().__post_init__()
        _robot23(self)


# --- A4: robuster fuer den echten Roboter (19.11. abends) ---------------------------
# Drei typische Sim-to-Real-Luecken, die A2 noch nicht gesehen hat:
#  1. Aktionslatenz 0-20 ms: neuer Sollwert wirkt erst nach 0-4 Physikschritten (je 5 ms)
#  2. Motorstaerke +-10 %: Drehmomentgrenzen pro Episode zufaellig skaliert
#  3. Drehbefehle bis +-1 rad/s (A2 kannte nur +-0,1..0,2 -> drehte in MuJoCo kaum)
import torch  # noqa: E402

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg  # noqa: E402
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction  # noqa: E402
from isaaclab.managers import CurriculumTermCfg as CurrTerm  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402


class DelayedJointPositionAction(JointPositionAction):
    """JointPositionAction mit zufaelliger Verzoegerung (in Physikschritten) pro Umgebung."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._prev = self._target_default()
        self._delay = torch.randint(cfg.delay_min, cfg.delay_max + 1, (self.num_envs,), device=self.device)
        self._sub = 0

    def _target_default(self):
        off = self._offset
        if isinstance(off, torch.Tensor):
            return off.expand(self.num_envs, -1).clone() if off.dim() == 2 else off.repeat(self.num_envs, 1)
        return torch.full((self.num_envs, self.action_dim), float(off), device=self.device)

    def process_actions(self, actions):
        self._prev = self._processed_actions.clone()
        super().process_actions(actions)
        self._sub = 0

    def apply_actions(self):
        new = (self._sub >= self._delay).unsqueeze(1)
        target = torch.where(new, self._processed_actions, self._prev)
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)
        self._sub += 1

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        default = self._target_default()
        self._prev[env_ids] = default[env_ids]
        n = self._delay[env_ids].shape[0]
        self._delay[env_ids] = torch.randint(self.cfg.delay_min, self.cfg.delay_max + 1, (n,), device=self.device)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = DelayedJointPositionAction
    delay_min: int = 0
    delay_max: int = 4  # 4 x 5 ms = 20 ms


def randomize_motor_strength(env, env_ids, scale_range=(0.9, 1.1), asset_cfg=SceneEntityCfg("robot")):
    """Drehmomentgrenzen pro Episode mit Faktor aus scale_range skalieren."""
    asset = env.scene[asset_cfg.name]
    if not hasattr(env, "_claas_effort0"):
        env._claas_effort0 = asset.data.joint_effort_limits.clone()
        print("[claas_g1_23dof] Motorstaerke-Randomisierung aktiv", tuple(scale_range))
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    s = torch.empty((len(env_ids), 1), device=env.device).uniform_(*scale_range)
    asset.write_joint_effort_limit_to_sim(env._claas_effort0[env_ids] * s, env_ids=env_ids)


def cmd_adaptive_ang(env, env_ids, reward_term_name="track_ang_vel_z", frac=0.5, step=0.1):
    """Drehbefehle wachsen wie die linearen (ms.lin_vel_cmd_adaptive) bis limit_ranges."""
    cmd = env.command_manager.get_term("base_velocity").cfg
    w = env.reward_manager.get_term_cfg(reward_term_name).weight
    rew = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    if env.common_step_counter % env.max_episode_length == 0 and rew > w * frac:
        lo, hi = cmd.ranges.ang_vel_z
        llo, lhi = cmd.limit_ranges.ang_vel_z
        cmd.ranges.ang_vel_z = (max(lo - step, llo), min(hi + step, lhi))
    return torch.tensor(cmd.ranges.ang_vel_z[1], device=env.device)


def _robust(cfg):
    old = cfg.actions.JointPositionAction
    cfg.actions.JointPositionAction = DelayedJointPositionActionCfg(
        asset_name=old.asset_name,
        joint_names=old.joint_names,
        scale=old.scale,
        use_default_offset=old.use_default_offset,
        preserve_order=old.preserve_order,
        clip=old.clip,
    )
    cfg.events.motor_strength = EventTerm(func=randomize_motor_strength, mode="reset", params={"scale_range": (0.9, 1.1)})
    cmd = cfg.commands.base_velocity
    # ab A2 weiter: lineare Befehle gleich auf vollem Bereich, Drehen waechst von +-0,2 auf +-1,0
    cmd.ranges.lin_vel_x = cmd.limit_ranges.lin_vel_x
    cmd.ranges.lin_vel_y = cmd.limit_ranges.lin_vel_y
    cmd.ranges.ang_vel_z = (-0.2, 0.2)
    cmd.limit_ranges.ang_vel_z = (-1.0, 1.0)
    cfg.curriculum.ang_vel_cmd_levels = CurrTerm(func=cmd_adaptive_ang)


@configclass
class G23RobustEnvCfg(G23FlatTrackEnvCfg):
    """A4: A2-Rezept + Latenz 0-20 ms + Motorstaerke +-10 % + Drehen bis +-1 rad/s (Start: A2)."""

    def __post_init__(self):
        super().__post_init__()
        _robust(self)


@configclass
class G23PlayRobustCfg(G23PlayFlatPushCfg):
    """Video A4: eben mit Stoessen, Latenz aktiv, Befehl vorwaerts + Drehen."""

    def __post_init__(self):
        super().__post_init__()
        _robust(self)
        self.curriculum.ang_vel_cmd_levels = None
        c = self.commands.base_velocity.ranges
        c.lin_vel_x, c.lin_vel_y, c.ang_vel_z = (0.5, 0.5), (0.0, 0.0), (0.5, 0.5)


# --- A5 / T4: Drehen lernen (20.11.) ------------------------------------------------
# A4 hat Drehen nicht gelernt: Dreh-Reward 0,5 gegen 2,0 fuer Geschwindigkeit, Curriculum
# blieb bei +-0,5 rad/s haengen. Jetzt Dreh-Gewicht 1,5 und niedrigere Aufstiegsschwelle.
def _turn(cfg, weight=1.5, frac=0.4, start=0.5):
    cfg.rewards.track_ang_vel_z.weight = weight
    cfg.commands.base_velocity.ranges.ang_vel_z = (-start, start)
    cfg.commands.base_velocity.limit_ranges.ang_vel_z = (-1.0, 1.0)
    cfg.curriculum.ang_vel_cmd_levels = CurrTerm(func=cmd_adaptive_ang, params={"frac": frac})


@configclass
class G23RobustTurnEnvCfg(G23RobustEnvCfg):
    """A5: A4 + Dreh-Gewicht 1,5 (Start: A4)."""

    def __post_init__(self):
        super().__post_init__()
        _turn(self)


@configclass
class G23StairsRobustEnvCfg(G23StairsTrackEnvCfg):
    """T4: blinde Treppen (T3b-Rezept) + Latenz + Motorstaerke + Drehen (Start: T3b ck6500)."""

    def __post_init__(self):
        super().__post_init__()
        _robust(self)
        _turn(self, start=0.2)


@configclass
class G23PlayTurnCfg(G23PlayRobustCfg):
    """Video: langsam vorwaerts und deutlich drehen (Latenz aktiv)."""

    def __post_init__(self):
        super().__post_init__()
        c = self.commands.base_velocity.ranges
        c.lin_vel_x, c.lin_vel_y, c.ang_vel_z = (0.3, 0.3), (0.0, 0.0), (0.8, 0.8)


# --- A6: menschlicherer Gang (20.11.) --------------------------------------------
# Beobachtung im Video: kleine, schlurfende Schritte, kippelig. Ursache im Reward-Verhaeltnis:
# Geschwindigkeit folgen (2,0) ist viel wichtiger als Gangbild (gait 0,5, Fussabstand 1,0) -
# Schlurfen erfuellt die Geschwindigkeit ebenso und ist "billiger" (action_rate-Strafe).
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as _vmdp  # noqa: E402
from isaaclab.managers import RewardTermCfg as _RewTerm  # noqa: E402


def _gait(cfg):
    r = cfg.rewards
    r.gait.weight = 1.5  # abwechselnder Bodenkontakt im 0,8-s-Takt (Unitree feet_gait)
    r.feet_clearance.weight = 2.0  # Fuss in der Schwungphase ~10 cm anheben
    r.feet_slide.weight = -0.5  # Schleifen am Boden staerker bestrafen
    r.joint_deviation_arms.weight = -0.05  # Armschwung zulassen
    r.feet_air_time = _RewTerm(  # laengere Schwungphasen -> laengere Schritte
        func=_vmdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.4,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )


@configclass
class G23GaitEnvCfg(G23RobustTurnEnvCfg):
    """A6: wie A5 (robust + Drehen) + Gangbild-Rewards (Start: A4)."""

    def __post_init__(self):
        super().__post_init__()
        _gait(self)


# =====================================================================================
# 22.11. (Claas): Robustheitstest, A7 (Symmetrie), A8 (unebener Boden)
# =====================================================================================
import isaaclab.terrains as _terr  # noqa: E402

_FLAGS = {"eval": False, "symmetry": False}


def _patch_runner():
    """Schalter fuer rsl_rl, gesetzt von den Env-Konfigurationen (Env wird vor dem Runner gebaut):
    - eval: PPO.update ueberspringen -> Policy bleibt unveraendert, die Trainingsschleife misst nur
    - symmetry: Daten-Augmentierung links/rechts (rsl_rl symmetry_cfg) einschalten"""
    try:
        from rsl_rl.algorithms import PPO
        from rsl_rl.runners import OnPolicyRunner

        if getattr(PPO.update, "_claas", False):
            return
        orig_update = PPO.update

        def update(self, *a, **k):
            if _FLAGS["eval"]:
                self.storage.clear()
                return {"value_function": 0.0, "surrogate": 0.0, "entropy": 0.0}
            return orig_update(self, *a, **k)

        update._claas = True
        PPO.update = update

        orig_init = OnPolicyRunner.__init__

        def init(self, env, train_cfg, *a, **k):
            if _FLAGS["symmetry"]:
                train_cfg["algorithm"]["symmetry_cfg"] = {
                    "use_data_augmentation": True,
                    "use_mirror_loss": False,
                    "mirror_loss_coeff": 0.0,
                    "data_augmentation_func": mirror_augmentation,
                }
                print("[claas_g1_23dof] Symmetrie-Augmentierung aktiv")
            if _FLAGS["eval"]:
                print("[claas_g1_23dof] EVAL: keine Policy-Updates")
            return orig_init(self, env, train_cfg, *a, **k)

        OnPolicyRunner.__init__ = init
    except Exception as e:  # noqa: BLE001
        print("[claas_g1_23dof] Runner-Patch nicht aktiv:", repr(e))


_patch_runner()

# Gelenke in Isaac-Reihenfolge (aus deploy.yaml joint_ids_map):
# 0 L_hip_pitch 1 R_hip_pitch 2 waist_yaw 3 L_hip_roll 4 R_hip_roll 5 L_sh_pitch 6 R_sh_pitch
# 7 L_hip_yaw 8 R_hip_yaw 9 L_sh_roll 10 R_sh_roll 11 L_knee 12 R_knee 13 L_sh_yaw 14 R_sh_yaw
# 15 L_ank_pitch 16 R_ank_pitch 17 L_elbow 18 R_elbow 19 L_ank_roll 20 R_ank_roll
# 21 L_wrist_roll 22 R_wrist_roll
_PERM = [1, 0, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21]
_SIGN = [1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]
_H = 5  # Historie
# Spiegelung an der x-z-Ebene: Vektor (x,-y,z), Pseudovektor Drehrate (-x,y,-z), Befehl (vx,-vy,-wz)
_VEC = {"lin": [1, -1, 1], "ang": [-1, 1, -1], "grav": [1, -1, 1], "cmd": [1, -1, -1]}
_LAYOUT = {
    "policy": ["ang", "grav", "cmd", "joint", "joint", "joint"],
    "critic": ["lin", "ang", "grav", "cmd", "joint", "joint", "joint"],
}


def _mirror_block(x, kind):
    import torch

    n = x.shape[0]
    if kind == "joint":
        y = x.view(n, _H, 23)[:, :, _PERM] * torch.tensor(_SIGN, device=x.device, dtype=x.dtype)
    else:
        y = x.view(n, _H, 3) * torch.tensor(_VEC[kind], device=x.device, dtype=x.dtype)
    return y.reshape(n, -1)


def _mirror_obs(x, group):
    parts, i = [], 0
    for kind in _LAYOUT[group]:
        d = _H * (23 if kind == "joint" else 3)
        parts.append(_mirror_block(x[:, i : i + d], kind))
        i += d
    assert i == x.shape[1], f"Symmetrie: {group} hat {x.shape[1]} Werte, erwartet {i}"
    import torch

    return torch.cat(parts, dim=1)


def mirror_augmentation(obs=None, actions=None, env=None):
    """rsl_rl-Symmetrie: liefert (Original + gespiegelt) fuer Beobachtungen und Aktionen."""
    import torch

    obs_out = act_out = None
    if obs is not None:
        mir = obs.clone()
        for g in obs.keys():
            if g in _LAYOUT:
                mir[g] = _mirror_obs(obs[g], g)
        obs_out = torch.cat([obs, mir], dim=0)
    if actions is not None:
        sign = torch.tensor(_SIGN, device=actions.device, dtype=actions.dtype)
        act_out = torch.cat([actions, actions[:, _PERM] * sign], dim=0)
    return obs_out, act_out


# --- Robustheitstest von A5 (keine Updates, nur messen) ----------------------------


def _eval_base(cfg, friction=(0.3, 1.25), push=1.0, push_int=(5.0, 12.0), delay=(0, 4), mass=(-1.0, 3.0),
               motor=(0.9, 1.1), gains=0.2):
    _FLAGS["eval"] = True
    c = cfg.curriculum
    for k in ("dr_level", "lin_vel_cmd_levels", "ang_vel_cmd_levels", "terrain_levels"):
        if hasattr(c, k):
            setattr(c, k, None)
    cmd = cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = cmd.limit_ranges.lin_vel_x
    cmd.ranges.lin_vel_y = cmd.limit_ranges.lin_vel_y
    cmd.ranges.ang_vel_z = (-1.0, 1.0)
    ev = cfg.events
    ev.physics_material.params["static_friction_range"] = friction
    ev.physics_material.params["dynamic_friction_range"] = (friction[0] * 0.9, friction[1] * 0.9)
    ev.push_robot.params["velocity_range"] = {"x": (-push, push), "y": (-push, push)}
    ev.push_robot.interval_range_s = push_int
    ev.add_base_mass.params["mass_distribution_params"] = mass
    ev.add_base_mass.params["operation"] = "add"
    ev.motor_strength.params["scale_range"] = motor
    ev.actuator_gains.params["stiffness_distribution_params"] = (1 - gains, 1 + gains)
    ev.actuator_gains.params["damping_distribution_params"] = (1 - gains, 1 + gains)
    cfg.actions.JointPositionAction.delay_min, cfg.actions.JointPositionAction.delay_max = delay


def _eval_cls(name, **kw):
    @configclass
    class _C(G23RobustTurnEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            _eval_base(self, **kw)

    _C.__name__ = _C.__qualname__ = name
    return _C


EVAL_SZENARIEN = {
    "E0_nominal": {},
    "E1_glatt": {"friction": (0.15, 0.3)},
    "E2_stoesse": {"push": 1.5, "push_int": (2.0, 4.0)},
    "E3_latenz": {"delay": (6, 8)},
    "E4_last": {"mass": (4.0, 6.0)},
    "E5_motor": {"motor": (0.7, 0.8)},
    "E6_kombi": {"friction": (0.3, 0.5), "push": 1.2, "push_int": (3.0, 6.0), "delay": (5, 6),
                 "mass": (2.0, 3.0), "motor": (0.85, 0.9)},
}
for _n, _kw in EVAL_SZENARIEN.items():
    globals()["G23Eval_" + _n] = _eval_cls("G23Eval_" + _n, **_kw)


# --- A7: A5 + Symmetrie + leicht verstaerktes Gangbild ---------------------------------


@configclass
class G23SymEnvCfg(G23RobustTurnEnvCfg):
    """A7: A5 + Links/Rechts-Symmetrie (Daten-Augmentierung) + Gangbild moderat (Start: A5)."""

    def __post_init__(self):
        super().__post_init__()
        _FLAGS["symmetry"] = True
        r = self.rewards
        r.gait.weight = 1.0
        r.feet_clearance.weight = 1.5
        r.feet_slide.weight = -0.3


@configclass
class G23SymOnlyEnvCfg(G23RobustTurnEnvCfg):
    """A7b (22.11.): nur Symmetrie, Belohnungen wie A5 (Ablation: liegt der Drehfehler von A7 an Symmetrie oder Gangbild?)."""

    def __post_init__(self):
        super().__post_init__()
        _FLAGS["symmetry"] = True


@configclass
class G23GaitModEnvCfg(G23RobustTurnEnvCfg):
    """A7c (22.11.): Gangbild moderat wie A7, aber ohne Symmetrie (Gegenprobe)."""

    def __post_init__(self):
        super().__post_init__()
        r = self.rewards
        r.gait.weight = 1.0
        r.feet_clearance.weight = 1.5
        r.feet_slide.weight = -0.3


@configclass
class G23SlipEnvCfg(G23RobustTurnEnvCfg):
    """A9 (23.11.): A5 + glatterer Boden im Training (Reibung 0,15-1,25 statt 0,3-1,25), Start: A5."""

    def __post_init__(self):
        super().__post_init__()
        ev = self.events
        ev.physics_material.params["static_friction_range"] = (0.15, 1.25)
        ev.physics_material.params["dynamic_friction_range"] = (0.12, 1.2)


# --- A8: A5 auf leicht unebenem Boden (keine Treppen) ---------------------------------

ROUGH_LIGHT_CFG = ms.STAIRS_TERRAINS_CFG.copy()
ROUGH_LIGHT_CFG.sub_terrains = {
    "flat": _terr.MeshPlaneTerrainCfg(proportion=0.15),
    "rough": _terr.HfRandomUniformTerrainCfg(proportion=0.3, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25),
    "slope_up": _terr.HfPyramidSlopedTerrainCfg(proportion=0.15, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25),
    "slope_down": _terr.HfInvertedPyramidSlopedTerrainCfg(proportion=0.15, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25),
    "grid": _terr.MeshRandomGridTerrainCfg(proportion=0.25, grid_width=0.45, grid_height_range=(0.01, 0.05), platform_width=2.0),
}


def _rough(cfg):
    ms._stairs(cfg)
    cfg.scene.terrain.terrain_generator = ROUGH_LIGHT_CFG.copy()


@configclass
class G23RoughEnvCfg(G23RobustTurnEnvCfg):
    """A8: A5 + unebener Boden (Rauheit bis 6 cm, Rampen bis 0,3 rad, Kanten bis 5 cm), Start: A5."""

    def __post_init__(self):
        super().__post_init__()
        _rough(self)


@configclass
class G23PlayRoughCfg(G23PlayTurnCfg):
    def __post_init__(self):
        super().__post_init__()
        g = ROUGH_LIGHT_CFG.copy()
        g.curriculum = False
        g.difficulty_range = (0.5, 0.9)
        g.num_rows, g.num_cols = 3, 3
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = g
        c = self.commands.base_velocity.ranges
        c.lin_vel_x, c.ang_vel_z = (0.5, 0.5), (0.0, 0.0)


# --- 24.11. Nacht: A10 (A8 + Treppen) und A11 (A8 + mehr glatter Boden) -------------------


@configclass
class G23StairsA10EnvCfg(G23StairsTrackEnvCfg):
    """A10: blinde Treppen-Mischung (wie T3/T4) + Robustheit + Drehen wie A5/A8, Start: A8.
    Ziel: MuJoCo-Treppe 8 cm (A8 schafft 5 cm)."""

    def __post_init__(self):
        super().__post_init__()
        _robust(self)
        _turn(self)


@configclass
class G23RoughSlipEnvCfg(G23RoughEnvCfg):
    """A11: A8 + deutlich mehr glatter Boden (Reibung 0,1-0,9 statt 0,3-1,25), Start: A8.
    A9 (0,15-1,25) hatte zu wenig glatte Roboter (~13 % unter 0,3) und wurde nicht besser."""

    def __post_init__(self):
        super().__post_init__()
        ev = self.events
        ev.physics_material.params["static_friction_range"] = (0.1, 0.9)
        ev.physics_material.params["dynamic_friction_range"] = (0.08, 0.85)
