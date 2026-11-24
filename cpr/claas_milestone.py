"""Meilenstein 18./19.11. (Claas): stufenweises, adaptives Training ab Baseline 3000.

Aufbau (siehe Notion, Experimente "Meilenstein ..."):
  Stufe 1 (parallel, ab Baseline 3000 = Job 9337470, model_2999.pt)
    B1  Treppen-Mix, adaptive Curricula                    -> MsStairsEnvCfg
    B2  wie B1, Tracking-Gewicht 2,0 statt 1,0 (Reward-Fix) -> MsStairsTrackEnvCfg
    B3  eben, adaptive Domain Randomization (Absicherung)   -> MsFlatEnvCfg
  Stufe 2  Auswahl des besten Laufs, dann Lauf C ab dessen letztem Checkpoint.

Lehren aus Lauf A (cd1_robust.py, Job 9356805):
  - nicht alles ab Iteration 0: Gelaende startet auf Stufe 0, Stoesse/PD-Streuung
    wachsen erst, wenn die Roboter lange genug stehen bleiben (adaptiv)
  - Befehle wachsen adaptiv (Schwelle 50 % statt Unitrees 80 %), Start bei dem,
    was die Baseline kann (bis 0,5 m/s vorwaerts)
  - Gelaende-Aufstieg schon ab 2,4 m statt 4 m Laufstrecke (humanoid-taugliche
    Geschwindigkeiten; Isaac Labs Regel ist fuer Vierbeiner ausgelegt)

Kompatibilitaet: Actor UND Critic haben dieselben Eingaenge wie die Baseline
(blind, Unitrees ObservationsCfg unveraendert). Jeder Checkpoint aus B1/B2/B3/C
laesst sich deshalb in jede dieser Konfigurationen laden (--resume) und abspielen.

Nichts unter scripts/ oder vendor/ wird geaendert.
"""

import importlib

import torch

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from cpr.cd1_robust import STAIRS_TERRAINS_CFG
from cpr.unitree_g1_perceptive import foot_clearance_reward_rel, root_height_below_minimum_rel

_unitree = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg")

# --- adaptive Domain Randomization ----------------------------------------------
# Stufe k: Stoesse bis PUSH[k] m/s, PD-Gains +-GAIN[k]. Aufstieg, wenn die beendeten
# Episoden im Mittel >= UP_FRAC der Maximaldauer erreichen, Abstieg unter DOWN_FRAC.
PUSH = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
GAIN = (0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.20)
UP_FRAC, DOWN_FRAC = 0.75, 0.45


def dr_adaptive(env, env_ids, push_term: str = "push_robot", gain_term: str = "actuator_gains"):
    """Stoss- und PD-Gain-Bereich wachsen mit der Ueberlebensdauer (Automatic DR, Folie 11)."""
    st = env.__dict__.setdefault("_claas_dr", {"level": 0, "ema": 0.0, "last": 0})
    if len(env_ids) > 0:
        frac = env.episode_length_buf[env_ids].float().mean().item() / env.max_episode_length
        st["ema"] = 0.98 * st["ema"] + 0.02 * frac
    if env.common_step_counter - st["last"] >= env.max_episode_length:
        st["last"] = env.common_step_counter
        if st["ema"] >= UP_FRAC and st["level"] < len(PUSH) - 1:
            st["level"] += 1
        elif st["ema"] < DOWN_FRAC and st["level"] > 0:
            st["level"] -= 1
    lv = st["level"]
    em = env.event_manager
    if push_term in em.active_terms.get("interval", []):
        v = PUSH[lv]
        em.get_term_cfg(push_term).params["velocity_range"] = {"x": (-v, v), "y": (-v, v)}
    if gain_term in em.active_terms.get("reset", []):
        s = GAIN[lv]
        p = em.get_term_cfg(gain_term).params
        p["stiffness_distribution_params"] = (1.0 - s, 1.0 + s)
        p["damping_distribution_params"] = (1.0 - s, 1.0 + s)
    return torch.tensor(PUSH[lv], device=env.device)


def lin_vel_cmd_adaptive(env, env_ids, reward_term_name: str = "track_lin_vel_xy", frac: float = 0.5):
    """Wie Unitrees lin_vel_cmd_levels (mdp/curriculums.py), aber Schwelle frac statt 0,8."""
    cmd = env.command_manager.get_term("base_velocity").cfg
    ranges, limits = cmd.ranges, cmd.limit_ranges
    w = env.reward_manager.get_term_cfg(reward_term_name).weight
    rew = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    if env.common_step_counter % env.max_episode_length == 0 and rew > w * frac:
        for axis in ("lin_vel_x", "lin_vel_y"):
            lo, hi = getattr(ranges, axis)
            llo, lhi = getattr(limits, axis)
            setattr(ranges, axis, (max(lo - 0.1, llo), min(hi + 0.1, lhi)))
    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def terrain_levels_adaptive(
    env, env_ids, up_frac: float = 0.3, down_frac: float = 0.5, asset_cfg=SceneEntityCfg("robot")
):
    """Isaac Labs terrain_levels_vel, Aufstieg aber schon ab up_frac x Feldlaenge (statt 0,5).

    Abstieg, wenn weniger als down_frac der befohlenen Strecke geschafft wurde (Isaac Lab: 0,5).
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] * up_frac
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * down_frac
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def _common(cfg):
    """Gemeinsam fuer B1/B2/B3/C: DR-Terme, adaptive Curricula, Startbefehle."""
    ev = cfg.events
    ev.physics_material.params["static_friction_range"] = (0.3, 1.25)
    ev.physics_material.params["dynamic_friction_range"] = (0.3, 1.2)
    ev.base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
        },
    )
    ev.actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (1.0 - GAIN[0], 1.0 + GAIN[0]),
            "damping_distribution_params": (1.0 - GAIN[0], 1.0 + GAIN[0]),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    ev.push_robot.interval_range_s = (5.0, 12.0)
    ev.push_robot.params["velocity_range"] = {"x": (-PUSH[0], PUSH[0]), "y": (-PUSH[0], PUSH[0])}
    # Befehle: Start bei dem, was die Baseline sicher kann
    cmd = cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (-0.3, 0.5)
    cmd.ranges.lin_vel_y = (-0.2, 0.2)
    cfg.curriculum.lin_vel_cmd_levels = CurrTerm(func=lin_vel_cmd_adaptive)
    cfg.curriculum.dr_level = CurrTerm(func=dr_adaptive)


def base_height_l2_safe(env, target_height, asset_cfg=SceneEntityCfg("robot"), sensor_cfg=None):
    """Wie isaaclab base_height_l2, aber robust: Hoehenscanner-Strahlen ohne Treffer liefern inf.
    Mit dem Original wurde der Reward dann -inf und hat jedes Treppen-Training zerstoert
    (B2, B5, A3, T1 abgestuerzt, T2/T3 NaN - Diagnose 19.11.). Nur gueltige Strahlen mitteln;
    ohne gueltigen Strahl keine Strafe; Strafe auf 1,0 begrenzt."""
    import torch

    asset = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2]
    if sensor_cfg is not None:
        hits = env.scene[sensor_cfg.name].data.ray_hits_w[..., 2]
        ok = torch.isfinite(hits)
        n = ok.sum(dim=1)
        ground = torch.where(ok, hits, torch.zeros_like(hits)).sum(dim=1) / n.clamp(min=1)
        h = torch.where(n > 0, z - ground, torch.full_like(z, float(target_height)))
    else:
        h = z
    return torch.square(h - target_height).clamp(max=1.0)


def _reward_guard():
    """Sicherheitsnetz (19./20.11.): Rewards pro Schritt auf [-10, 10] begrenzen (nan/inf -> 0 bzw.
    Grenze) und Beobachtungen auf [-100, 100] (nan/inf abgefangen). T2b/T3b entgleisten trotz
    Reward-Fix, weil einzelne Physik-Ausreisser auf Treppen riesige, aber endliche Werte lieferten
    (value loss inf). Meldet einmalig, wenn es greift."""
    try:
        import torch
        from isaaclab.managers import ObservationManager, RewardManager

        orig = RewardManager.compute
        if getattr(orig, "_claas_guard", False):
            return

        def compute(self, dt):
            r = orig(self, dt)
            if not torch.isfinite(r).all() or (r.abs() > 10.0).any():
                if not getattr(self, "_claas_warned", False):
                    print("[claas_milestone] WARNUNG: Reward ungueltig/zu gross -> begrenzt")
                    self._claas_warned = True
                r = torch.nan_to_num(r, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
                self._reward_buf = r
            return r

        compute._claas_guard = True
        RewardManager.compute = compute

        ocompute_orig = ObservationManager.compute

        def _fix(x):
            if isinstance(x, torch.Tensor):
                return torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
            if isinstance(x, dict):
                return {k: _fix(v) for k, v in x.items()}
            return x

        def ocompute(self, *a, **k):
            return _fix(ocompute_orig(self, *a, **k))

        ObservationManager.compute = ocompute
        print("[claas_milestone] Reward-/Beobachtungs-Schutz aktiv")
    except Exception as e:  # noqa: BLE001
        print("[claas_milestone] Reward-Schutz nicht aktiv:", repr(e))


_reward_guard()


def _stairs(cfg):
    cfg.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG.copy()
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum.terrain_levels = CurrTerm(func=terrain_levels_adaptive)
    # Hoehen relativ zum Boden (E-008/E-010) - auf Treppen noetig
    cfg.rewards.base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
    cfg.rewards.base_height.func = base_height_l2_safe  # Fix 19.11.: inf bei Strahlen ohne Treffer
    cfg.rewards.feet_clearance.func = foot_clearance_reward_rel
    cfg.terminations.base_height.func = root_height_below_minimum_rel


# --- Training ----------------------------------------------------------------------


@configclass
class MsStairsEnvCfg(_unitree.RobotEnvCfg):
    """B1: Treppen-Mix, adaptive Curricula, Unitrees Rewards."""

    def __post_init__(self):
        _stairs(self)
        super().__post_init__()  # schaltet terrain_generator.curriculum ein
        _common(self)


@configclass
class MsStairsTrackEnvCfg(MsStairsEnvCfg):
    """B2: wie B1, Tracking-Gewicht 2,0 (gegen das Auf-der-Stelle-Treten aus Lauf A)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy.weight = 2.0


@configclass
class MsFlatEnvCfg(_unitree.RobotEnvCfg):
    """B3: ebener Boden, adaptive DR - robuste Absicherung."""

    def __post_init__(self):
        super().__post_init__()
        _common(self)


# --- Videos ------------------------------------------------------------------------


def _video_base(cfg, speed=0.5):
    cfg.scene.num_envs = 9
    for name in ("terrain_levels", "lin_vel_cmd_levels", "dr_level"):
        setattr(cfg.curriculum, name, None)
    cmd = cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (speed, speed)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.rel_standing_envs = 0.0
    cmd.resampling_time_range = (1e6, 1e6)
    cfg.events.actuator_gains = None
    cfg.events.push_robot = None
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (2.5, 2.5, 1.6)
    cfg.viewer.lookat = (0.0, 0.0, 0.4)


def _video_stairs(cfg, difficulty):
    gen = STAIRS_TERRAINS_CFG.replace(
        num_rows=3,
        num_cols=4,
        curriculum=False,
        difficulty_range=difficulty,
        sub_terrains={
            "stairs_down": STAIRS_TERRAINS_CFG.sub_terrains["stairs_down"],
            "stairs_up": STAIRS_TERRAINS_CFG.sub_terrains["stairs_up"],
        },
    )
    cfg.scene.terrain.terrain_generator = gen
    cfg.scene.terrain.max_init_terrain_level = None


@configclass
class MsPlayStairsEasyCfg(MsStairsEnvCfg):
    """Video: Treppen 2-7 cm, 0,5 m/s geradeaus."""

    def __post_init__(self):
        super().__post_init__()
        _video_base(self)
        _video_stairs(self, (0.0, 0.3))


@configclass
class MsPlayStairsMidCfg(MsStairsEnvCfg):
    """Video: Treppen ca. 7-12 cm, 0,5 m/s geradeaus."""

    def __post_init__(self):
        super().__post_init__()
        _video_base(self)
        _video_stairs(self, (0.3, 0.6))


@configclass
class MsPlayFlatPushCfg(MsFlatEnvCfg):
    """Video: eben, 0,6 m/s, Stoesse bis 0,8 m/s alle 3 s (Robustheit)."""

    def __post_init__(self):
        super().__post_init__()
        _video_base(self, speed=0.6)
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 5
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 3.0),
            params={"velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}},
        )


# === Version 2 (18.11. abends, B4/B5) ================================================
# Befund aus B1/B2 nach ~400 Iterationen: Stoesse und Befehle stiegen fast sofort auf
# das Maximum (Kriterium Episodenlaenge war zu gutmuetig - Stuerze kommen oft spaet in
# der Episode), Gelaende blieb auf Stufe ~0. Version 2:
#   - DR-Stufe folgt der STURZRATE: hoch bei < 30 %, runter bei > 50 %
#   - Befehle wachsen erst ab 70 % Tracking-Reward (statt 50 %)
#   - Gelaende-Abstieg erst unter 25 % der befohlenen Strecke (statt 50 %)
FALL_UP, FALL_DOWN = 0.30, 0.50


def dr_adaptive_falls(env, env_ids, push_term: str = "push_robot", gain_term: str = "actuator_gains"):
    """Wie dr_adaptive, aber gesteuert ueber den Anteil der Episoden, die mit Sturz enden."""
    st = env.__dict__.setdefault("_claas_dr2", {"level": 0, "ema": 1.0, "last": 0})
    if len(env_ids) > 0:
        tm = env.termination_manager
        fell = tm.get_term("bad_orientation")[env_ids] | tm.get_term("base_height")[env_ids]
        st["ema"] = 0.98 * st["ema"] + 0.02 * fell.float().mean().item()
    if env.common_step_counter - st["last"] >= env.max_episode_length:
        st["last"] = env.common_step_counter
        if st["ema"] < FALL_UP and st["level"] < len(PUSH) - 1:
            st["level"] += 1
        elif st["ema"] > FALL_DOWN and st["level"] > 0:
            st["level"] -= 1
    lv = st["level"]
    em = env.event_manager
    if push_term in em.active_terms.get("interval", []):
        v = PUSH[lv]
        em.get_term_cfg(push_term).params["velocity_range"] = {"x": (-v, v), "y": (-v, v)}
    if gain_term in em.active_terms.get("reset", []):
        s = GAIN[lv]
        p = em.get_term_cfg(gain_term).params
        p["stiffness_distribution_params"] = (1.0 - s, 1.0 + s)
        p["damping_distribution_params"] = (1.0 - s, 1.0 + s)
    return torch.tensor(PUSH[lv], device=env.device)


def _v2(cfg):
    cfg.curriculum.dr_level = CurrTerm(func=dr_adaptive_falls)
    cfg.curriculum.lin_vel_cmd_levels = CurrTerm(func=lin_vel_cmd_adaptive, params={"frac": 0.7})
    cfg.curriculum.terrain_levels = CurrTerm(func=terrain_levels_adaptive, params={"down_frac": 0.25})


@configclass
class MsStairsV2EnvCfg(MsStairsEnvCfg):
    """B4: Treppen, Version-2-Curricula, Unitrees Rewards."""

    def __post_init__(self):
        super().__post_init__()
        _v2(self)


@configclass
class MsStairsV2TrackEnvCfg(MsStairsV2EnvCfg):
    """B5: wie B4, Tracking-Gewicht 2,0 (Reward-Fix aus B2)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy.weight = 2.0
