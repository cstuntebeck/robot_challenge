"""CD2: Lehrer im Hoehenkarten-Format des EDS (Vorlesung RL_v2, Dennis Bank) - Claas, 23.11.

Aufteilung laut Vorlesung (S. 21): EDS schaetzt aus Kamera/LiDAR/IMU eine Hoehenkarte,
die Policy macht aus Hoehenkarte + Propriozeption Aktionen. Damit die Policy spaeter mit der
GESCHAETZTEN Hoehenkarte laeuft, trainieren wir sie hier schon so:

- Raster wie auf Folie 36: 11 (quer) x 15 (laengs) = 165 Punkte, Abstand 1/14 m,
  quer -0,36..+0,36 m, laengs -0,3..+0,7 m um das Becken, mit Gierwinkel ausgerichtet.
  (Aus der Folie abgelesen - mit Dennis abgleichen, siehe ToDo CD2.)
- Policy bekommt eine "EDS-artige" Hoehenkarte: nur alle 5 Policy-Schritte neu (10 Hz),
  3x3-geglaettet (feine Kanten verwischt, Folie 37/38), Rauschen ~2-3 cm (MAE der Folie)
  plus ein langsamer Versatz je Roboter (Drift). Der Critic bekommt die exakte Karte.
- Start: A8 (robust, unebener Boden). Neue 165 Eingaenge werden mit 0 angehaengt
  (Resume-Patch aus claas_g1_23dof_teacher), d. h. am Anfang laeuft er exakt wie A8.
- Gelaende/Robustheit wie L2 (Treppen-Mix, Latenz, Motorstaerke, Drehen).
"""

import copy

import torch

from isaaclab.envs import mdp as il_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.utils import configclass

from cpr import claas_g1_23dof as g23
from cpr import claas_g1_23dof_teacher as teach  # bringt _patch_resume mit
from cpr.unitree_g1_perceptive import _history_per_term  # nur gelesen

EDS = SceneEntityCfg("height_scanner_eds")
EDS_RES = 1.0 / 14.0
EDS_SIZE = (1.0, 10.0 / 14.0)  # laengs 15 Punkte, quer 11 Punkte
EDS_CENTER_X = 0.2  # Raster -0,3..+0,7 m
HM_PERIOD = 5  # Policy 50 Hz, Hoehenkarte 10 Hz
HM_NOISE = 0.025
HM_DRIFT = 0.02


def height_scan_eds_exact(env, sensor_cfg=EDS, offset: float = 0.5):
    h = il_mdp.height_scan(env, sensor_cfg, offset)
    return torch.nan_to_num(h, nan=-1.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def _blur3(h, n_x, n_y):
    """3x3-Mittelung im Raster (Randwerte wiederholt). h: (N, n_x*n_y) in Isaac-Reihenfolge."""
    import torch.nn.functional as F

    n = h.shape[0]
    g = h.view(n, 1, n_y, n_x)
    g = F.pad(g, (1, 1, 1, 1), mode="replicate")
    g = F.avg_pool2d(g, 3, stride=1)
    return g.reshape(n, -1)


def height_scan_eds_est(env, sensor_cfg=EDS, offset: float = 0.5, period: int = HM_PERIOD,
                        noise: float = HM_NOISE, drift: float = HM_DRIFT, blur: bool = True):
    """Hoehenkarte, wie sie der EDS liefern wuerde: 10 Hz, geglaettet, verrauscht, leicht versetzt."""
    exact = height_scan_eds_exact(env, sensor_cfg, offset)
    n, d = exact.shape
    st = getattr(env, "_claas_eds", None)
    if st is None or st["buf"].shape != exact.shape:
        st = {"buf": exact.clone(), "bias": torch.zeros(n, 1, device=exact.device), "nx": None}
        sensor = env.scene[sensor_cfg.name]
        pts = sensor.ray_starts[0, :, :2]
        st["nx"] = int(torch.unique(torch.round(pts[:, 0] * 1000)).numel())
        env._claas_eds = st
    nx = st["nx"]
    ny = d // nx
    est = _blur3(exact, nx, ny) if blur and nx * ny == d else exact
    # Drift: langsamer Zufallsversatz je Roboter, bei Episodenstart neu
    reset = env.episode_length_buf <= 1
    st["bias"] = 0.98 * st["bias"] + 0.02 * torch.randn_like(st["bias"]) * drift * 5
    st["bias"][reset] = torch.randn_like(st["bias"][reset]) * drift
    est = est + st["bias"] + torch.randn_like(est) * noise
    upd = (env.episode_length_buf % period == 0) | reset
    st["buf"][upd] = est[upd]
    return st["buf"].clamp(-1.0, 1.0)


@configclass
class EdsPolicyCfg(g23._unitree.ObservationsCfg.PolicyCfg):
    height_scan = ObsTerm(func=height_scan_eds_est, params={"sensor_cfg": EDS})


@configclass
class EdsCriticCfg(g23._unitree.ObservationsCfg.CriticCfg):
    height_scan = ObsTerm(func=height_scan_eds_exact, params={"sensor_cfg": EDS})


@configclass
class EdsExactPolicyCfg(g23._unitree.ObservationsCfg.PolicyCfg):
    height_scan = ObsTerm(func=height_scan_eds_exact, params={"sensor_cfg": EDS})


def _eds_scanner(cfg):
    sc = copy.deepcopy(cfg.scene.height_scanner)
    sc.offset.pos = (EDS_CENTER_X, 0.0, 20.0)
    sc.pattern_cfg = patterns.GridPatternCfg(resolution=EDS_RES, size=EDS_SIZE)
    cfg.scene.height_scanner_eds = sc


def _eds(cfg, exact_policy=False):
    _eds_scanner(cfg)
    if hasattr(cfg.scene, "height_scanner_front"):
        cfg.scene.height_scanner_front = None  # 187er-Raster von L1/L2 wird nicht mehr gebraucht
    cfg.observations.policy = EdsExactPolicyCfg() if exact_policy else EdsPolicyCfg()
    cfg.observations.critic = EdsCriticCfg()
    _history_per_term(cfg.observations.policy)
    _history_per_term(cfg.observations.critic)


@configclass
class G23TeacherEdsEnvCfg(teach.G23TeacherRobustEnvCfg):
    """L3: Lehrer mit EDS-Raster (165), geschaetzte Karte fuer die Policy, Start A8."""

    def __post_init__(self):
        super().__post_init__()
        _eds(self)


@configclass
class G23TeacherEdsExactEnvCfg(teach.G23TeacherRobustEnvCfg):
    """L3x: wie L3, aber Policy sieht die exakte Karte (Vergleich: was kostet die Schaetzung?)."""

    def __post_init__(self):
        super().__post_init__()
        _eds(self, exact_policy=True)


@configclass
class G23TeacherEdsPlayStairsEasyCfg(g23.G23PlayStairsEasyCfg):
    def __post_init__(self):
        super().__post_init__()
        _eds(self)


@configclass
class G23TeacherEdsPlayStairsMidCfg(g23.G23PlayStairsMidCfg):
    def __post_init__(self):
        super().__post_init__()
        _eds(self)


@configclass
class G23TeacherEdsPlayFlatPushCfg(g23.G23PlayFlatPushCfg):
    def __post_init__(self):
        super().__post_init__()
        _eds(self)
