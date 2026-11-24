"""Abgabe-Videos Treppen (Claas, 23.11.): ein Roboter von der Seite verfolgt, 1280x720, lange Episode.

Gleiches Gelaende wie die Treppen-Videos (Treppen auf/ab, 0,5 m/s geradeaus), nur Kamera und Laenge
fuer ein offizielles Video angepasst. Blinde Policies (A8, T3b) und der Lehrer L1 (mit Hoehenraster).
"""

from isaaclab.utils import configclass

from cpr import claas_g1_23dof as g23
from cpr import claas_g1_23dof_teacher as teach


def _show(cfg):
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (0.6, -3.0, 1.0)  # seitlich, leicht erhoeht
    cfg.viewer.lookat = (0.6, 0.0, 0.3)
    cfg.viewer.resolution = (1280, 720)
    cfg.episode_length_s = 60.0
    cfg.commands.base_velocity.debug_vis = False


@configclass
class G23ShowStairsEasyCfg(g23.G23PlayStairsEasyCfg):
    def __post_init__(self):
        super().__post_init__()
        _show(self)


@configclass
class G23ShowStairsMidCfg(g23.G23PlayStairsMidCfg):
    def __post_init__(self):
        super().__post_init__()
        _show(self)


@configclass
class G23TeacherShowStairsEasyCfg(teach.G23TeacherPlayStairsEasyCfg):
    def __post_init__(self):
        super().__post_init__()
        _show(self)


@configclass
class G23TeacherShowStairsMidCfg(teach.G23TeacherPlayStairsMidCfg):
    def __post_init__(self):
        super().__post_init__()
        _show(self)
