"""Abgabe-Videos Treppen (Claas, 23.11.): ein Roboter von der Seite verfolgt, 1280x720, lange Episode.

Gleiches Gelaende wie die Treppen-Videos (Treppen auf/ab, 0,5 m/s geradeaus), nur Kamera und Laenge
fuer ein offizielles Video angepasst. Blinde Policies (A8, T3b) und der Lehrer L1 (mit Hoehenraster).
"""

from isaaclab.utils import configclass

from cpr import claas_g1_23dof as g23
from cpr import claas_g1_23dof_teacher as teach
from cpr.cd1_robust import STAIRS_TERRAINS_CFG

# v2 (23.11. abends): v1 lief nach wenigen Sekunden aus dem Treppenfeld heraus (zufaellige Kachel,
# zufaellige Blickrichtung). Jetzt: jeder der 9 Roboter hat eine eigene Spur (Spalte), startet in
# Reihe 0 mit Blick nach +x und laeuft durch die Reihen = steigende Stufenhoehe (Grube -> hoch,
# naechste Kachel -> runter in die Grube -> wieder hoch ...).


def _show_terrain(cfg, difficulty):
    gen = STAIRS_TERRAINS_CFG.replace(
        num_rows=4, num_cols=9, curriculum=True, difficulty_range=difficulty,
        sub_terrains={"stairs_up": STAIRS_TERRAINS_CFG.sub_terrains["stairs_up"].replace(proportion=1.0)},
    )
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = gen
    cfg.scene.terrain.max_init_terrain_level = 0
    if hasattr(cfg.curriculum, "terrain_levels"):
        cfg.curriculum.terrain_levels = None
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}


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
        _show_terrain(self, (0.0, 0.4))


@configclass
class G23ShowStairsMidCfg(g23.G23PlayStairsMidCfg):
    def __post_init__(self):
        super().__post_init__()
        _show(self)
        _show_terrain(self, (0.3, 0.8))


@configclass
class G23TeacherShowStairsEasyCfg(teach.G23TeacherPlayStairsEasyCfg):
    def __post_init__(self):
        super().__post_init__()
        _show(self)
        _show_terrain(self, (0.0, 0.4))


@configclass
class G23TeacherShowStairsMidCfg(teach.G23TeacherPlayStairsMidCfg):
    def __post_init__(self):
        super().__post_init__()
        _show(self)
        _show_terrain(self, (0.3, 0.8))


# --- CD2: L3 (EDS-Raster) im selben Treppen-Gelaende (24.11.) ---
from cpr import claas_g1_23dof_eds as _eds_mod  # noqa: E402


@configclass
class G23TeacherEdsShowStairsEasyCfg(g23.G23PlayStairsEasyCfg):
    def __post_init__(self):
        super().__post_init__()
        _show(self)
        _show_terrain(self, (0.0, 0.4))
        _eds_mod._eds(self)


@configclass
class G23TeacherEdsShowStairsMidCfg(g23.G23PlayStairsMidCfg):
    def __post_init__(self):
        super().__post_init__()
        _show(self)
        _show_terrain(self, (0.3, 0.8))
        _eds_mod._eds(self)
