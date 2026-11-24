"""Lehrer-Modell (Teacher) fuer Treppen: 23-DoF-G1 mit exaktem Hoehenraster vor dem Roboter.

Claas, 19.11. Idee (Vorlesung, Teacher-Student): Der Lehrer bekommt im Simulator
privilegierte, exakte Informationen - hier das Hoehenraster vor den Fuessen ohne Rauschen -
und lernt damit die Treppen. Spaeter lernt ein Schueler mit realistischer, verrauschter
Kamera-Hoehenkarte (D435) per Distillation vom Lehrer (CD2).

Basis:
- Rezept wie T3/B5 (Treppen-Mix, Curricula V2, Tracking-Gewicht 2,0, abgesicherter
  base_height-Reward) aus cpr/claas_g1_23dof.py / cpr/claas_milestone.py
- Hoehenraster wie in der Team-Konfiguration cpr/unitree_g1_perceptive.py (Unitrees Scanner,
  Historie pro Term), aber: eigener Scanner nach vorne versetzt (ca. -0,3..+1,3 m, wie im
  ToDo "Policy mit Height Map trainieren"), ohne Rauschen, inf/nan abgefangen.
  Der vorhandene Scanner unter dem Roboter bleibt fuer Reward/Abbruch unveraendert.
- Start vom ebenen 23-DoF-Modell A2: Das Netz bekommt 187 zusaetzliche Eingaenge. Beim Laden
  werden die A2-Gewichte uebernommen und die neuen Eingaenge mit Gewicht 0 angehaengt -> am
  Anfang verhaelt sich der Lehrer exakt wie A2 und lernt dann, das Raster zu nutzen.
  Der Optimizer-Zustand wird dabei nicht geladen (passt nicht zur neuen Groesse).
"""

import copy

import torch

from isaaclab.envs import mdp as il_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from cpr import claas_g1_23dof as g23
from cpr.unitree_g1_perceptive import _history_per_term  # nur gelesen, Datei bleibt unveraendert

FRONT = SceneEntityCfg("height_scanner_front")


def height_scan_safe(env, sensor_cfg, offset: float = 0.5):
    """Isaac Labs height_scan, aber Strahlen ohne Treffer (inf/nan) werden zu 'tief' (-1)."""
    h = il_mdp.height_scan(env, sensor_cfg, offset)
    return torch.nan_to_num(h, nan=-1.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


@configclass
class TeacherPolicyCfg(g23._unitree.ObservationsCfg.PolicyCfg):
    height_scan = ObsTerm(func=height_scan_safe, params={"sensor_cfg": FRONT})


@configclass
class TeacherCriticCfg(g23._unitree.ObservationsCfg.CriticCfg):
    height_scan = ObsTerm(func=height_scan_safe, params={"sensor_cfg": FRONT})


def _teacher(cfg):
    front = copy.deepcopy(cfg.scene.height_scanner)
    front.offset.pos = (0.5, 0.0, 20.0)  # Raster 1,6 m lang -> ca. -0,3..+1,3 m vor dem Becken
    cfg.scene.height_scanner_front = front
    cfg.observations.policy = TeacherPolicyCfg()
    cfg.observations.critic = TeacherCriticCfg()
    _history_per_term(cfg.observations.policy)
    _history_per_term(cfg.observations.critic)


def _patch_resume():
    """Laden des kleineren A2-Netzes in das groessere Lehrer-Netz erlauben."""
    try:
        from rsl_rl.modules import ActorCritic
        from rsl_rl.runners import OnPolicyRunner

        if getattr(ActorCritic.load_state_dict, "_claas", False):
            return
        orig_lsd = ActorCritic.load_state_dict

        def load_state_dict(self, sd, strict=True, *a, **k):
            own = self.state_dict()
            sd = dict(sd)
            for name, v in list(sd.items()):
                o = own.get(name)
                if o is not None and v.dim() == 2 and o.dim() == 2 and o.shape[0] == v.shape[0] and o.shape[1] > v.shape[1]:
                    pad = torch.zeros(v.shape[0], o.shape[1] - v.shape[1], dtype=v.dtype, device=v.device)
                    sd[name] = torch.cat([v, pad], dim=1)
                    print(f"[claas_teacher] {name}: {tuple(v.shape)} -> {tuple(o.shape)} (neue Eingaenge = 0)")
            return orig_lsd(self, sd, strict, *a, **k)

        load_state_dict._claas = True
        ActorCritic.load_state_dict = load_state_dict

        orig_load = OnPolicyRunner.load

        def load(self, path, load_optimizer=True, *a, **k):
            return orig_load(self, path, False, *a, **k)

        OnPolicyRunner.load = load
        print("[claas_teacher] Resume mit erweitertem Eingang aktiv (Optimizer wird neu gestartet)")
    except Exception as e:  # noqa: BLE001
        print("[claas_teacher] Resume-Patch nicht aktiv:", repr(e))


_patch_resume()


@configclass
class G23TeacherEnvCfg(g23.G23StairsTrackEnvCfg):
    """Lehrer: Treppen-Mix + exaktes Hoehenraster vorne, Start ab A2."""

    def __post_init__(self):
        super().__post_init__()
        _teacher(self)
        cmd = self.commands.base_velocity
        # ab A2: lineare Befehle sofort im vollen Bereich (A2 beherrscht sie eben)
        cmd.ranges.lin_vel_x = cmd.limit_ranges.lin_vel_x
        cmd.ranges.lin_vel_y = cmd.limit_ranges.lin_vel_y


@configclass
class G23TeacherPlayStairsEasyCfg(g23.G23PlayStairsEasyCfg):
    def __post_init__(self):
        super().__post_init__()
        _teacher(self)


@configclass
class G23TeacherPlayStairsMidCfg(g23.G23PlayStairsMidCfg):
    def __post_init__(self):
        super().__post_init__()
        _teacher(self)


@configclass
class G23TeacherPlayFlatPushCfg(g23.G23PlayFlatPushCfg):
    def __post_init__(self):
        super().__post_init__()
        _teacher(self)


@configclass
class G23TeacherRobustEnvCfg(G23TeacherEnvCfg):
    """L2: Lehrer + Latenz + Motorstaerke + Drehen (Start: L1). Robuster Lehrer = bessere Vorlage
    fuer den Schueler (CD2)."""

    def __post_init__(self):
        super().__post_init__()
        g23._robust(self)
        g23._turn(self, start=0.2)
