"""Starter fuer Unitree RL Lab - KOPIE von scripts/unitree_run.py (Claas).

scripts/unitree_run.py wird fuer beide Gruppen gepflegt und nicht geaendert
(KAMERA_D435.md). Diese Kopie registriert zusaetzlich die eigenen Tasks von Claas:
  - CD1 Lauf A (cpr/cd1_robust.py)
  - Meilenstein 18./19.11. (cpr/claas_milestone.py)

Aufruf im Container:
    /isaac-sim/python.sh claas/unitree_run.py {train|play} <Argumente fuer Unitrees Skript>
"""

import os
import runpy
import sys

import gymnasium as gym

ROOT = os.environ["UNITREE_RL_LAB"]  # hochgeladene Kopie, siehe scripts/vendor_unitree.sh
CFG = "cpr.unitree_g1_perceptive"
MS = "cpr.claas_milestone"
AGENT = "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg"

TASKS = {
    "Unitree-G1-29dof-Velocity": ("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg:RobotEnvCfg","unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg:RobotPlayEnvCfg",),
    # Lauf B: Hoehenraster + unebenes Gelaende
    "Unitree-G1-29dof-Velocity-Perceptive": ("PerceptiveEnvCfg", "PerceptivePlayEnvCfg"),
    # Lauf B mit bodenrelativer Koerperhoehe (E-008)
    "Unitree-G1-29dof-Velocity-PerceptiveRelHeight": ("PerceptiveRelHeightEnvCfg", "PerceptiveRelHeightPlayEnvCfg"),
    # E-010: Befehls-Zeitplan; B zusaetzlich Abbruch und Fussfreiheit bodenrelativ
    "Unitree-G1-29dof-Velocity-CmdSchedule": ("CmdScheduleEnvCfg", "BlindRoughPlayEnvCfg"),
    "Unitree-G1-29dof-Velocity-PerceptiveCmdSchedule": ("PerceptiveCmdScheduleEnvCfg", "PerceptiveCmdSchedulePlayEnvCfg"),
    # nur zum Abspielen: Policy aus A im Gelaende von B
    "Unitree-G1-29dof-Velocity-BlindRough": ("BlindRoughPlayEnvCfg", "BlindRoughPlayEnvCfg"),
    # --- Claas -----------------------------------------------------------------
    # CD1 Lauf A (17.11.): Treppen + DR ab Iteration 0 - gescheitert, nur zum Nachvollziehen
    "Unitree-G1-29dof-Velocity-CD1-Claas": ("cpr.cd1_robust:CD1RobustEnvCfg", "cpr.cd1_robust:CD1RobustPlayEnvCfg"),
    # Meilenstein 18./19.11.: Stufe 1 (parallel, ab Baseline 3000)
    "Unitree-G1-Claas-MS-B1": (f"{MS}:MsStairsEnvCfg", f"{MS}:MsPlayStairsEasyCfg"),
    "Unitree-G1-Claas-MS-B2": (f"{MS}:MsStairsTrackEnvCfg", f"{MS}:MsPlayStairsEasyCfg"),
    "Unitree-G1-Claas-MS-B3": (f"{MS}:MsFlatEnvCfg", f"{MS}:MsPlayFlatPushCfg"),
    # Version 2 (18.11. abends): Sturzraten-gesteuerte DR, strengere Befehle, milderer Gelaende-Abstieg
    "Unitree-G1-Claas-MS-B4": (f"{MS}:MsStairsV2EnvCfg", f"{MS}:MsPlayStairsEasyCfg"),
    "Unitree-G1-Claas-MS-B5": (f"{MS}:MsStairsV2TrackEnvCfg", f"{MS}:MsPlayStairsEasyCfg"),
    "Unitree-G1-Claas-MS-C-StairsV2": (f"{MS}:MsStairsV2EnvCfg", f"{MS}:MsPlayStairsEasyCfg"),
    "Unitree-G1-Claas-MS-C-StairsV2Track": (f"{MS}:MsStairsV2TrackEnvCfg", f"{MS}:MsPlayStairsEasyCfg"),
    # Stufe 2: Lauf C (Konfiguration je nach Auswahl)
    "Unitree-G1-Claas-MS-C-Stairs": (f"{MS}:MsStairsEnvCfg", f"{MS}:MsPlayStairsEasyCfg"),
    "Unitree-G1-Claas-MS-C-StairsTrack": (f"{MS}:MsStairsTrackEnvCfg", f"{MS}:MsPlayStairsEasyCfg"),
    # --- 23 DoF (Roboter der RC, ab 18.11. abends) - cpr/claas_g1_23dof.py ---
    "Unitree-G1-Claas-23-Flat": ("cpr.claas_g1_23dof:G23FlatEnvCfg", "cpr.claas_g1_23dof:G23PlayFlatPushCfg"),
    "Unitree-G1-Claas-23-FlatTrack": ("cpr.claas_g1_23dof:G23FlatTrackEnvCfg", "cpr.claas_g1_23dof:G23PlayFlatPushCfg"),
    "Unitree-G1-Claas-23-StairsTrack": ("cpr.claas_g1_23dof:G23StairsTrackEnvCfg", "cpr.claas_g1_23dof:G23PlayStairsEasyCfg"),
    "Unitree-G1-Claas-23-Stairs": ("cpr.claas_g1_23dof:G23StairsEnvCfg", "cpr.claas_g1_23dof:G23PlayStairsEasyCfg"),
    "Unitree-G1-Claas-23-Play-StairsEasy": ("cpr.claas_g1_23dof:G23PlayStairsEasyCfg", "cpr.claas_g1_23dof:G23PlayStairsEasyCfg"),
    "Unitree-G1-Claas-23-Play-StairsMid": ("cpr.claas_g1_23dof:G23PlayStairsMidCfg", "cpr.claas_g1_23dof:G23PlayStairsMidCfg"),
    "Unitree-G1-Claas-23-Robust": ("cpr.claas_g1_23dof:G23RobustEnvCfg", "cpr.claas_g1_23dof:G23PlayRobustCfg"),
    "Unitree-G1-Claas-23-Play-Robust": ("cpr.claas_g1_23dof:G23PlayRobustCfg", "cpr.claas_g1_23dof:G23PlayRobustCfg"),
    "Unitree-G1-Claas-23-Teacher": ("cpr.claas_g1_23dof_teacher:G23TeacherEnvCfg", "cpr.claas_g1_23dof_teacher:G23TeacherPlayStairsEasyCfg"),
    "Unitree-G1-Claas-23-Teacher-Play-StairsEasy": ("cpr.claas_g1_23dof_teacher:G23TeacherPlayStairsEasyCfg", "cpr.claas_g1_23dof_teacher:G23TeacherPlayStairsEasyCfg"),
    "Unitree-G1-Claas-23-Teacher-Play-StairsMid": ("cpr.claas_g1_23dof_teacher:G23TeacherPlayStairsMidCfg", "cpr.claas_g1_23dof_teacher:G23TeacherPlayStairsMidCfg"),
    "Unitree-G1-Claas-23-Teacher-Play-FlatPush": ("cpr.claas_g1_23dof_teacher:G23TeacherPlayFlatPushCfg", "cpr.claas_g1_23dof_teacher:G23TeacherPlayFlatPushCfg"),
    "Unitree-G1-Claas-23-RobustTurn": ("cpr.claas_g1_23dof:G23RobustTurnEnvCfg", "cpr.claas_g1_23dof:G23PlayTurnCfg"),
    "Unitree-G1-Claas-23-Play-Turn": ("cpr.claas_g1_23dof:G23PlayTurnCfg", "cpr.claas_g1_23dof:G23PlayTurnCfg"),
    "Unitree-G1-Claas-23-StairsRobust": ("cpr.claas_g1_23dof:G23StairsRobustEnvCfg", "cpr.claas_g1_23dof:G23PlayStairsEasyCfg"),
    "Unitree-G1-Claas-23-Play-ShowStairsEasy": ("cpr.claas_g1_23dof_show:G23ShowStairsEasyCfg", "cpr.claas_g1_23dof_show:G23ShowStairsEasyCfg"),
    "Unitree-G1-Claas-23-Play-ShowStairsMid": ("cpr.claas_g1_23dof_show:G23ShowStairsMidCfg", "cpr.claas_g1_23dof_show:G23ShowStairsMidCfg"),
    "Unitree-G1-Claas-23-Teacher-Play-ShowStairsEasy": ("cpr.claas_g1_23dof_show:G23TeacherShowStairsEasyCfg", "cpr.claas_g1_23dof_show:G23TeacherShowStairsEasyCfg"),
    "Unitree-G1-Claas-23-Teacher-Play-ShowStairsMid": ("cpr.claas_g1_23dof_show:G23TeacherShowStairsMidCfg", "cpr.claas_g1_23dof_show:G23TeacherShowStairsMidCfg"),
    "Unitree-G1-Claas-23-TeacherEds-Play-ShowStairsEasy": ("cpr.claas_g1_23dof_show:G23TeacherEdsShowStairsEasyCfg", "cpr.claas_g1_23dof_show:G23TeacherEdsShowStairsEasyCfg"),
    "Unitree-G1-Claas-23-TeacherEds-Play-ShowStairsMid": ("cpr.claas_g1_23dof_show:G23TeacherEdsShowStairsMidCfg", "cpr.claas_g1_23dof_show:G23TeacherEdsShowStairsMidCfg"),
    "Unitree-G1-Claas-23-TeacherEds": ("cpr.claas_g1_23dof_eds:G23TeacherEdsEnvCfg", "cpr.claas_g1_23dof_eds:G23TeacherEdsPlayStairsEasyCfg"),
    "Unitree-G1-Claas-23-TeacherEdsExact": ("cpr.claas_g1_23dof_eds:G23TeacherEdsExactEnvCfg", "cpr.claas_g1_23dof_eds:G23TeacherEdsPlayStairsEasyCfg"),
    "Unitree-G1-Claas-23-TeacherEds-Play-StairsEasy": ("cpr.claas_g1_23dof_eds:G23TeacherEdsPlayStairsEasyCfg", "cpr.claas_g1_23dof_eds:G23TeacherEdsPlayStairsEasyCfg"),
    "Unitree-G1-Claas-23-TeacherEds-Play-StairsMid": ("cpr.claas_g1_23dof_eds:G23TeacherEdsPlayStairsMidCfg", "cpr.claas_g1_23dof_eds:G23TeacherEdsPlayStairsMidCfg"),
    "Unitree-G1-Claas-23-TeacherEds-Play-FlatPush": ("cpr.claas_g1_23dof_eds:G23TeacherEdsPlayFlatPushCfg", "cpr.claas_g1_23dof_eds:G23TeacherEdsPlayFlatPushCfg"),
    "Unitree-G1-Claas-23-TeacherRobust": ("cpr.claas_g1_23dof_teacher:G23TeacherRobustEnvCfg", "cpr.claas_g1_23dof_teacher:G23TeacherPlayStairsEasyCfg"),
    "Unitree-G1-Claas-23-Gait": ("cpr.claas_g1_23dof:G23GaitEnvCfg", "cpr.claas_g1_23dof:G23PlayTurnCfg"),
    "Unitree-G1-Claas-23-Sym": ("cpr.claas_g1_23dof:G23SymEnvCfg", "cpr.claas_g1_23dof:G23PlayTurnCfg"),
    "Unitree-G1-Claas-23-Rough": ("cpr.claas_g1_23dof:G23RoughEnvCfg", "cpr.claas_g1_23dof:G23PlayRoughCfg"),
    "Unitree-G1-Claas-23-StairsA10": ("cpr.claas_g1_23dof:G23StairsA10EnvCfg", "cpr.claas_g1_23dof:G23PlayTurnCfg"),
    "Unitree-G1-Claas-23-RoughSlip": ("cpr.claas_g1_23dof:G23RoughSlipEnvCfg", "cpr.claas_g1_23dof:G23PlayRoughCfg"),
    "Unitree-G1-Claas-23-Slip": ("cpr.claas_g1_23dof:G23SlipEnvCfg", "cpr.claas_g1_23dof:G23PlayTurnCfg"),
    "Unitree-G1-Claas-23-SymOnly": ("cpr.claas_g1_23dof:G23SymOnlyEnvCfg", "cpr.claas_g1_23dof:G23PlayTurnCfg"),
    "Unitree-G1-Claas-23-GaitMod": ("cpr.claas_g1_23dof:G23GaitModEnvCfg", "cpr.claas_g1_23dof:G23PlayTurnCfg"),
    "Unitree-G1-Claas-23-Play-Rough": ("cpr.claas_g1_23dof:G23PlayRoughCfg", "cpr.claas_g1_23dof:G23PlayRoughCfg"),
    "Unitree-G1-Claas-23-Eval-E0_nominal": ("cpr.claas_g1_23dof:G23Eval_E0_nominal", "cpr.claas_g1_23dof:G23PlayTurnCfg"),
    "Unitree-G1-Claas-23-Eval-E1_glatt": ("cpr.claas_g1_23dof:G23Eval_E1_glatt", "cpr.claas_g1_23dof:G23PlayTurnCfg"),
    "Unitree-G1-Claas-23-Eval-E2_stoesse": ("cpr.claas_g1_23dof:G23Eval_E2_stoesse", "cpr.claas_g1_23dof:G23PlayTurnCfg"),
    "Unitree-G1-Claas-23-Eval-E3_latenz": ("cpr.claas_g1_23dof:G23Eval_E3_latenz", "cpr.claas_g1_23dof:G23PlayTurnCfg"),
    "Unitree-G1-Claas-23-Eval-E4_last": ("cpr.claas_g1_23dof:G23Eval_E4_last", "cpr.claas_g1_23dof:G23PlayTurnCfg"),
    "Unitree-G1-Claas-23-Eval-E5_motor": ("cpr.claas_g1_23dof:G23Eval_E5_motor", "cpr.claas_g1_23dof:G23PlayTurnCfg"),
    "Unitree-G1-Claas-23-Eval-E6_kombi": ("cpr.claas_g1_23dof:G23Eval_E6_kombi", "cpr.claas_g1_23dof:G23PlayTurnCfg"),
    "Unitree-G1-Claas-23-Play-FlatPush": ("cpr.claas_g1_23dof:G23PlayFlatPushCfg", "cpr.claas_g1_23dof:G23PlayFlatPushCfg"),
    # Video-Szenarien 29 DoF (jeder Checkpoint aus B1/B2/B3/C passt in jede davon)
    "Unitree-G1-Claas-MS-Play-StairsEasy": (f"{MS}:MsPlayStairsEasyCfg", f"{MS}:MsPlayStairsEasyCfg"),
    "Unitree-G1-Claas-MS-Play-StairsMid": (f"{MS}:MsPlayStairsMidCfg", f"{MS}:MsPlayStairsMidCfg"),
    "Unitree-G1-Claas-MS-Play-FlatPush": (f"{MS}:MsPlayFlatPushCfg", f"{MS}:MsPlayFlatPushCfg"),
}
for task, (env_cfg, play_cfg) in TASKS.items():
    if task in gym.registry:
        continue

    env_cfg_entry = env_cfg if ":" in env_cfg else f"{CFG}:{env_cfg}"
    play_cfg_entry = play_cfg if ":" in play_cfg else f"{CFG}:{play_cfg}"

    gym.register(
        id=task,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry,
            "play_env_cfg_entry_point": play_cfg_entry,
            "rsl_rl_cfg_entry_point": AGENT,
        },
    )

mode = sys.argv[1] if len(sys.argv) > 1 else ""
if mode not in ("train", "play", "script"):
    sys.exit("Aufruf: unitree_run.py {train|play|script <pfad.py>} ...")
if mode == "script":  # eigenes Skript mit allen registrierten Tasks (Claas, 23.11., EDS-Daten)
    script = sys.argv[2]
    sys.argv = [script] + sys.argv[3:]
    sys.path.insert(0, os.path.join(ROOT, "scripts", "rsl_rl"))  # cli_args
else:
    script = os.path.join(ROOT, "scripts", "rsl_rl", f"{mode}.py")
    # Unitrees Skript sieht argv, als waere es direkt aufgerufen; seine eigenen
    # Hilfsmodule (cli_args, list_envs) findet es ueber sein Verzeichnis
    sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
