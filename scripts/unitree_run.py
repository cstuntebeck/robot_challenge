"""Starter fuer Unitree RL Lab (T-004): eigene Tasks registrieren, dann Unitrees
train.py oder play.py unveraendert ausfuehren.

Aufruf im Container:
    /isaac-sim/python.sh unitree_run.py {train|play} <Argumente fuer Unitrees Skript>

Warum ein Starter: Unitrees Skripte nehmen bei --task nur Namen an, die schon vor
dem Parsen registriert sind (scripts/rsl_rl/train.py:21-24, :41). Die Tasks fuer
Lauf B liegen in unserem Paket cpr, nicht in Unitrees Baum - sie werden hier
registriert, mit Zeichenketten-Einstiegspunkten, damit vor dem App-Start nichts
aus isaaclab importiert wird. Namen muessen "Unitree" enthalten (Filter :22).
"""

import os
import runpy
import sys

import gymnasium as gym

ROOT = os.environ["UNITREE_RL_LAB"]  # hochgeladene Kopie, siehe scripts/vendor_unitree.sh
CFG = "cpr.unitree_g1_perceptive"
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
if mode not in ("train", "play"):
    sys.exit("Aufruf: unitree_run.py {train|play} ...")
script = os.path.join(ROOT, "scripts", "rsl_rl", f"{mode}.py")
# Unitrees Skript sieht argv, als waere es direkt aufgerufen; seine eigenen
# Hilfsmodule (cli_args, list_envs) findet es ueber sein Verzeichnis
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
