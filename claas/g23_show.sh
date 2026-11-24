#!/bin/bash
# 23.11. (Claas): Abgabe-Videos Treppen (Isaac, 30 s, Seitenansicht) + MuJoCo-Treppe (Sim-to-Sim)
set -eu
cd /bigwork/nhkbrc2a/isaac
source scripts/job_result.sh
python3 -m py_compile claas/sim2sim/g23_sim2sim.py cpr/claas_g1_23dof_show.py
G="-p gpu --exclude=gpu-20-n[005-008]"
R="--reservation=imes.gpu"
J=runs/claas/g23_jobs.env
echo "# --- Abgabe-Videos Treppe (23.11.) ---" >> $J
A8RUN=$(job_run_dir logs/slurm_claas_ms_9419440.out)
T3RUN=$(job_run_dir logs/slurm_claas_ms_9391391.out)
T3CK=$T3RUN/model_6500.pt
[ -f "$T3CK" ] || { echo "T3b ck6500 fehlt: $T3CK"; exit 2; }
SB="Unitree-G1-Claas-23-Play-ShowStairsEasy Unitree-G1-Claas-23-Play-ShowStairsMid"
ST="Unitree-G1-Claas-23-Teacher-Play-ShowStairsEasy Unitree-G1-Claas-23-Teacher-Play-ShowStairsMid"
VA8=$(sbatch --parsable $R -J g23_showA8_claas --time=01:30:00 --export="ALL,TRAIN_JOB=9419440,TAG=show_A8_9419440,SCENARIOS=$SB,VIDEO_LENGTH=1500" claas/ms_play.sh)
VT3=$(sbatch --parsable $R -J g23_showT3b_claas --time=01:30:00 --export="ALL,TRAIN_JOB=9391391,CKPT=$T3CK,TAG=show_T3b_ck6500,SCENARIOS=$SB,VIDEO_LENGTH=1500" claas/ms_play.sh)
VL1=$(sbatch --parsable $R -J g23_showL1_claas --time=01:30:00 --export="ALL,TRAIN_JOB=9391719,TAG=show_L1_9391719,SCENARIOS=$ST,VIDEO_LENGTH=1500" claas/ms_play.sh)
echo "SHOW_A8=$VA8 SHOW_T3b=$VT3 SHOW_L1=$VL1" | tee -a $J
# MuJoCo-Treppe: Policy aus dem Export des Show-Jobs, deploy.yaml aus dem Trainingslauf
for P in A8:$VA8:$A8RUN T3b:$VT3:$T3RUN; do
  N=${P%%:*}; REST=${P#*:}; V=${REST%%:*}; RUN=${REST#*:}
  for H in 0.05 0.08; do
    X=$(sbatch --parsable $G --dependency=afterany:$V --export="ALL,POLICY=runs/claas/play/${V}_ShowStairsEasy/exported/policy.pt,DEPLOY=$RUN/params/deploy.yaml,TAG=${N}_treppe_${H/0./}cm,S2S_STAIRS=$H" claas/sim2sim/run_sim2sim.sh)
    echo "S2S_TREPPE_${N}_${H}=$X" | tee -a $J
  done
done
echo SHOW OK
