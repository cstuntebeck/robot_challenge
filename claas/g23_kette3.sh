#!/bin/bash
# 20.11. (Claas): drei Laeufe parallel, jeder: Smoke -> Training -> Videos -> (Sim-to-Sim)
#   A5  eben robust + Drehen lernen         (Start: A4)
#   T4  blinde Treppen robust + Drehen      (Start: T3b ck6500)
#   L2  Lehrer robust + Drehen              (Start: L1)
# Alle mit neuem Reward-/Beobachtungs-Schutz (cpr/claas_milestone.py).
set -eu
cd /bigwork/nhkbrc2a/isaac
source scripts/job_result.sh
G="-p gpu --exclude=gpu-20-n[005-008]"
R="--reservation=imes.gpu"
CK_A4=$(job_last_checkpoint "$(job_run_dir logs/slurm_claas_ms_9391705.out)")
CK_T3=$PWD/runs/logs/rsl_rl/unitree_g1_claas_23_stairstrack/2026-11-19_21-19-31/model_6500.pt
CK_L1=$(job_last_checkpoint "$(job_run_dir logs/slurm_claas_ms_9391719.out)")
echo "A4=$CK_A4"; echo "T3b=$CK_T3"; echo "L1=$CK_L1"
P23="Unitree-G1-Claas-23-Play-StairsEasy Unitree-G1-Claas-23-Play-StairsMid Unitree-G1-Claas-23-Play-FlatPush"
PL="Unitree-G1-Claas-23-Teacher-Play-StairsEasy Unitree-G1-Claas-23-Teacher-Play-StairsMid Unitree-G1-Claas-23-Teacher-Play-FlatPush"

kette() {  # name task ckpt iters scenarios s2s(0/1)
  local N=$1 TASK=$2 CK=$3 IT=$4 SC=$5 S2S=$6
  local S T V X=""
  S=$(sbatch --parsable $G -J g23_${N}smoke_claas --time=00:30:00 --export=ALL,TASK=$TASK,INIT_CKPT=$CK,NUM_ENVS=64,MAX_ITER=5,VIDEO=0 claas/ms_train.sh)
  T=$(sbatch --parsable $G -J g23_${N}_claas --time=08:00:00 --dependency=afterok:$S --export=ALL,TASK=$TASK,INIT_CKPT=$CK,MAX_ITER=$IT,VIDEO=0 claas/ms_train.sh)
  V=$(sbatch --parsable $R -J g23_V${N}_claas --dependency=afterany:$T --export="ALL,TRAIN_JOB=$T,TAG=g23_${N}_$T,SCENARIOS=$SC" claas/ms_play.sh)
  [ "$S2S" = 1 ] && X=$(sbatch --parsable $G --dependency=afterok:$V --export=ALL,TRAIN_JOB=$T,PLAY_JOB=$V,TAG=${N}_$T claas/sim2sim/s2s_nach_training.sh)
  printf "%s_SMOKE=%s\n%s=%s\nV%s=%s\nS2S_%s=%s\n" $N $S $N $T $N $V $N "$X" | tee -a runs/claas/g23_jobs.env
}
kette A5 Unitree-G1-Claas-23-RobustTurn "$CK_A4" 3000 "Unitree-G1-Claas-23-Play-Turn Unitree-G1-Claas-23-Play-FlatPush Unitree-G1-Claas-23-Play-StairsEasy" 1
kette T4 Unitree-G1-Claas-23-StairsRobust "$CK_T3" 4000 "$P23 Unitree-G1-Claas-23-Play-Turn" 1
kette L2 Unitree-G1-Claas-23-TeacherRobust "$CK_L1" 4000 "$PL" 0
echo KETTE3 OK
