#!/bin/bash
# 22.11. abends (Claas): A5s1 neu (alter Lauf haengt nach Vulkan DEVICE_LOST), A7-Ablation: A7b nur Symmetrie, A7c nur Gangbild
set -eu
cd /bigwork/nhkbrc2a/isaac
source scripts/job_result.sh
scancel 9419444 9419445 9419446 || true
G="-p gpu --exclude=gpu-20-n[005-008],gpu-25-n003"
R="--reservation=imes.gpu"
A4CK=$(job_last_checkpoint "$(job_run_dir logs/slurm_claas_ms_9391705.out)")
A5CK=$(job_last_checkpoint "$(job_run_dir logs/slurm_claas_ms_9399921.out)")
J=runs/claas/g23_jobs.env
echo "# --- Kette 5 (22.11. abends) ---" >> $J
kette() {  # name task ckpt iters scenarios [extra]
  local N=$1 TASK=$2 CK=$3 IT=$4 SC=$5 EX=${6:-}
  local S T V X
  S=$(sbatch --parsable $G -J g23_${N}smoke_claas --time=00:30:00 --export=ALL,TASK=$TASK,INIT_CKPT=$CK,NUM_ENVS=64,MAX_ITER=5,VIDEO=0 claas/ms_train.sh)
  T=$(sbatch --parsable $G -J g23_${N}_claas --time=08:00:00 --dependency=afterok:$S --export="ALL,TASK=$TASK,INIT_CKPT=$CK,MAX_ITER=$IT,VIDEO=0,EXTRA_ARGS=$EX" claas/ms_train.sh)
  V=$(sbatch --parsable $R -J g23_V${N}_claas --dependency=afterany:$T --export="ALL,TRAIN_JOB=$T,TAG=g23_${N}_$T,SCENARIOS=$SC" claas/ms_play.sh)
  X=$(sbatch --parsable $G --dependency=afterok:$V --export=ALL,TRAIN_JOB=$T,PLAY_JOB=$V,TAG=${N}_$T claas/sim2sim/s2s_nach_training.sh)
  printf "%s_SMOKE=%s\n%s=%s\nV%s=%s\nS2S_%s=%s\n" $N $S $N $T $N $V $N $X | tee -a $J
}
PT="Unitree-G1-Claas-23-Play-Turn Unitree-G1-Claas-23-Play-FlatPush"
kette A5s1b Unitree-G1-Claas-23-RobustTurn "$A4CK" 3000 "Unitree-G1-Claas-23-Play-Turn" "--seed 1"
kette A7b Unitree-G1-Claas-23-SymOnly "$A5CK" 3000 "$PT"
kette A7c Unitree-G1-Claas-23-GaitMod "$A5CK" 3000 "$PT"
echo KETTE5 OK
