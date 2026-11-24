#!/bin/bash
# 23.11. (Claas): Deploy A5s2/A8, Robustheitstest A5s2 + A8, A9 glatter Boden
set -eu
cd /bigwork/nhkbrc2a/isaac
source scripts/job_result.sh
G="-p gpu --exclude=gpu-20-n[005-008]"
R="--reservation=imes.gpu"
A5CK=$(job_last_checkpoint "$(job_run_dir logs/slurm_claas_ms_9399921.out)")
A5S2CK=$(job_last_checkpoint "$(job_run_dir logs/slurm_claas_ms_9419448.out)")
A8CK=$(job_last_checkpoint "$(job_run_dir logs/slurm_claas_ms_9419440.out)")
echo "A5=$A5CK A5s2=$A5S2CK A8=$A8CK"
J=runs/claas/g23_jobs.env
echo "# --- Kette 6 (23.11.) ---" >> $J
bash claas/deploy_pakete.sh > runs/claas/deploy_23-11.log 2>&1 || true
for P in A5s2:$A5S2CK A8:$A8CK; do
  N=${P%%:*}; CK=${P#*:}
  E0=$(sbatch --parsable $G -J g23_${N}_E0_nominal --time=00:40:00 --export=ALL,TASK=Unitree-G1-Claas-23-Eval-E0_nominal,INIT_CKPT=$CK,MAX_ITER=60,VIDEO=0 claas/ms_train.sh)
  echo "EVAL_${N}_E0_nominal=$E0" | tee -a $J
  for S in E1_glatt E2_stoesse E3_latenz E4_last E5_motor E6_kombi; do
    X=$(sbatch --parsable $G -J g23_${N}_$S --time=00:40:00 --dependency=afterok:$E0 --export=ALL,TASK=Unitree-G1-Claas-23-Eval-$S,INIT_CKPT=$CK,MAX_ITER=60,VIDEO=0 claas/ms_train.sh)
    echo "EVAL_${N}_$S=$X" | tee -a $J
  done
done
kette() {  # name task ckpt iters scenarios [extra]
  local N=$1 TASK=$2 CK=$3 IT=$4 SC=$5 EX=${6:-}
  local S T V X
  S=$(sbatch --parsable $G -J g23_${N}smoke_claas --time=00:30:00 --export=ALL,TASK=$TASK,INIT_CKPT=$CK,NUM_ENVS=64,MAX_ITER=5,VIDEO=0 claas/ms_train.sh)
  T=$(sbatch --parsable $G -J g23_${N}_claas --time=08:00:00 --dependency=afterok:$S --export="ALL,TASK=$TASK,INIT_CKPT=$CK,MAX_ITER=$IT,VIDEO=0,EXTRA_ARGS=$EX" claas/ms_train.sh)
  V=$(sbatch --parsable $R -J g23_V${N}_claas --dependency=afterany:$T --export="ALL,TRAIN_JOB=$T,TAG=g23_${N}_$T,SCENARIOS=$SC" claas/ms_play.sh)
  X=$(sbatch --parsable $G --dependency=afterok:$V --export=ALL,TRAIN_JOB=$T,PLAY_JOB=$V,TAG=${N}_$T claas/sim2sim/s2s_nach_training.sh)
  printf "%s_SMOKE=%s\n%s=%s\nV%s=%s\nS2S_%s=%s\n" $N $S $N $T $N $V $N $X | tee -a $J
}
kette A9 Unitree-G1-Claas-23-Slip "$A5CK" 3000 "Unitree-G1-Claas-23-Play-Turn Unitree-G1-Claas-23-Play-FlatPush"
echo KETTE6 OK
