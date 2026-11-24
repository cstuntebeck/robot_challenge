#!/bin/bash
# 24.11. Nacht (Claas): A10 = A8 + Treppen, A11 = A8 + glatter Boden; je Smoke -> Training -> Videos -> MuJoCo (+Treppe) / Robustheitstest
set -eu
cd /bigwork/nhkbrc2a/isaac
source scripts/job_result.sh
G="-p gpu --exclude=gpu-20-n[005-008]"
V="-p imes.gpu"
A8RUN=$(job_run_dir logs/slurm_claas_ms_9419440.out)
A8CK=$(job_last_checkpoint "$A8RUN")
J=runs/claas/g23_jobs.env
echo "# --- Kette 7 (24.11. Nacht) ---" >> $J
kette() {  # name task iters scenarios
  local N=$1 TASK=$2 IT=$3 SC=$4
  local S T VJ X
  S=$(sbatch --parsable $G -J g23_${N}smoke_claas --time=00:30:00 --export=ALL,TASK=$TASK,INIT_CKPT=$A8CK,NUM_ENVS=64,MAX_ITER=5,VIDEO=0 claas/ms_train.sh)
  T=$(sbatch --parsable $G -J g23_${N}_claas --time=08:00:00 --dependency=afterok:$S --export="ALL,TASK=$TASK,INIT_CKPT=$A8CK,MAX_ITER=$IT,VIDEO=0" claas/ms_train.sh)
  VJ=$(sbatch --parsable $V -J g23_V${N}_claas --time=01:30:00 --dependency=afterany:$T --export="ALL,TRAIN_JOB=$T,TAG=g23_${N}_$T,SCENARIOS=$SC,VIDEO_LENGTH=2000" claas/ms_play.sh)
  X=$(sbatch --parsable $G --dependency=afterok:$VJ --export=ALL,TRAIN_JOB=$T,PLAY_JOB=$VJ,TAG=${N}_$T claas/sim2sim/s2s_nach_training.sh)
  printf "%s_SMOKE=%s\n%s=%s\nV%s=%s\nS2S_%s=%s\n" $N $S $N $T $N $VJ $N $X | tee -a $J
  LAST_T=$T; LAST_V=$VJ
}
kette A10 Unitree-G1-Claas-23-StairsA10 4000 "Unitree-G1-Claas-23-Play-ShowStairsEasy Unitree-G1-Claas-23-Play-ShowStairsMid Unitree-G1-Claas-23-Play-Turn"
T10=$LAST_T; V10=$LAST_V
for H in 0.05 0.08 0.10; do
  X=$(sbatch --parsable $G --dependency=afterok:$V10 --export="ALL,POLICY=runs/claas/play/${V10}_ShowStairsEasy/exported/policy.pt,DEPLOY=$A8RUN/params/deploy.yaml,TAG=A10_treppe_${H/0./}cm,S2S_STAIRS=$H" claas/sim2sim/run_sim2sim.sh)
  echo "S2S_TREPPE_A10_$H=$X" | tee -a $J
done
kette A11 Unitree-G1-Claas-23-RoughSlip 3000 "Unitree-G1-Claas-23-Play-Turn Unitree-G1-Claas-23-Play-FlatPush"
T11=$LAST_T
# Robustheitstest A11 (E0-E6): morgens per Hand (braucht den fertigen Checkpoint)
echo KETTE7 OK
