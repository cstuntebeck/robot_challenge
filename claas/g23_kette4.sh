#!/bin/bash
# 22.11. (Claas): Robustheitstest A5 (Isaac + MuJoCo), A7 Symmetrie, A8 unebener Boden, A5-Seeds
set -eu
cd /bigwork/nhkbrc2a/isaac
source scripts/job_result.sh
G="-p gpu --exclude=gpu-20-n[005-008]"
R="--reservation=imes.gpu"
A4CK=$(job_last_checkpoint "$(job_run_dir logs/slurm_claas_ms_9391705.out)")
A5RUN=$(job_run_dir logs/slurm_claas_ms_9399921.out)
A5CK=$(job_last_checkpoint "$A5RUN")
echo "A4=$A4CK"; echo "A5=$A5CK"
J=runs/claas/g23_jobs.env
echo "# --- Kette 4 (22.11.) ---" >> $J

# 1) Robustheitstest Isaac: 4096 Roboter, 60 Iterationen ohne Policy-Update je Szenario
E0=$(sbatch --parsable $G -J g23_E0_nominal --time=00:40:00 --export=ALL,TASK=Unitree-G1-Claas-23-Eval-E0_nominal,INIT_CKPT=$A5CK,MAX_ITER=60,VIDEO=0 claas/ms_train.sh)
echo "EVAL_E0_nominal=$E0" | tee -a $J
for S in E1_glatt E2_stoesse E3_latenz E4_last E5_motor E6_kombi; do
  X=$(sbatch --parsable $G -J g23_$S --time=00:40:00 --dependency=afterok:$E0 --export=ALL,TASK=Unitree-G1-Claas-23-Eval-$S,INIT_CKPT=$A5CK,MAX_ITER=60,VIDEO=0 claas/ms_train.sh)
  echo "EVAL_$S=$X" | tee -a $J
done

# 2) Robustheitstest MuJoCo (A5-Policy, eigener Ablauf mit Stoerungen)
M=$(sbatch --parsable $G --time=01:00:00 --export=ALL,A5RUN=$A5RUN claas/sim2sim/robust_mujoco.sh)
echo "S2S_ROBUST_A5=$M" | tee -a $J

kette() {  # name task ckpt iters scenarios [extra]
  local N=$1 TASK=$2 CK=$3 IT=$4 SC=$5 EX=${6:-}
  local S T V X
  S=$(sbatch --parsable $G -J g23_${N}smoke_claas --time=00:30:00 --export=ALL,TASK=$TASK,INIT_CKPT=$CK,NUM_ENVS=64,MAX_ITER=5,VIDEO=0 claas/ms_train.sh)
  T=$(sbatch --parsable $G -J g23_${N}_claas --time=08:00:00 --dependency=afterok:$S --export="ALL,TASK=$TASK,INIT_CKPT=$CK,MAX_ITER=$IT,VIDEO=0,EXTRA_ARGS=$EX" claas/ms_train.sh)
  V=$(sbatch --parsable $R -J g23_V${N}_claas --dependency=afterany:$T --export="ALL,TRAIN_JOB=$T,TAG=g23_${N}_$T,SCENARIOS=$SC" claas/ms_play.sh)
  X=$(sbatch --parsable $G --dependency=afterok:$V --export=ALL,TRAIN_JOB=$T,PLAY_JOB=$V,TAG=${N}_$T claas/sim2sim/s2s_nach_training.sh)
  printf "%s_SMOKE=%s\n%s=%s\nV%s=%s\nS2S_%s=%s\n" $N $S $N $T $N $V $N $X | tee -a $J
}
PT="Unitree-G1-Claas-23-Play-Turn Unitree-G1-Claas-23-Play-FlatPush Unitree-G1-Claas-23-Play-StairsEasy"
# 3) A7 Symmetrie + Gangbild moderat (ab A5)
kette A7 Unitree-G1-Claas-23-Sym "$A5CK" 3000 "$PT"
# 4) A8 unebener Boden (ab A5)
kette A8 Unitree-G1-Claas-23-Rough "$A5CK" 4000 "Unitree-G1-Claas-23-Play-Rough $PT"
# 5) A5-Rezept mit anderen Seeds (ab A4, gleiche Task wie A5) -> Reproduzierbarkeit
kette A5s1 Unitree-G1-Claas-23-RobustTurn "$A4CK" 3000 "Unitree-G1-Claas-23-Play-Turn" "--seed 1"
kette A5s2 Unitree-G1-Claas-23-RobustTurn "$A4CK" 3000 "Unitree-G1-Claas-23-Play-Turn" "--seed 2"
echo KETTE4 OK
