#!/bin/bash
# 23.11. (Claas, CD2): L3 Lehrer im EDS-Format (geschaetzte Hoehenkarte) + L3x (exakte Karte), Start A8
set -eu
cd /bigwork/nhkbrc2a/isaac
source scripts/job_result.sh
G="-p gpu --exclude=gpu-20-n[005-008]"
R="--reservation=imes.gpu"
A8CK=$(job_last_checkpoint "$(job_run_dir logs/slurm_claas_ms_9419440.out)")
J=runs/claas/g23_jobs.env
echo "# --- CD2 L3 (23.11.) ---" >> $J
PS="Unitree-G1-Claas-23-TeacherEds-Play-StairsEasy Unitree-G1-Claas-23-TeacherEds-Play-StairsMid Unitree-G1-Claas-23-TeacherEds-Play-FlatPush"
for P in L3:Unitree-G1-Claas-23-TeacherEds L3x:Unitree-G1-Claas-23-TeacherEdsExact; do
  N=${P%%:*}; TASK=${P#*:}
  S=$(sbatch --parsable $G -J g23_${N}smoke_claas --time=00:30:00 --export=ALL,TASK=$TASK,INIT_CKPT=$A8CK,NUM_ENVS=64,MAX_ITER=5,VIDEO=0 claas/ms_train.sh)
  T=$(sbatch --parsable $G -J g23_${N}_claas --time=08:00:00 --dependency=afterok:$S --export=ALL,TASK=$TASK,INIT_CKPT=$A8CK,MAX_ITER=6000,VIDEO=0 claas/ms_train.sh)
  V=$(sbatch --parsable $R -J g23_V${N}_claas --dependency=afterany:$T --export="ALL,TRAIN_JOB=$T,TAG=g23_${N}_$T,SCENARIOS=$PS" claas/ms_play.sh)
  printf "%s_SMOKE=%s\n%s=%s\nV%s=%s\n" $N $S $N $T $N $V | tee -a $J
done
echo KETTE_L3 OK
