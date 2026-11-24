#!/bin/bash
# 23.11. (Claas): Robustheitstest fuer A9 (glatter Boden)
set -eu
cd /bigwork/nhkbrc2a/isaac
source scripts/job_result.sh
G="-p gpu --exclude=gpu-20-n[005-008]"
CK=$(job_last_checkpoint "$(job_run_dir logs/slurm_claas_ms_9425284.out)")
J=runs/claas/g23_jobs.env
E0=$(sbatch --parsable $G -J g23_A9_E0_nominal --time=00:40:00 --export=ALL,TASK=Unitree-G1-Claas-23-Eval-E0_nominal,INIT_CKPT=$CK,MAX_ITER=60,VIDEO=0 claas/ms_train.sh)
echo "EVAL_A9_E0_nominal=$E0" | tee -a $J
for S in E1_glatt E2_stoesse E3_latenz E4_last E5_motor E6_kombi; do
  X=$(sbatch --parsable $G -J g23_A9_$S --time=00:40:00 --dependency=afterok:$E0 --export=ALL,TASK=Unitree-G1-Claas-23-Eval-$S,INIT_CKPT=$CK,MAX_ITER=60,VIDEO=0 claas/ms_train.sh)
  echo "EVAL_A9_$S=$X" | tee -a $J
done
grep PRUEF runs/claas/deploy_23-11.log > claas/deploy/PRUEFUNG.txt || true
