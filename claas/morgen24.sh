#!/bin/bash
# 24.11. morgens (Claas): Kurzauswertung A10/A11, Robustheitstest A11, Kopplung mit EDS neu
cd /bigwork/nhkbrc2a/isaac
source scripts/job_result.sh
O=runs/claas/morgen24.txt; : > $O
for N in A10:9432211 A11:9432218; do K=${N%%:*}; J=${N##*:}
  echo "== $K" >> $O
  tail -c 9000 logs/slurm_claas_ms_$J.out | grep -E "Learning iteration|track_ang|error_vel|terrain_levels|bad_orientation" | tail -6 | tr -s ' ' >> $O
done
G="-p gpu --exclude=gpu-20-n[005-008]"
CK=$(job_last_checkpoint "$(job_run_dir logs/slurm_claas_ms_9432218.out)")
J=runs/claas/g23_jobs.env
E0=$(sbatch --parsable $G -J g23_A11_E0_nominal --time=00:40:00 --export=ALL,TASK=Unitree-G1-Claas-23-Eval-E0_nominal,INIT_CKPT=$CK,MAX_ITER=60,VIDEO=0 claas/ms_train.sh)
echo "EVAL_A11_E0_nominal=$E0" | tee -a $J >> $O
for S in E1_glatt E2_stoesse E3_latenz E4_last E5_motor E6_kombi; do
  X=$(sbatch --parsable $G -J g23_A11_$S --time=00:40:00 --dependency=afterok:$E0 --export=ALL,TASK=Unitree-G1-Claas-23-Eval-$S,INIT_CKPT=$CK,MAX_ITER=60,VIDEO=0 claas/ms_train.sh)
  echo "EVAL_A11_$S=$X" | tee -a $J >> $O
done
L3CK=$(job_last_checkpoint "$(job_run_dir logs/slurm_claas_ms_9426234.out)")
C=$(sbatch --parsable --reservation=imes.gpu --time=03:00:00 --export=ALL,CKPT=$L3CK,MODELS=runs/eds_models/v1 claas/eds/submit_eds_couple.sh)
echo "COUPLE_FULL2=$C" | tee -a $J >> $O
