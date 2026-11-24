#!/bin/bash
#SBATCH --job-name=g23_s2s_claas
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_claas_s2s_%j.out
# Sim-to-Sim automatisch nach Training + Video (Claas, 20.11.):
#   sbatch --dependency=afterok:<PLAY_JOB> --export=ALL,TRAIN_JOB=<id>,PLAY_JOB=<id>,TAG=<name> claas/sim2sim/s2s_nach_training.sh
cd /bigwork/nhkbrc2a/isaac
source scripts/job_result.sh
RUN_DIR=$(job_run_dir logs/slurm_claas_ms_${TRAIN_JOB}.out)
POL=$(ls runs/claas/play/${PLAY_JOB}_*/exported/policy.pt 2>/dev/null | head -1)
echo "run=$RUN_DIR policy=$POL"
[ -n "$POL" ] && [ -f "$RUN_DIR/params/deploy.yaml" ] || { echo "FEHLER: policy oder deploy.yaml fehlt"; exit 2; }
export OMP_NUM_THREADS=4
POLICY=$POL DEPLOY=$RUN_DIR/params/deploy.yaml TAG=$TAG bash claas/sim2sim/run_sim2sim.sh
