#!/bin/bash
#SBATCH --job-name=g23_s2s_claas
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_claas_s2s_%j.out
# Sim-to-Sim (MuJoCo) fuer eine 23-DoF-Policy - Claas, 19.11.
#   sbatch --reservation=imes.gpu --export=ALL,POLICY=<policy.pt>,DEPLOY=<deploy.yaml>,TAG=<name> claas/sim2sim/run_sim2sim.sh
# oder direkt (ohne Video):  bash claas/sim2sim/run_sim2sim.sh <policy.pt> <deploy.yaml> <tag>
cd /bigwork/nhkbrc2a/isaac
POLICY=${POLICY:-$1}
DEPLOY=${DEPLOY:-$2}
TAG=${TAG:-${3:-test}}
export PYTHONPATH=/bigwork/nhkbrc2a/isaac/claas/ext/pylib
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=$MUJOCO_GL
apptainer exec --nv -B /bigwork containers/isaac-lab-2.3.2.sif /isaac-sim/python.sh \
  claas/sim2sim/g23_sim2sim.py "$POLICY" "$DEPLOY" "$TAG"
