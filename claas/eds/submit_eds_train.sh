#!/bin/bash
#SBATCH --job-name=g23_eds_train_claas
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_claas_edstrain_%j.out
# CD2 (Claas, 23.11.): EDS trainieren. --export: DATA (Ordner), OUT, STAGE, MODAL, EPOCHS
cd /bigwork/nhkbrc2a/isaac
echo "=== node: $(hostname) | stage=$STAGE modal=${MODAL:-fusion} data=$DATA out=$OUT"
nvidia-smi -L
apptainer exec --nv -B /bigwork containers/isaac-lab-2.3.2.sif /isaac-sim/python.sh claas/eds/eds_train.py \
  --data $DATA --out "$OUT" --stage "$STAGE" --modal "${MODAL:-fusion}" --epochs "${EPOCHS:-30}" --state_cols "${STATE_COLS:-all}"
echo "EDSTRAIN_DONE rc=$?"
