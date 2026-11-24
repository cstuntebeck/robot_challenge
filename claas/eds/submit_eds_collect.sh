#!/bin/bash
#SBATCH --job-name=g23_eds_collect_claas
#SBATCH --partition=imes.gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_claas_eds_%j.out
# CD2 (Claas, 23.11.): EDS-Datensatz aufnehmen (Kamera + LiDAR + exakte Hoehenkarte), braucht RT-Kerne (L40S).
# Parameter (--export): TASKS (Play-Tasks des Lehrers), CKPT, NUM_ENVS (16), SIM_SECONDS (300), EXTRA
set -u
BASE=/bigwork/nhkbrc2a/isaac
RUNS=$BASE/runs
TASKS=${TASKS:-Unitree-G1-Claas-23-Teacher-Play-StairsEasy Unitree-G1-Claas-23-Teacher-Play-StairsMid}
CKPT=${CKPT:?CKPT fehlt}
NUM_ENVS=${NUM_ENVS:-16}
SECONDS_SIM=${SIM_SECONDS:-300}
EXTRA=${EXTRA:-}
echo "=== node: $(hostname) | ckpt=$CKPT | tasks=$TASKS"
nvidia-smi -L
UNITREE=$BASE/vendor/unitree_rl_lab
export APPTAINERENV_UNITREE_RL_LAB=$UNITREE
export APPTAINERENV_PYTHONPATH=$BASE:$UNITREE/source/unitree_rl_lab:$BASE/vendor/pylib
RUN=$BASE/scripts/run_isaaclab.sh
ASSET_ROOT=$(dirname "$(find "$BASE/assets" -maxdepth 4 -type d -name NVIDIA | head -1)")
KIT_ARGS="--/persistent/isaac/asset_root/default=$ASSET_ROOT --/persistent/isaac/asset_root/cloud=$ASSET_ROOT --/persistent/isaac/asset_root/nvidia=$ASSET_ROOT"
OUTBASE=$RUNS/eds_data/$SLURM_JOB_ID
mkdir -p "$OUTBASE"
cd "$RUNS" || exit 2
rc_all=0
for T in $TASKS; do
  short=${T##*-Play-}
  for attempt in 1 2 3; do
    t0=$SECONDS
    bash "$RUN" /isaac-sim/python.sh "$BASE/claas/unitree_run.py" script "$BASE/claas/eds/eds_collect.py" \
      --task "$T" --checkpoint "$CKPT" --out "$OUTBASE/$short" --num_envs "$NUM_ENVS" --seconds "$SECONDS_SIM" \
      --kit_args "$KIT_ARGS" $EXTRA
    rc=$?
    dt=$((SECONDS - t0))
    echo "EDS $short attempt $attempt exit=$rc after ${dt}s"
    [ -f "$OUTBASE/$short/meta.json" ] && break
    [ $dt -ge 240 ] && break
    sleep 10
  done
  if [ -f "$OUTBASE/$short/meta.json" ]; then
    echo "EDS_OK: $OUTBASE/$short ($(du -sh "$OUTBASE/$short" | cut -f1))"
  else
    echo "FEHLER: kein Datensatz fuer $short"; rc_all=3
  fi
done
echo "EDS_DONE rc=$rc_all"
exit $rc_all
