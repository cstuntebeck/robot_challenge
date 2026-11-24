#!/bin/bash
#SBATCH --job-name=isaac_train
#SBATCH --partition=imes.gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_train_%j.out

# Aufruf:
#   sbatch --export=ALL,TASK=Isaac-Velocity-Rough-G1-v0,NUM_ENVS=2048,MAX_ITER=3000 submit_train.sh
# Optional: VIDEO=1 zeichnet Rollouts auf (Deliverable "Video vom laufenden Beispiel").
#
# Faellt der Job innerhalb der imes.gpu-FLEX-Reservierung (werktags 8-20 Uhr) auf
# ReqNodeNotAvail, zusaetzlich --reservation=imes.gpu mitgeben.
#
# Erfolgreich ist der Job nur, wenn train.py mit 0 endet UND der finale
# Checkpoint auf BIGWORK liegt. Sonst endet er mit Exit != 0, und sacct sagt
# FAILED - nicht mehr wie frueher immer COMPLETED 0:0. Der Checkpoint ist dabei
# das eigentliche Kriterium: train.py endet auch nach einer Python-Exception mit
# 0 (Negativtest 8374044: NameNotFound, "ATTEMPT 1 exit=0").
# Ergebnis: runs/results/<jobid>.json (gelesen von scripts/verify.py).
set -u
BASE=/bigwork/nhkbrc2a/isaac
RUNS=$BASE/runs
LOG=$BASE/logs/slurm_train_${SLURM_JOB_ID}.out
HELPER=$BASE/scripts/job_result.sh
[ -f "$HELPER" ] || { echo "FEHLER: $HELPER fehlt - erst cluster/scripts hochladen"; exit 2; }
source "$HELPER"

TASK=${TASK:-Isaac-Velocity-Rough-G1-v0}
NUM_ENVS=${NUM_ENVS:-2048}
MAX_ITER=${MAX_ITER:-3000}
VIDEO=${VIDEO:-0}

echo "=== node: $(hostname) | task=$TASK envs=$NUM_ENVS iters=$MAX_ITER video=$VIDEO"
nvidia-smi -L

export APPTAINERENV_PYTHONPATH=$BASE
RUN=$BASE/scripts/run_isaaclab.sh
ASSET_ROOT=$(dirname "$(find "$BASE/assets" -maxdepth 4 -type d -name NVIDIA | head -1)")
KIT_ARGS="--/persistent/isaac/asset_root/default=$ASSET_ROOT"
KIT_ARGS="$KIT_ARGS --/persistent/isaac/asset_root/cloud=$ASSET_ROOT"
KIT_ARGS="$KIT_ARGS --/persistent/isaac/asset_root/nvidia=$ASSET_ROOT"

EXTRA=""
if [ "$VIDEO" = "1" ]; then
  # Compute-Knoten sind headless; --video rendert offscreen nach <laufverzeichnis>/videos/train/
  EXTRA="--video --video_length 400 --video_interval 500"
fi

# train.py schreibt logs/rsl_rl/... relativ zum Arbeitsverzeichnis. Ohne dieses
# cd waere das $HOME - im Container ein Wegwerf-Verzeichnis, alle Checkpoints
# waeren nach Jobende weg.
mkdir -p "$RUNS"
cd "$RUNS" || exit 2

# Retry-Schleife: der DISPLAY-/Shared-Memory-Segfault schlaegt immer INNERHALB
# der ersten ~180 s zu. Spaetere Abbrueche sind echte Fehler und werden nicht
# wiederholt.
for attempt in 1 2 3; do
  t0=$SECONDS
  bash $RUN /isaac-sim/python.sh /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$TASK" --headless --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITER" \
    $EXTRA --kit_args "$KIT_ARGS"
  rc=$?
  dt=$((SECONDS - t0))
  echo "ATTEMPT $attempt exit=$rc after ${dt}s"
  [ $rc -eq 0 ] && break
  if [ $dt -ge 180 ]; then
    echo "Abbruch nach ${dt}s - kein Startup-Segfault, kein Retry."
    break
  fi
  echo "Startup-Absturz, Versuch $((attempt + 1)) ..."
  sleep 10
done
echo "TRAIN_EXIT: $rc"

# Hat das Ergebnis den Job ueberlebt, und ist es vollstaendig?
RUN_DIR=$(job_run_dir "$LOG")
LAST_CKPT=""
LAST_ITER=-1
[ -n "$RUN_DIR" ] && LAST_CKPT=$(job_last_checkpoint "$RUN_DIR")
[ -n "$LAST_CKPT" ] && LAST_ITER=$(job_checkpoint_iter "$LAST_CKPT")
VIDEOS=0
[ -n "$RUN_DIR" ] && VIDEOS=$(ls -1 "$RUN_DIR"/videos/train/*.mp4 2>/dev/null | wc -l | tr -d " ")
echo "TRAIN_RUN_DIR: ${RUN_DIR:-<nicht im Log gefunden>}"
echo "TRAIN_CHECKPOINT: ${LAST_CKPT:-<fehlt>} (Iteration $LAST_ITER von $MAX_ITER)"
if [ $rc -eq 0 ]; then
  if [ -z "$LAST_CKPT" ]; then
    echo "FEHLER: train.py endete mit 0, aber kein model_*.pt unter ${RUN_DIR:-?} - Ausgaben verloren."
    rc=3
  elif [ "$LAST_ITER" -lt $((MAX_ITER - 1)) ]; then
    echo "FEHLER: letzter Checkpoint ist Iteration $LAST_ITER, erwartet mindestens $((MAX_ITER - 1))."
    rc=3
  fi
fi

job_write_result "$RUNS/results/${SLURM_JOB_ID}.json" \
  kind=train job_id="$SLURM_JOB_ID" task="$TASK" num_envs="$NUM_ENVS" \
  max_iter="$MAX_ITER" video="$VIDEO" exit_code="$rc" attempts="$attempt" \
  run_dir="$RUN_DIR" last_checkpoint="$LAST_CKPT" last_iter="$LAST_ITER" \
  videos="$VIDEOS" node="$(hostname)" finished="$(date -Iseconds)"
echo "TRAIN_DONE"
exit $rc
