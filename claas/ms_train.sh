#!/bin/bash
#SBATCH --job-name=claas_ms_train
#SBATCH --partition=imes.gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_claas_ms_%j.out

# Meilenstein (Claas): Training, das an einem vorhandenen Checkpoint WEITERtrainiert.
# Abgeleitet von scripts/submit_train_unitree.sh (dort unveraendert).
#
# Parameter (--export):
#   TASK            registrierter Task aus claas/unitree_run.py
#   INIT_CKPT       absoluter Pfad zum Start-Checkpoint (model_*.pt)
#   FROM_SELECTION  =1: TASK und INIT_CKPT aus runs/claas/ms_auswahl.env (Lauf C)
#   NUM_ENVS        Default 4096
#   MAX_ITER        zusaetzliche Iterationen, Default 3000
#   VIDEO_EVERY     Trainingsvideo alle N Schritte (24 Schritte = 1 Iteration), Default 24000
#
# Der Start-Checkpoint wird nach runs/logs/rsl_rl/<experiment>/init_<jobid>/ kopiert,
# weil Unitrees train.py --load_run nur im Experiment-Ordner des Tasks sucht.
set -u
BASE=/bigwork/nhkbrc2a/isaac
RUNS=$BASE/runs
LOG=$BASE/logs/slurm_claas_ms_${SLURM_JOB_ID}.out
source "$BASE/scripts/job_result.sh"

if [ "${FROM_SELECTION:-0}" = "1" ]; then
  SEL=$RUNS/claas/ms_auswahl.env
  [ -f "$SEL" ] || { echo "FEHLER: $SEL fehlt - Auswahl-Job gelaufen?"; exit 2; }
  source "$SEL"
  TASK=$C_TASK
  INIT_CKPT=$C_INIT_CKPT
  echo "=== Auswahl: $C_REASON"
fi
TASK=${TASK:?TASK fehlt}
INIT_CKPT=${INIT_CKPT:-none}   # none = Training ab null
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITER=${MAX_ITER:-3000}
VIDEO_EVERY=${VIDEO_EVERY:-24000}
VIDEO=${VIDEO:-1}  # 0 auf GPUs ohne RT-Kerne (A100/H100/H200): dort keine Kamerabilder
VIDEO_ARGS=""
[ "$VIDEO" = "1" ] && VIDEO_ARGS="--video --video_length 300 --video_interval $VIDEO_EVERY"
[ "$INIT_CKPT" = "none" ] || [ -f "$INIT_CKPT" ] || { echo "FEHLER: Start-Checkpoint $INIT_CKPT fehlt"; exit 2; }

echo "=== node: $(hostname) | task=$TASK envs=$NUM_ENVS iters=+$MAX_ITER init=$INIT_CKPT"
nvidia-smi -L

UNITREE=$BASE/vendor/unitree_rl_lab
export APPTAINERENV_UNITREE_RL_LAB=$UNITREE
export APPTAINERENV_PYTHONPATH=$BASE:$UNITREE/source/unitree_rl_lab:$BASE/vendor/pylib
RUN=$BASE/scripts/run_isaaclab.sh
ASSET_ROOT=$(dirname "$(find "$BASE/assets" -maxdepth 4 -type d -name NVIDIA | head -1)")
KIT_ARGS="--/persistent/isaac/asset_root/default=$ASSET_ROOT"
KIT_ARGS="$KIT_ARGS --/persistent/isaac/asset_root/cloud=$ASSET_ROOT"
KIT_ARGS="$KIT_ARGS --/persistent/isaac/asset_root/nvidia=$ASSET_ROOT"

EXP=$(echo "$TASK" | tr 'A-Z-' 'a-z_')
INIT_RUN=init_${SLURM_JOB_ID}
RESUME_ARGS=""
if [ "$INIT_CKPT" != "none" ]; then
  mkdir -p "$RUNS/logs/rsl_rl/$EXP/$INIT_RUN"
  cp "$INIT_CKPT" "$RUNS/logs/rsl_rl/$EXP/$INIT_RUN/model_init.pt"
  echo "$INIT_CKPT" > "$RUNS/logs/rsl_rl/$EXP/$INIT_RUN/QUELLE.txt"
  RESUME_ARGS="--resume --load_run $INIT_RUN --checkpoint model_init.pt"
fi

mkdir -p "$RUNS"
cd "$RUNS" || exit 2

for attempt in 1 2 3; do
  t0=$SECONDS
  bash $RUN /isaac-sim/python.sh $BASE/claas/unitree_run.py train \
    --task "$TASK" --headless --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITER" \
    $RESUME_ARGS \
    ${EXTRA_ARGS:-} \
    $VIDEO_ARGS \
    --kit_args "$KIT_ARGS"
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

RUN_DIR=$(job_run_dir "$LOG")
LAST_CKPT=""
LAST_ITER=-1
[ -n "$RUN_DIR" ] && LAST_CKPT=$(job_last_checkpoint "$RUN_DIR")
[ -n "$LAST_CKPT" ] && LAST_ITER=$(job_checkpoint_iter "$LAST_CKPT")
LOADED=$(grep -a 'Loading model checkpoint from:' "$LOG" | tail -n 1 | sed 's/.*from: *//')
echo "TRAIN_RUN_DIR: ${RUN_DIR:-<nicht im Log gefunden>}"
echo "TRAIN_INIT: ${LOADED:-<NICHT GELADEN>}"
echo "TRAIN_CHECKPOINT: ${LAST_CKPT:-<fehlt>} (Iteration $LAST_ITER)"
if [ $rc -eq 0 ]; then
  if [ -z "$LAST_CKPT" ]; then
    echo "FEHLER: kein model_*.pt unter ${RUN_DIR:-?}"; rc=3
  elif [ "$INIT_CKPT" != "none" ] && [ -z "$LOADED" ]; then
    echo "FEHLER: Start-Checkpoint wurde nicht geladen"; rc=3
  fi
fi

job_write_result "$RUNS/results/${SLURM_JOB_ID}.json" \
  kind=claas_ms_train job_id="$SLURM_JOB_ID" task="$TASK" init="$INIT_CKPT" num_envs="$NUM_ENVS" \
  add_iter="$MAX_ITER" exit_code="$rc" attempts="$attempt" run_dir="$RUN_DIR" \
  last_checkpoint="$LAST_CKPT" last_iter="$LAST_ITER" node="$(hostname)" finished="$(date -Iseconds)"
echo "TRAIN_DONE"
exit $rc
