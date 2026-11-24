#!/bin/bash
#SBATCH --job-name=kurs_train
#SBATCH --partition=imes.gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_train_kurs_%j.out

# Kurs (T-005): ein Training der gelaendebewussten Pipeline B mit den Rewards
# einer Kursgruppe. Kopie von submit_train_unitree.sh; anders sind nur TEAM, die
# Vorpruefung der Gruppendatei und der Aufruf ueber kurs/_intern/run.py.
#
# Aufruf (eine Gruppe = ein Job):
#   ./bin/cluster submit cluster/scripts/submit_train_kurs.sh --env TEAM=blue --note "Kurs: Gruppe blue"
# Gruppendatei: $BASE/kurs/teams/$TEAM.py (vorher `cluster push kurs kurs`).
# Training ohne Rendering darf auf die H200 (CLAUDE.md, RT-Regel):
#   --partition gpu --gres gpu:h200:1
# Faellt der Job auf imes.gpu in die FLEX-Reservierung (werktags 8-20 Uhr),
# --reservation imes.gpu mitgeben.
#
# Erfolgreich nur, wenn train.py mit 0 endet UND der finale Checkpoint auf
# BIGWORK liegt (train.py endet auch nach einer Exception mit 0, Job 8374044).
# Ergebnis: runs/results/<jobid>.json (gelesen von scripts/verify.py).
set -u
BASE=/bigwork/nhkbrc2a/isaac
RUNS=$BASE/runs
LOG=$BASE/logs/slurm_train_kurs_${SLURM_JOB_ID}.out
HELPER=$BASE/scripts/job_result.sh
[ -f "$HELPER" ] || { echo "FEHLER: $HELPER fehlt - erst cluster/scripts hochladen"; exit 2; }
source "$HELPER"

TEAM=${TEAM:-}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITER=${MAX_ITER:-3000}
TASK=Unitree-G1-29dof-Velocity-Kurs
REWARDS=$BASE/kurs/teams/$TEAM.py

echo "=== node: $(hostname) | team=$TEAM envs=$NUM_ENVS iters=$MAX_ITER"
nvidia-smi -L

case "$TEAM" in
  ""|*[!A-Za-z0-9_-]*) echo "FEHLER: TEAM fehlt oder enthaelt andere Zeichen als A-Z a-z 0-9 _ - ('$TEAM')"; exit 2 ;;
esac
[ -f "$REWARDS" ] || { echo "FEHLER: $REWARDS fehlt - Gruppendatei nach kurs/teams/, dann cluster push kurs kurs"; exit 2; }

UNITREE=$BASE/vendor/unitree_rl_lab
[ -f "$UNITREE/VENDOR.txt" ] || { echo "FEHLER: $UNITREE fehlt - scripts/vendor_unitree.sh, dann push"; exit 2; }
[ -f "$BASE/vendor/pylib/VENDOR.txt" ] || { echo "FEHLER: $BASE/vendor/pylib fehlt - scripts/vendor_unitree.sh, dann push"; exit 2; }
export APPTAINERENV_UNITREE_RL_LAB=$UNITREE
export APPTAINERENV_KURS_REWARDS=$REWARDS
# $BASE: cpr (Lauf B) und kurs; vendor/pylib: im Container fehlende Pakete
export APPTAINERENV_PYTHONPATH=$BASE:$UNITREE/source/unitree_rl_lab:$BASE/vendor/pylib
RUN=$BASE/scripts/run_isaaclab.sh
ASSET_ROOT=$(dirname "$(find "$BASE/assets" -maxdepth 4 -type d -name NVIDIA | head -1)")
KIT_ARGS="--/persistent/isaac/asset_root/default=$ASSET_ROOT"
KIT_ARGS="$KIT_ARGS --/persistent/isaac/asset_root/cloud=$ASSET_ROOT"
KIT_ARGS="$KIT_ARGS --/persistent/isaac/asset_root/nvidia=$ASSET_ROOT"

# train.py schreibt logs/rsl_rl/... relativ zum Arbeitsverzeichnis. Ohne dieses
# cd waere das $HOME - im Container ein Wegwerf-Verzeichnis.
mkdir -p "$RUNS"
cd "$RUNS" || exit 2

# Gruppendatei pruefen, bevor die GPU eine Stunde rechnet (dieselbe Pruefung
# wie kurs/check_rewards.py auf dem Laptop)
bash $RUN /isaac-sim/python.sh $BASE/kurs/check_rewards.py "$REWARDS"
rc=$?
if [ $rc -ne 0 ]; then
  echo "FEHLER: Gruppendatei $REWARDS besteht die Pruefung nicht (siehe oben)."
  job_write_result "$RUNS/results/${SLURM_JOB_ID}.json" \
    kind=train job_id="$SLURM_JOB_ID" task="$TASK" team="$TEAM" exit_code=4 \
    node="$(hostname)" finished="$(date -Iseconds)"
  exit 4
fi
# Funktionstest mit Platzhalter-Robotern: nur ins Log, kein Abbruch - NaN wird im
# Training durch 0 ersetzt (E-009), und es gibt nur einen Nachtlauf pro Gruppe
bash $RUN /isaac-sim/python.sh $BASE/kurs/test_rewards.py "$REWARDS" \
  || echo "WARNUNG: test_rewards.py meldet Probleme (siehe oben) - Training laeuft trotzdem (E-009)."

# Retry nur fuer den bekannten Startup-Segfault (immer in den ersten ~180 s)
for attempt in 1 2 3; do
  t0=$SECONDS
  bash $RUN /isaac-sim/python.sh $BASE/kurs/_intern/run.py train \
    --task "$TASK" --headless --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITER" \
    --run_name "$TEAM" --kit_args "$KIT_ARGS"
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
# job_run_dir liest den Zeitstempel aus "Exact experiment name requested ...";
# train.py haengt --run_name erst danach an (Unitree scripts/rsl_rl/train.py:150-155).
# Ohne diese Zeile sucht die Pruefung im falschen Ordner (Fehlalarm in Job 9081343).
[ -n "$RUN_DIR" ] && RUN_DIR="${RUN_DIR}_${TEAM}"
LAST_CKPT=""
LAST_ITER=-1
[ -n "$RUN_DIR" ] && LAST_CKPT=$(job_last_checkpoint "$RUN_DIR")
[ -n "$LAST_CKPT" ] && LAST_ITER=$(job_checkpoint_iter "$LAST_CKPT")
# welche Rewards dieser Lauf hatte, gehoert neben seine Checkpoints
[ -n "$RUN_DIR" ] && cp "$REWARDS" "$RUN_DIR/team_rewards.py"
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
  kind=train job_id="$SLURM_JOB_ID" task="$TASK" team="$TEAM" num_envs="$NUM_ENVS" \
  max_iter="$MAX_ITER" exit_code="$rc" attempts="$attempt" \
  run_dir="$RUN_DIR" last_checkpoint="$LAST_CKPT" last_iter="$LAST_ITER" \
  node="$(hostname)" finished="$(date -Iseconds)"
echo "TRAIN_DONE"
exit $rc
