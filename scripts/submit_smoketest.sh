#!/bin/bash
#SBATCH --job-name=isaac_smoke
#SBATCH --partition=imes.gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_smoke_%j.out

# Bestanden, wenn ALLE vier Eigenschaften belegt sind - der Job prueft die
# entscheidenden selbst und endet sonst mit Exit != 0:
#   1. "no internet (as expected)"           -> der Compute-Knoten ist offline
#   2. Version 5.1.0-rc.19...                -> die richtige Software laeuft
#   3. Szene laedt ohne FileNotFoundError auf eine https://-URL, und mindestens
#      eine "Learning iteration" erscheint   -> ein Training kommt zustande
#   4. danach liegt ein model_*.pt auf BIGWORK -> die Ausgaben ueberleben den Job
# Punkt 4 fehlte im ersten Smoke-Test (8178097): er lief durch, aber sein
# Checkpoint landete im fluechtigen Container-HOME und war danach weg.
set -u
BASE=/bigwork/nhkbrc2a/isaac
RUNS=$BASE/runs
LOG=$BASE/logs/slurm_smoke_${SLURM_JOB_ID}.out
HELPER=$BASE/scripts/job_result.sh
[ -f "$HELPER" ] || { echo "FEHLER: $HELPER fehlt - erst cluster/scripts hochladen"; exit 2; }
source "$HELPER"

echo "=== node: $(hostname)"
nvidia-smi -L
echo "=== internet check (expect none):"
if timeout 10 curl -sI https://example.com >/dev/null 2>&1; then
  INTERNET=yes; echo "has internet"
else
  INTERNET=no; echo "no internet (as expected)"
fi

export APPTAINERENV_PYTHONPATH=$BASE
RUN=$BASE/scripts/run_isaaclab.sh
ASSET_ROOT=$(dirname "$(find "$BASE/assets" -maxdepth 4 -type d -name NVIDIA | head -1)")
echo "=== asset root: $ASSET_ROOT"

# train.py schreibt logs/rsl_rl/... relativ zum Arbeitsverzeichnis. Ohne dieses
# cd waere das $HOME - und das ist im Container ein Wegwerf-Verzeichnis.
mkdir -p "$RUNS"
cd "$RUNS" || exit 2

echo "=== [1/2] versions:"
SIM_VERSION=$(bash $RUN cat /isaac-sim/VERSION | head -n 1)
echo "$SIM_VERSION"
bash $RUN bash -lc "cd /workspace/isaaclab && cat VERSION"

# Das Stock-Trainingsskript nutzt den Offline-Fix nicht, deshalb werden die
# Asset-Wurzeln hier direkt als kit-Argumente angehaengt.
KIT_ARGS="--/persistent/isaac/asset_root/default=$ASSET_ROOT"
KIT_ARGS="$KIT_ARGS --/persistent/isaac/asset_root/cloud=$ASSET_ROOT"
KIT_ARGS="$KIT_ARGS --/persistent/isaac/asset_root/nvidia=$ASSET_ROOT"

echo "=== [2/2] offline asset load + 2 PPO iterations on a stock task:"
bash $RUN /isaac-sim/python.sh /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 16 --max_iterations 2 \
  --kit_args "$KIT_ARGS"
rc=$?
echo "SMOKE_EXIT: $rc"

# Eigenschaft 4: hat der Checkpoint den Job ueberlebt?
RUN_DIR=$(job_run_dir "$LOG")
LAST_CKPT=""
[ -n "$RUN_DIR" ] && LAST_CKPT=$(job_last_checkpoint "$RUN_DIR")
echo "SMOKE_RUN_DIR: ${RUN_DIR:-<nicht im Log gefunden>}"
echo "SMOKE_CHECKPOINT: ${LAST_CKPT:-<fehlt>}"
if [ $rc -eq 0 ] && [ -z "$LAST_CKPT" ]; then
  echo "FEHLER: Training lief, aber kein model_*.pt unter ${RUN_DIR:-?} - Ausgaben verloren."
  rc=3
fi

job_write_result "$RUNS/results/${SLURM_JOB_ID}.json" \
  kind=smoke job_id="$SLURM_JOB_ID" task=Isaac-Velocity-Flat-G1-v0 max_iter=2 \
  exit_code="$rc" internet="$INTERNET" isaac_sim_version="$SIM_VERSION" \
  run_dir="$RUN_DIR" last_checkpoint="$LAST_CKPT" node="$(hostname)" \
  finished="$(date -Iseconds)"
echo "SMOKE_DONE"
exit $rc
