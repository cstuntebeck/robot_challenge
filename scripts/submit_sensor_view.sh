#!/bin/bash
#SBATCH --job-name=t002_sensor_view
#SBATCH --partition=imes.gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_sensor_view_%j.out

# T-002: Blickfeld-Video der D435-Halterung vor dem G1 (sensor_view.py).
# Rendert nacheinander jede Kombination aus MOUNTS x VARIANTS, je ein eigener
# Isaac-Sim-Prozess (zwei Umgebungen in einem Prozess sind nicht verlaesslich).
#   ./bin/cluster submit cluster/scripts/submit_sensor_view.sh --note "..." --yes
#
# Parameter (--env):
#   MOUNTS     Default "pelvis torso_link"
#   VARIANTS   Default "unten beide"
#   STEPS      Policy-Schritte je Video (50 Hz), Default 750 = 15 s
#   SPEED      Vorwaerts-Kommando [m/s], Default 0.6
#   CLEARANCE  Luft zwischen Koerper und Halterungsplatte [m], Default 0.02
#
# Erfolgreich nur, wenn JEDE Kombination ein mp4 und ein json mit "ok": true
# hinterlaesst. Der Exit-Code von Isaac-Sim-Skripten beweist nichts (Job 8374044).
set -u
BASE=/bigwork/nhkbrc2a/isaac
RUNS=$BASE/runs
HELPER=$BASE/scripts/job_result.sh
[ -f "$HELPER" ] || { echo "FEHLER: $HELPER fehlt - erst cluster/scripts hochladen"; exit 2; }
source "$HELPER"

MOUNTS=${MOUNTS:-pelvis torso_link}
VARIANTS=${VARIANTS:-unten beide}
STEPS=${STEPS:-750}
SPEED=${SPEED:-0.6}
CLEARANCE=${CLEARANCE:-0.02}

echo "=== node: $(hostname) | mounts=$MOUNTS variants=$VARIANTS steps=$STEPS speed=$SPEED clearance=$CLEARANCE"
nvidia-smi -L

export APPTAINERENV_PYTHONPATH=$BASE
RUN=$BASE/scripts/run_isaaclab.sh
ASSET_ROOT=$(dirname "$(find "$BASE/assets" -maxdepth 4 -type d -name NVIDIA | head -1)")
SRC=$ASSET_ROOT/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Rough-G1-v0/checkpoint.pt
[ -f "$SRC" ] || { echo "FEHLER: kein veroeffentlichter Checkpoint: $SRC"; exit 2; }
OUT=$RUNS/sensor_view/$SLURM_JOB_ID
mkdir -p "$OUT"
cp "$SRC" "$OUT/checkpoint.pt"

# Relative Pfade landen sonst im Wegwerf-HOME des Containers.
cd "$RUNS" || exit 2

rc_all=0
done_list=""
for MOUNT in $MOUNTS; do
  for VARIANT in $VARIANTS; do
    TAG=${MOUNT}_${VARIANT}
    # Retry nur gegen den Startup-Segfault der ersten ~180 s (wie submit_train.sh).
    for attempt in 1 2 3; do
      t0=$SECONDS
      bash $RUN /isaac-sim/python.sh "$BASE/scripts/sensor_view.py" \
        --mount "$MOUNT" --variant "$VARIANT" --checkpoint "$OUT/checkpoint.pt" --out "$OUT" \
        --steps "$STEPS" --speed "$SPEED" --clearance "$CLEARANCE" --headless --enable_cameras
      rc=$?
      dt=$((SECONDS - t0))
      echo "RUN $TAG attempt $attempt exit=$rc after ${dt}s"
      [ $rc -eq 0 ] && [ -s "$OUT/$TAG.mp4" ] && break
      if [ $dt -ge 180 ]; then
        echo "Abbruch $TAG nach ${dt}s - kein Startup-Segfault, kein Retry."
        break
      fi
      echo "Startup-Absturz, Versuch $((attempt + 1)) ..."
      sleep 10
    done
    # Nur das Artefakt zaehlt: Video vorhanden und das Skript meldet ok.
    if [ -s "$OUT/$TAG.mp4" ] && grep -q '"ok": true' "$OUT/$TAG.json" 2>/dev/null; then
      echo "SENSOR_VIEW_OK: $OUT/$TAG.mp4 ($(stat -c %s "$OUT/$TAG.mp4") Bytes)"
      done_list="$done_list $TAG"
    else
      echo "FEHLER: $TAG ohne Video oder ohne ok im json"
      rc_all=3
    fi
  done
done

job_write_result "$RUNS/results/${SLURM_JOB_ID}.json" \
  kind=sensor_view job_id="$SLURM_JOB_ID" out="$OUT" mounts="$MOUNTS" variants="$VARIANTS" \
  steps="$STEPS" speed="$SPEED" clearance="$CLEARANCE" fertig="${done_list# }" \
  exit_code="$rc_all" node="$(hostname)" finished="$(date -Iseconds)"
echo "SENSOR_VIEW_DONE rc=$rc_all"
exit $rc_all
