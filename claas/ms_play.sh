#!/bin/bash
#SBATCH --job-name=claas_ms_play
#SBATCH --partition=imes.gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_claas_ms_play_%j.out

# Meilenstein (Claas): Videos zum NEUESTEN Checkpoint eines Trainingsjobs, mehrere Szenarien.
# Laeuft auch, wenn das Training abgebrochen ist (afterany) - dann mit dem letzten Zwischenstand.
#
# Parameter (--export):
#   TRAIN_JOB     Job-ID des Trainings (Log logs/slurm_claas_ms_<id>.out)
#   SCENARIOS     Tasks, Default "Unitree-G1-Claas-MS-Play-StairsEasy Unitree-G1-Claas-MS-Play-StairsMid Unitree-G1-Claas-MS-Play-FlatPush"
#   VIDEO_LENGTH  Schritte (50 pro s), Default 750 = 15 s
#   TAG           Kurzname fuer die Videodateien, Default ms_<TRAIN_JOB>
#
# Videos: runs/videos/<TAG>_<szenario>.mp4. Jedes Szenario bekommt eine eigene Kopie des
# Checkpoints, weil play.py das Video NEBEN den Checkpoint schreibt und sich die Laeufe
# sonst gegenseitig ueberschreiben.
set -u
BASE=/bigwork/nhkbrc2a/isaac
RUNS=$BASE/runs
LOG=$BASE/logs/slurm_claas_ms_play_${SLURM_JOB_ID}.out
source "$BASE/scripts/job_result.sh"

TRAIN_JOB=${TRAIN_JOB:?TRAIN_JOB fehlt}
SCENARIOS=${SCENARIOS:-Unitree-G1-Claas-MS-Play-StairsEasy Unitree-G1-Claas-MS-Play-StairsMid Unitree-G1-Claas-MS-Play-FlatPush}
VIDEO_LENGTH=${VIDEO_LENGTH:-750}
TAG=${TAG:-ms_${TRAIN_JOB}}

TLOG=$BASE/logs/slurm_claas_ms_${TRAIN_JOB}.out
RUN_DIR=$(job_run_dir "$TLOG")
CKPT=${CKPT:-}   # optional fester Checkpoint (sonst letzter des Laufs)
[ -z "$CKPT" ] && [ -n "$RUN_DIR" ] && CKPT=$(job_last_checkpoint "$RUN_DIR")
[ -n "$CKPT" ] || { echo "FEHLER: kein Checkpoint fuer Job $TRAIN_JOB (Laufordner: ${RUN_DIR:-?})"; exit 2; }
echo "=== node: $(hostname) | checkpoint=$CKPT | szenarien=$SCENARIOS"
nvidia-smi -L

UNITREE=$BASE/vendor/unitree_rl_lab
export APPTAINERENV_UNITREE_RL_LAB=$UNITREE
export APPTAINERENV_PYTHONPATH=$BASE:$UNITREE/source/unitree_rl_lab:$BASE/vendor/pylib
RUN=$BASE/scripts/run_isaaclab.sh
ASSET_ROOT=$(dirname "$(find "$BASE/assets" -maxdepth 4 -type d -name NVIDIA | head -1)")
KIT_ARGS="--/persistent/isaac/asset_root/default=$ASSET_ROOT"
KIT_ARGS="$KIT_ARGS --/persistent/isaac/asset_root/cloud=$ASSET_ROOT"
KIT_ARGS="$KIT_ARGS --/persistent/isaac/asset_root/nvidia=$ASSET_ROOT"

mkdir -p "$RUNS/videos" "$RUNS/claas/play"
cd "$RUNS" || exit 2
rc_all=0
videos=""
for SC in $SCENARIOS; do
  short=${SC##*-Play-}
  dir=$RUNS/claas/play/${SLURM_JOB_ID}_${short}
  mkdir -p "$dir"
  cp "$CKPT" "$dir/$(basename "$CKPT")"
  for attempt in 1 2 3; do
    t0=$SECONDS
    bash "$RUN" /isaac-sim/python.sh "$BASE/claas/unitree_run.py" play \
      --task "$SC" --headless --video --video_length "$VIDEO_LENGTH" --num_envs 9 \
      --checkpoint "$dir/$(basename "$CKPT")" --kit_args "$KIT_ARGS"
    rc=$?
    dt=$((SECONDS - t0))
    echo "PLAY $short attempt $attempt exit=$rc after ${dt}s"
    src=$(find "$dir/videos/play" -name '*.mp4' 2>/dev/null | head -n 1)
    [ -n "$src" ] && break
    [ $dt -ge 180 ] && break
    sleep 10
  done
  if [ -n "$src" ]; then
    out=$RUNS/videos/${TAG}_${short}.mp4
    cp "$src" "$out"
    echo "VIDEO_OK: $out ($(stat -c %s "$out") Bytes)"
    videos="$videos $out"
  else
    echo "FEHLER: kein Video fuer $short"
    rc_all=3
  fi
done

job_write_result "$RUNS/results/${SLURM_JOB_ID}.json" \
  kind=claas_ms_play job_id="$SLURM_JOB_ID" train_job="$TRAIN_JOB" checkpoint="$CKPT" \
  videos="${videos# }" exit_code="$rc_all" node="$(hostname)" finished="$(date -Iseconds)"
echo "PLAY_DONE rc=$rc_all"
exit $rc_all
