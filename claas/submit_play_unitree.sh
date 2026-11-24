#!/bin/bash
#SBATCH --job-name=claas_play
#SBATCH --partition=imes.gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_play_%j.out

# Eine Policy abspielen und ein Video aufzeichnen (Deliverables 1 und 2, spaeter
# jedes Video eines eigenen Laufs). Aufruf ueber den Wrapper:
#   ./bin/cluster submit cluster/scripts/submit_play.sh \
#       --env TASK=Isaac-Velocity-Rough-G1-Play-v0 --env CAM=follow --note "..." --yes
#
# Parameter (--env):
#   TASK=${TASK:-Isaac-Velocity-Flat-G1-Play-v0}
#   CHECKPOINT=${CHECKPOINT:-/bigwork/nhkbrc2a/isaac/runs/logs/rsl_rl/unitree_g1_29dof_velocity/2026-11-16_17-29-25/model_999.pt
#   VIDEO_LENGTH  Schritte, Default 500 = 10 s (ein Schritt = 4 x 0,005 s, 50 fps)
#   NUM_ENVS      Default: was die Play-Konfiguration vorgibt (G1: 50)
#   CAM           overview (Default): feste Kamera (7.5, 7.5, 7.5) auf den Weltursprung
#                 env: Kamera relativ zum Startpunkt von Umgebung 0 - nah am Roboter,
#                      folgt ihm aber nicht
#
# Erfolgreich nur, wenn danach ein mp4 auf BIGWORK liegt. Der Exit-Code von
# play.py beweist nichts: eine Python-Exception endet dort mit 0 (Job 8374044),
# und ohne veroeffentlichten Checkpoint kehrt play.py still zurueck
# (play.py:103-107).
#
# play.py schreibt Video und exportierte Policy NEBEN den Checkpoint
# (play.py:113, 128, 171). Mit --use_pretrained_checkpoint ist das im
# Offline-Assetpack: retrieve_file_path gibt eine lokale Datei unveraendert
# zurueck, statt sie zu kopieren (utils/assets.py:68, 90). So landeten die Videos
# von 8380327/8380614 im Assetpack, und die Pruefung suchte vergeblich in runs/.
# Deshalb wird der veroeffentlichte Checkpoint hier zuerst nach
# runs/pretrained/<jobid>_<task>/ kopiert und per --checkpoint uebergeben. Das
# Video wird danach nach runs/videos/<jobid>_<task>.mp4 kopiert, damit es ein
# spaeterer Lauf nicht ueberschreibt.
set -u
BASE=/bigwork/nhkbrc2a/isaac
RUNS=$BASE/runs
LOG=$BASE/logs/slurm_play_${SLURM_JOB_ID}.out
HELPER=$BASE/scripts/job_result.sh
[ -f "$HELPER" ] || { echo "FEHLER: $HELPER fehlt - erst cluster/scripts hochladen"; exit 2; }
source "$HELPER"

TASK=${TASK:-Unitree-G1-29dof-Velocity}
CHECKPOINT=${CHECKPOINT:-/bigwork/nhkbrc2a/isaac/runs/logs/rsl_rl/unitree_g1_29dof_velocity/2026-11-16_17-29-25/model_999.pt}
VIDEO_LENGTH=${VIDEO_LENGTH:-2000}
NUM_ENVS=${NUM_ENVS:-1}
CAM=${CAM:-overview}

echo "=== node: $(hostname) | task=$TASK checkpoint=$CHECKPOINT length=$VIDEO_LENGTH cam=$CAM"
nvidia-smi -L

UNITREE=$BASE/vendor/unitree_rl_lab

export APPTAINERENV_UNITREE_RL_LAB=$UNITREE
export APPTAINERENV_PYTHONPATH=$BASE:$UNITREE/source/unitree_rl_lab:$BASE/vendor/pylib
RUN=$BASE/scripts/run_isaaclab.sh
ASSET_ROOT=$(dirname "$(find "$BASE/assets" -maxdepth 4 -type d -name NVIDIA | head -1)")
# Das Stock-Skript nutzt den Offline-Fix nicht. Ueber die cloud-Wurzel findet
# es auch den veroeffentlichten Checkpoint lokal (utils/assets.py:30-39).
KIT_ARGS="--/persistent/isaac/asset_root/default=$ASSET_ROOT"
KIT_ARGS="$KIT_ARGS --/persistent/isaac/asset_root/cloud=$ASSET_ROOT"
KIT_ARGS="$KIT_ARGS --/persistent/isaac/asset_root/nvidia=$ASSET_ROOT"

ARGS=(--task "$TASK" --headless --video --video_length "$VIDEO_LENGTH" --kit_args "$KIT_ARGS")
[ -n "$NUM_ENVS" ] && ARGS+=(--num_envs "$NUM_ENVS")
if [ "$CHECKPOINT" = "pretrained" ]; then
  # Gleiche Namensregel wie play.py:88 - der Checkpoint gehoert zur Trainings-Task
  TRAIN_TASK=${TASK/-Play/}
  SRC=$ASSET_ROOT/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/$TRAIN_TASK/checkpoint.pt
  [ -f "$SRC" ] || { echo "FEHLER: kein veroeffentlichter Checkpoint: $SRC"; exit 2; }
  CKPT_DIR=$RUNS/pretrained/${SLURM_JOB_ID}_${TRAIN_TASK}
  mkdir -p "$CKPT_DIR" && cp "$SRC" "$CKPT_DIR/checkpoint.pt"
  ARGS+=(--checkpoint "$CKPT_DIR/checkpoint.pt")
else
  ARGS+=(--checkpoint "$CHECKPOINT")
fi
case "$CAM" in
  overview) ;;
  env)
    # Hydra-Overrides (play.py:42-48); Listen werden zu Tupeln (utils/dict.py:109)
    ARGS+=(env.viewer.origin_type=env
           "env.viewer.eye=[5.0,5.0,3.0]" "env.viewer.lookat=[0.0,0.0,0.5]") ;;
  follow)
    # Folgen braeuchte origin_type=asset_root UND asset_name=robot. asset_name ist
    # per Default None, und Isaac Lab lehnt beim Uebernehmen der Hydra-Werte dann
    # jeden String ab (Job 8380328: "Incorrect type under namespace:
    # /viewer/asset_name. Expected: NoneType"). Geht nur mit eigenem Play-Skript.
    echo "FEHLER: CAM=follow ist per Hydra-Override nicht moeglich - CAM=env nehmen"; exit 2 ;;
  *) echo "FEHLER: CAM=$CAM unbekannt (overview|env)"; exit 2 ;;
esac

# Relative Pfade (logs/rsl_rl/...) loest play.py gegen das Arbeitsverzeichnis
# auf. Ohne dieses cd waere das $HOME - im Container ein Wegwerf-Verzeichnis.
mkdir -p "$RUNS/videos"
cd "$RUNS" || exit 2
MARK=$RUNS/.play_start_${SLURM_JOB_ID}
touch "$MARK"

# Retry nur gegen den Startup-Segfault der ersten ~180 s (wie submit_train.sh).
for attempt in 1 2 3; do
  t0=$SECONDS
  bash "$RUN" /isaac-sim/python.sh "$BASE/claas/unitree_run.py" play \
    "${ARGS[@]}"
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
echo "PLAY_EXIT: $rc"

# Gibt es ein neues Video? Nur das zaehlt. Gesucht wird dort, wo play.py laut
# eigenem Log hinschreibt (video_folder), nicht dort, wo wir es vermuten.
CKPT_USED=$(grep -a 'Loading model checkpoint from:' "$LOG" | tail -n 1 | sed 's/.*from: *//')
VIDEO_DIR=$(grep -a 'video_folder:' "$LOG" | tail -n 1 | sed 's/.*video_folder: *//')
VIDEO_SRC=""
[ -n "$VIDEO_DIR" ] && VIDEO_SRC=$(find "$VIDEO_DIR" -name '*.mp4' -newer "$MARK" 2>/dev/null | head -n 1)
VIDEO=""
VIDEO_BYTES=0
if [ -n "$VIDEO_SRC" ]; then
  VIDEO=$RUNS/videos/${SLURM_JOB_ID}_${TASK}.mp4
  cp "$VIDEO_SRC" "$VIDEO" && VIDEO_BYTES=$(stat -c %s "$VIDEO")
fi
rm -f "$MARK"
echo "PLAY_CHECKPOINT: ${CKPT_USED:-<nicht im Log>}"
echo "PLAY_VIDEO: ${VIDEO:-<keins>} ($VIDEO_BYTES Bytes, Quelle ${VIDEO_SRC:-?})"
if [ $rc -eq 0 ] && [ "$VIDEO_BYTES" -eq 0 ]; then
  echo "FEHLER: play.py endete mit 0, aber es entstand kein Video."
  rc=3
fi

job_write_result "$RUNS/results/${SLURM_JOB_ID}.json" \
  kind=play job_id="$SLURM_JOB_ID" task="$TASK" checkpoint="$CHECKPOINT" \
  checkpoint_used="$CKPT_USED" cam="$CAM" video_length="$VIDEO_LENGTH" \
  exit_code="$rc" attempts="$attempt" video="$VIDEO" video_bytes="$VIDEO_BYTES" \
  video_source="$VIDEO_SRC" node="$(hostname)" finished="$(date -Iseconds)"
echo "PLAY_DONE"
exit $rc
