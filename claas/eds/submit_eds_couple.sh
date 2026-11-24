#!/bin/bash
#SBATCH --job-name=g23_eds_couple_claas
#SBATCH --partition=imes.gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_claas_edscouple_%j.out
# CD2 ToDo 5 (Claas, 24.11.): L3 mit exakter / verrauschter / EDS-Hoehenkarte auf der Show-Treppe.
# --export: CKPT (L3), MODELS (Ordner mit ae_*.pt, decoder_*.pt)
set -u
BASE=/bigwork/nhkbrc2a/isaac
RUNS=$BASE/runs
UNITREE=$BASE/vendor/unitree_rl_lab
export APPTAINERENV_UNITREE_RL_LAB=$UNITREE
export APPTAINERENV_PYTHONPATH=$BASE:$UNITREE/source/unitree_rl_lab:$BASE/vendor/pylib
RUN=$BASE/scripts/run_isaaclab.sh
ASSET_ROOT=$(dirname "$(find "$BASE/assets" -maxdepth 4 -type d -name NVIDIA | head -1)")
KIT_ARGS="--/persistent/isaac/asset_root/default=$ASSET_ROOT --/persistent/isaac/asset_root/cloud=$ASSET_ROOT --/persistent/isaac/asset_root/nvidia=$ASSET_ROOT"
case "$MODELS" in /*) ;; *) MODELS=$BASE/$MODELS ;; esac  # relativ zu BASE (Fehler 9438063: cd $RUNS)
OUT=$RUNS/eds_couple/$SLURM_JOB_ID
mkdir -p "$OUT"
cd "$RUNS" || exit 2
echo "=== node: $(hostname) | ckpt=$CKPT models=$MODELS"
run() {  # tag task mode [decoder] [video]
  local TAG=$1 T=$2 M=$3 D=${4:-} VID=${5:-}
  local EXTRA=""
  [ -n "$D" ] && { [ -f "$D" ] || { echo "SKIP $TAG: $D fehlt"; return; }; EXTRA="--decoder $D"; }
  [ -n "$VID" ] && EXTRA="$EXTRA --video --num_envs 9"
  for attempt in 1 2; do
    bash "$RUN" /isaac-sim/python.sh "$BASE/claas/unitree_run.py" script "$BASE/claas/eds/eds_couple.py" \
      --task "$T" --mode "$M" --checkpoint "$CKPT" --out "$OUT/$TAG" --kit_args "$KIT_ARGS" $EXTRA
    [ -f "$OUT/$TAG.json" ] && { echo "COUPLE_OK $TAG $(cat $OUT/$TAG.json | tr -d '\n')"; return; }
    sleep 10
  done
  echo "FEHLER $TAG"
}
for LV in Easy Mid; do
  T=Unitree-G1-Claas-23-TeacherEds-Play-ShowStairs$LV
  if [ -z "${SKIP_BASE:-}" ]; then run exact_$LV $T exact; run est_$LV $T est; fi
  run eds_fusion_$LV $T eds $MODELS/decoder_fusion.pt
  run eds_depth_$LV $T eds $MODELS/decoder_depth.pt
  run eds_lidar_$LV $T eds $MODELS/decoder_lidar.pt
  run eds_fusion_real_$LV $T eds $MODELS/decoder_fusion_real.pt
done
run video_eds_fusion_Mid Unitree-G1-Claas-23-TeacherEds-Play-ShowStairsMid eds $MODELS/decoder_fusion_real.pt video
V=$(find "$OUT" -name '*.mp4' | head -1); [ -n "$V" ] && cp "$V" $RUNS/videos/eds_couple_L3_fusion_real_Mid.mp4
echo "COUPLE_DONE"
