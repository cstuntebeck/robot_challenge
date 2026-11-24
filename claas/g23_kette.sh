#!/bin/bash
# 23 DoF (Roboter der RC), Claas, 18.11. abends. Auf dem Login-Knoten:
#   bash claas/g23_kette.sh
#
#   Smoke (5 It.) -> Video-Test (alle 3 Szenarien, kurz)
#        -> A1 eben, Unitree-Rewards         -> Videos A1
#        -> A2 eben, Tracking 2,0            -> Videos A2
#        -> A3 Treppen ab null, Tracking 2,0 -> Videos A3
# Training ab null (kein 23-DoF-Modell vorhanden). Trainings auf Partition gpu (H200, ohne
# Video; ohne V100-Knoten gpu-20-n005..008, dort laeuft der Container nicht), Videos auf imes.gpu. Alles haengt am erfolgreichen Smoke-Test (afterok).
# Morgen: besten Lauf auf Treppen weitertrainieren (claas/ms_train.sh mit INIT_CKPT) + Sim-to-Sim.
set -eu
cd /bigwork/nhkbrc2a/isaac
R="--reservation=imes.gpu"
P23="Unitree-G1-Claas-23-Play-StairsEasy Unitree-G1-Claas-23-Play-StairsMid Unitree-G1-Claas-23-Play-FlatPush"

S=$(sbatch --parsable -p gpu --exclude="gpu-20-n[005-008]" -J g23_smoke_claas --time=00:30:00 \
  --export=ALL,TASK=Unitree-G1-Claas-23-Flat,NUM_ENVS=64,MAX_ITER=5,VIDEO=0 claas/ms_train.sh)
SV=$(sbatch --parsable $R -J g23_smokevid_claas --time=00:40:00 --dependency=afterok:$S \
  --export="ALL,TRAIN_JOB=$S,VIDEO_LENGTH=100,TAG=g23smoke_$S,SCENARIOS=$P23" claas/ms_play.sh)

A1=$(sbatch --parsable -p gpu --exclude="gpu-20-n[005-008]" -J g23_A1_claas --time=08:00:00 --dependency=afterok:$S \
  --export=ALL,TASK=Unitree-G1-Claas-23-Flat,MAX_ITER=4000,VIDEO=0 claas/ms_train.sh)
A2=$(sbatch --parsable -p gpu --exclude="gpu-20-n[005-008]" -J g23_A2_claas --time=08:00:00 --dependency=afterok:$S \
  --export=ALL,TASK=Unitree-G1-Claas-23-FlatTrack,MAX_ITER=4000,VIDEO=0 claas/ms_train.sh)
A3=$(sbatch --parsable -p gpu --exclude="gpu-20-n[005-008]" -J g23_A3_claas --time=10:00:00 --dependency=afterok:$S \
  --export=ALL,TASK=Unitree-G1-Claas-23-StairsTrack,MAX_ITER=5000,VIDEO=0 claas/ms_train.sh)

V1=$(sbatch --parsable $R -J g23_V1_claas --dependency=afterany:$A1 --export="ALL,TRAIN_JOB=$A1,TAG=g23_A1_$A1,SCENARIOS=$P23" claas/ms_play.sh)
V2=$(sbatch --parsable $R -J g23_V2_claas --dependency=afterany:$A2 --export="ALL,TRAIN_JOB=$A2,TAG=g23_A2_$A2,SCENARIOS=$P23" claas/ms_play.sh)
V3=$(sbatch --parsable $R -J g23_V3_claas --dependency=afterany:$A3 --export="ALL,TRAIN_JOB=$A3,TAG=g23_A3_$A3,SCENARIOS=$P23" claas/ms_play.sh)

mkdir -p runs/claas
cat > runs/claas/g23_jobs.env <<EOF
SMOKE=$S
SMOKE_VIDEO=$SV
A1=$A1
A2=$A2
A3=$A3
V1=$V1
V2=$V2
V3=$V3
EOF
echo "G23 KETTE OK"; cat runs/claas/g23_jobs.env
