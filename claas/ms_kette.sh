#!/bin/bash
# Meilenstein 18./19.11. (Claas): komplette Job-Kette einreihen. Auf dem Login-Knoten ausfuehren:
#   bash claas/ms_kette.sh <SMOKE_JOBID>
# Alles haengt am erfolgreichen Smoke-Test (afterok). Scheitert er, startet nichts.
#
#   Smoke ──► B1 (Treppen)          ──► Video B1 ─┐
#        ├──► B2 (Treppen, Track 2) ──► Video B2 ─┤
#        └──► B3 (eben, DR)         ──► Video B3 ─┤
#   B1+B2+B3 ──► Auswahl ──► C (Bester weiter) ──► Video C ──► Bericht
set -eu
cd /bigwork/nhkbrc2a/isaac
S=${1:?Smoke-Job-ID fehlt}
BASE_CKPT=/bigwork/nhkbrc2a/isaac/runs/logs/rsl_rl/unitree_g1_29dof_velocity/2026-11-16_12-44-10/model_2999.pt
R="--reservation=imes.gpu"   # FLEX-Reservierung: Start auch ausserhalb 8-20 Uhr moeglich

B1=$(sbatch --parsable $R -J ms_B1_claas --time=10:00:00 --dependency=afterok:$S \
  --export=ALL,TASK=Unitree-G1-Claas-MS-B1,INIT_CKPT=$BASE_CKPT,MAX_ITER=3000 claas/ms_train.sh)
B2=$(sbatch --parsable $R -J ms_B2_claas --time=10:00:00 --dependency=afterok:$S \
  --export=ALL,TASK=Unitree-G1-Claas-MS-B2,INIT_CKPT=$BASE_CKPT,MAX_ITER=3000 claas/ms_train.sh)
B3=$(sbatch --parsable $R -J ms_B3_claas --time=06:00:00 --dependency=afterok:$S \
  --export=ALL,TASK=Unitree-G1-Claas-MS-B3,INIT_CKPT=$BASE_CKPT,MAX_ITER=2000 claas/ms_train.sh)

V1=$(sbatch --parsable $R -J ms_V1_claas --dependency=afterany:$B1 --export=ALL,TRAIN_JOB=$B1,TAG=ms_B1_$B1 claas/ms_play.sh)
V2=$(sbatch --parsable $R -J ms_V2_claas --dependency=afterany:$B2 --export=ALL,TRAIN_JOB=$B2,TAG=ms_B2_$B2 claas/ms_play.sh)
V3=$(sbatch --parsable $R -J ms_V3_claas --dependency=afterany:$B3 \
  --export="ALL,TRAIN_JOB=$B3,TAG=ms_B3_$B3,SCENARIOS=Unitree-G1-Claas-MS-Play-FlatPush Unitree-G1-Claas-MS-Play-StairsEasy" claas/ms_play.sh)

SEL=$(sbatch --parsable $R -J ms_auswahl_claas --dependency=afterany:$B1:$B2:$B3 \
  --export="ALL,MODE=select,ARGS=B1=$B1 B2=$B2 B3=$B3 BASELINE=$BASE_CKPT" claas/ms_auswahl.sh)
C=$(sbatch --parsable $R -J ms_C_claas --time=14:00:00 --dependency=afterok:$SEL \
  --export=ALL,FROM_SELECTION=1,MAX_ITER=5000 claas/ms_train.sh)
VC=$(sbatch --parsable $R -J ms_VC_claas --dependency=afterany:$C --export=ALL,TRAIN_JOB=$C,TAG=ms_C_$C claas/ms_play.sh)
REP=$(sbatch --parsable $R -J ms_bericht_claas --dependency=afterany:$VC:$V1:$V2:$V3 \
  --export="ALL,MODE=report,ARGS=B1=$B1 B2=$B2 B3=$B3 C=$C" claas/ms_auswahl.sh)

mkdir -p runs/claas
cat > runs/claas/ms_jobs.env <<EOF
SMOKE=$S
B1=$B1
B2=$B2
B3=$B3
V1=$V1
V2=$V2
V3=$V3
SEL=$SEL
C=$C
VC=$VC
REP=$REP
EOF
echo "KETTE OK"; cat runs/claas/ms_jobs.env
