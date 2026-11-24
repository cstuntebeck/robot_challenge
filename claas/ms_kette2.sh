#!/bin/bash
# Meilenstein 18./19.11. (Claas), Erweiterung um B4/B5 (Curricula Version 2).
#   bash claas/ms_kette2.sh
# - B1, B2, B3 und ihre Videos bleiben unveraendert.
# - Abgebrochen werden nur die noch NICHT gestarteten Jobs Auswahl, C, Video C, Bericht
#   (sie kannten nur B1-B3) und mit B1-B5 neu eingereiht.
# - B4/B5 und Lauf C trainieren auf Partition gpu (H200, ohne Video; freie GPUs),
#   alle Videos laufen auf imes.gpu (L40S mit RT-Kernen).
#
#   B1 ─┐
#   B2 ─┤
#   B3 ─┼──► Auswahl ──► C ──► Video C ──► Bericht + Abgabe-Video
#   B4 ─┤  (neu, gpu) ──► Video B4
#   B5 ─┘  (neu, gpu) ──► Video B5
set -eu
cd /bigwork/nhkbrc2a/isaac
source runs/claas/ms_jobs.env
BASE_CKPT=/bigwork/nhkbrc2a/isaac/runs/logs/rsl_rl/unitree_g1_29dof_velocity/2026-11-16_12-44-10/model_2999.pt
R="--reservation=imes.gpu"

scancel "$SEL" "$C" "$VC" "$REP"
echo "abgebrochen (noch nicht gestartet): $SEL $C $VC $REP"

B4=$(sbatch --parsable -p gpu -J ms_B4_claas --time=10:00:00 \
  --export=ALL,TASK=Unitree-G1-Claas-MS-B4,INIT_CKPT=$BASE_CKPT,MAX_ITER=3000,VIDEO=0 claas/ms_train.sh)
B5=$(sbatch --parsable -p gpu -J ms_B5_claas --time=10:00:00 \
  --export=ALL,TASK=Unitree-G1-Claas-MS-B5,INIT_CKPT=$BASE_CKPT,MAX_ITER=3000,VIDEO=0 claas/ms_train.sh)
V4=$(sbatch --parsable $R -J ms_V4_claas --dependency=afterany:$B4 --export=ALL,TRAIN_JOB=$B4,TAG=ms_B4_$B4 claas/ms_play.sh)
V5=$(sbatch --parsable $R -J ms_V5_claas --dependency=afterany:$B5 --export=ALL,TRAIN_JOB=$B5,TAG=ms_B5_$B5 claas/ms_play.sh)

SEL=$(sbatch --parsable $R -J ms_auswahl_claas --dependency=afterany:$B1:$B2:$B3:$B4:$B5 \
  --export="ALL,MODE=select,ARGS=B1=$B1 B2=$B2 B3=$B3 B4=$B4 B5=$B5 BASELINE=$BASE_CKPT" claas/ms_auswahl.sh)
C=$(sbatch --parsable -p gpu -J ms_C_claas --time=14:00:00 --dependency=afterok:$SEL \
  --export=ALL,FROM_SELECTION=1,MAX_ITER=5000,VIDEO=0 claas/ms_train.sh)
VC=$(sbatch --parsable $R -J ms_VC_claas --dependency=afterany:$C --export=ALL,TRAIN_JOB=$C,TAG=ms_C_$C claas/ms_play.sh)
REP=$(sbatch --parsable $R -J ms_bericht_claas --time=00:30:00 --dependency=afterany:$VC:$V1:$V2:$V3:$V4:$V5 \
  --export="ALL,MODE=report,ARGS=B1=$B1 B2=$B2 B3=$B3 B4=$B4 B5=$B5 C=$C" claas/ms_auswahl.sh)

cat > runs/claas/ms_jobs.env <<EOF
SMOKE=$SMOKE
B1=$B1
B2=$B2
B3=$B3
B4=$B4
B5=$B5
V1=$V1
V2=$V2
V3=$V3
V4=$V4
V5=$V5
SEL=$SEL
C=$C
VC=$VC
REP=$REP
EOF
echo "KETTE2 OK"; cat runs/claas/ms_jobs.env
