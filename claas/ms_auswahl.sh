#!/bin/bash
#SBATCH --job-name=claas_ms_auswahl
#SBATCH --partition=imes.gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_claas_ms_auswahl_%j.out

# Meilenstein (Claas): Auswahl nach Stufe 1 (MODE=select) oder Abschlussbericht (MODE=report).
# Parameter (--export): MODE, ARGS (z. B. "B1=123 B2=124 B3=125 BASELINE=/pfad/model_2999.pt")
set -u
BASE=/bigwork/nhkbrc2a/isaac
cd "$BASE/runs" || exit 2
bash "$BASE/scripts/run_isaaclab.sh" /isaac-sim/python.sh "$BASE/claas/ms_auswahl.py" "${MODE:?}" ${ARGS:?}
rc=$?
if [ "$MODE" = "select" ] && [ ! -s "$BASE/runs/claas/ms_auswahl.env" ]; then
  echo "FEHLER: ms_auswahl.env nicht geschrieben"; rc=3
fi
echo "AUSWAHL_DONE rc=$rc"
exit $rc
