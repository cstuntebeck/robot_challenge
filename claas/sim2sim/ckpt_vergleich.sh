#!/bin/bash
#SBATCH --job-name=g23_ckvgl_claas
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_claas_s2s_%j.out
# Ueberanpassung an Isaac pruefen (Claas, 19.11.): mehrere Checkpoints EINES Laufs in MuJoCo testen.
#   sbatch -p gpu --exclude="gpu-20-n[005-008]" --export=ALL,RUN=<run-ordner>,POLICY=<exportierte policy.pt>,NAME=A2,CKPTS="500 1000 ..." claas/sim2sim/ckpt_vergleich.sh
# Ergebnisse: runs/sim2sim/<NAME>_ck<N>.json, Videos runs/videos/sim2sim_<NAME>_ck<N>.mp4, Tabelle runs/sim2sim/<NAME>_vergleich.md
cd /bigwork/nhkbrc2a/isaac
BASEPOL=$POLICY
export PYTHONPATH=/bigwork/nhkbrc2a/isaac/claas/ext/pylib OMP_NUM_THREADS=4
SIF=containers/isaac-lab-2.3.2.sif
TMP=runs/sim2sim/export_${SLURM_JOB_ID}; mkdir -p $TMP
for N in $CKPTS; do
  apptainer exec --nv -B /bigwork $SIF /isaac-sim/python.sh claas/sim2sim/export_ckpt.py "$BASEPOL" "$RUN/model_${N}.pt" "$TMP/policy_${N}.pt" 2>&1 | grep -E "EXPORT_OK|FEHLER|Error"
  POLICY="$TMP/policy_${N}.pt" DEPLOY="$RUN/params/deploy.yaml" TAG=${NAME}_ck${N} bash claas/sim2sim/run_sim2sim.sh 2>&1 | grep -E "ERGEBNIS|VIDEO_OK|Error"
done
apptainer exec -B /bigwork $SIF /isaac-sim/python.sh - "$NAME" $CKPTS <<'PY'
import json, sys
name, cks = sys.argv[1], sys.argv[2:]
rows = ["| Checkpoint | Sturz bei s | vx 0,5 | vx 1,0 | vy 0,3 | rueckw. 0,4 | Drehen 0,2 | Stoesse vx |", "|---|---|---|---|---|---|---|---|"]
for n in cks:
    try:
        d = json.load(open(f"runs/sim2sim/{name}_ck{n}.json"))
    except Exception:
        rows.append(f"| {n} | Fehler | | | | | | |"); continue
    a = {x["abschnitt"]: x for x in d["abschnitte"]}
    g = lambda k, f: a[k][f] if k in a else "-"
    rows.append(f"| {n} | {d['gestuerzt_bei_s'] or 'kein Sturz'} | {g('vor 0,5','gemessen_vx')} | {g('vor 1,0','gemessen_vx')} | {g('seitlich 0,3','gemessen_vy')} | {g('rueckwaerts 0,4','gemessen_vx')} | {g('Kurve','gemessen_wz')} | {g('Stoesse','gemessen_vx')} |")
open(f"runs/sim2sim/{name}_vergleich.md", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
PY
