#!/bin/bash
#SBATCH --job-name=g23_s2srob_claas
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_claas_s2s_%j.out
# Robustheitstest A5 in MuJoCo (Claas, 22.11.): gleicher 33-s-Ablauf unter Stoerungen.
# Tabelle: runs/sim2sim/A5_robust.md
cd /bigwork/nhkbrc2a/isaac
P=$(ls runs/claas/play/9399922_*/exported/policy.pt | head -1)
D=$A5RUN/params/deploy.yaml
export OMP_NUM_THREADS=4
NAMES=""
while read -r NAME ENVS; do
  [ -z "$NAME" ] && continue
  echo "== $NAME ($ENVS)"
  env $ENVS POLICY=$P DEPLOY=$D TAG=A5rob_$NAME bash claas/sim2sim/run_sim2sim.sh 2>&1 | grep -E "ERGEBNIS|Error"
  NAMES="$NAMES $NAME"
done <<'EOF'
nominal       S2S_DUMMY=1
glatt_0.3     S2S_FRICTION=0.3
last_5kg      S2S_MASS=5
latenz_30ms   S2S_DELAY=15
latenz_40ms   S2S_DELAY=20
stoss_1.0     S2S_PUSH=1.0
stoss_1.5     S2S_PUSH=1.5
kombi         S2S_FRICTION=0.5 S2S_MASS=3 S2S_DELAY=12 S2S_PUSH=1.0
EOF
apptainer exec -B /bigwork containers/isaac-lab-2.3.2.sif /isaac-sim/python.sh - $NAMES <<'PY'
import json, sys
rows = ["| Szenario | Sturz bei s | vor 1,0 | seitl. 0,3 | drehen 0,8 | Stossabschnitt vx |", "|---|---|---|---|---|---|"]
for n in sys.argv[1:]:
    try:
        d = json.load(open(f"runs/sim2sim/A5rob_{n}.json"))
    except Exception:
        rows.append(f"| {n} | Fehler | | | | |"); continue
    a = {x["abschnitt"]: x for x in d["abschnitte"]}
    g = lambda k, f: a[k][f] if k in a else "-"
    rows.append(f"| {n} | {d['gestuerzt_bei_s'] or 'kein Sturz'} | {g('vor 1,0','gemessen_vx')} | {g('seitlich 0,3','gemessen_vy')} | {g('drehen 0,8','gemessen_wz')} | {g('Stoesse','gemessen_vx')} |")
open("runs/sim2sim/A5_robust.md", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
PY
