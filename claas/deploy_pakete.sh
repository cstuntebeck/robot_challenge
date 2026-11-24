#!/bin/bash
# Deploy-Pakete fuer den echten G1 (23 DoF) schnueren - Claas, 22.11.
#   bash claas/deploy_pakete.sh
# Ergebnis: claas/deploy/<Name>/ mit policy.onnx, policy.pt, deploy.yaml, Checkpoint, INFO.txt
# und ein gemeinsames README (claas/deploy/README.md).
set -u
cd /bigwork/nhkbrc2a/isaac
source scripts/job_result.sh
OUT=claas/deploy
mkdir -p $OUT

# Name  Trainingsjob  Videojob  Beschreibung
while read -r NAME TJ PJ DESC; do
  [ -z "$NAME" ] && continue
  RUN=$(job_run_dir logs/slurm_claas_ms_${TJ}.out)
  EXP=$(ls -d runs/claas/play/${PJ}_*/exported 2>/dev/null | head -1)
  CK=$(ls runs/claas/play/${PJ}_*/model_*.pt 2>/dev/null | head -1)
  D=$OUT/$NAME; mkdir -p $D
  if [ -z "$EXP" ] || [ ! -f "$RUN/params/deploy.yaml" ]; then echo "FEHLER $NAME: export=$EXP run=$RUN"; continue; fi
  cp "$EXP/policy.onnx" "$EXP/policy.pt" "$RUN/params/deploy.yaml" "$RUN/params/env.yaml" "$RUN/params/agent.yaml" $D/
  cp "$CK" $D/
  {
    echo "Name:        $NAME"
    echo "Beschreibung: ${DESC//_/ }"
    echo "Trainingsjob: $TJ   Lauf: $RUN"
    echo "Checkpoint:  $(basename $CK)"
    echo "Export aus:  $EXP (Videojob $PJ)"
    echo "Erstellt:    $(date -Iseconds)"
    echo
    echo "sha256:"; (cd $D && sha256sum policy.onnx policy.pt deploy.yaml)
  } > $D/INFO.txt
  echo "OK $NAME <- $(basename $CK)"
done <<'EOF'
A5_robust_drehen   9399921 9399922 HAUPTKANDIDAT:_eben,_Latenz_0-20ms,_Motorstaerke_+-10%,_Drehen_+-1rad/s
A6_gangbild        9400164 9400165 wie_A5_+_Gangbild-Rewards_(Alternative,_Gang_im_Video_pruefen)
A4_robust          9391705 9391706 eben,_Latenz_+_Motorstaerke,_dreht_kaum_(Rueckfall)
A5s2_seed2         9419448 9419449 A5-Rezept_Seed_2,_dreht_in_MuJoCo_genauer_(0,80_bei_0,8)
A8_uneben          9419440 9419441 A5_+_unebener_Boden_(Rampen,_Kanten_bis_5_cm),_Reserve
A2_eben            9379135 9379138 eben,_ohne_Latenz/Motorstreuung_(Basis)
EOF

# Pruefen: ONNX und TorchScript liefern dieselbe Aktion
apptainer exec -B /bigwork containers/isaac-lab-2.3.2.sif /isaac-sim/python.sh - <<'PY' 2>&1 | grep -E "PRUEF|Error|rror"
import glob, numpy as np, torch
try:
    import onnxruntime as ort
except Exception as e:
    ort = None; print("PRUEF onnxruntime fehlt im Container - ONNX-Vergleich uebersprungen")
for d in sorted(glob.glob("claas/deploy/*/")):
    pt = torch.jit.load(d + "policy.pt", map_location="cpu").eval()
    n = 390
    x = np.random.default_rng(0).standard_normal((4, n)).astype(np.float32) * 0.3
    a = pt(torch.from_numpy(x)).detach().numpy()
    msg = f"PRUEF {d}: Eingang {n}, Ausgang {a.shape[1]}"
    if ort:
        s = ort.InferenceSession(d + "policy.onnx")
        b = s.run(None, {s.get_inputs()[0].name: x})[0]
        msg += f", max |onnx-pt| = {np.abs(a-b).max():.2e}"
    print(msg)
PY
ls -la $OUT/*/
