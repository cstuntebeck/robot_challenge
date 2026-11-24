#!/bin/bash
# Nur die ONNX/TorchScript-Pruefung der Deploy-Pakete (Claas, 22.11.)
cd /bigwork/nhkbrc2a/isaac
# Pruefen: ONNX und TorchScript liefern dieselbe Aktion
apptainer exec -B /bigwork containers/isaac-lab-2.3.2.sif /isaac-sim/python.sh - <<'PY' 2>&1 | grep -E "PRUEF|Error|rror" | tee claas/deploy/PRUEFUNG.txt
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
