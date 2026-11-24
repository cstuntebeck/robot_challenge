"""Checkpoint model_N.pt als TorchScript-Policy exportieren (Claas, 19.11.).
Nimmt eine vorhandene exportierte policy.pt desselben Laufs als Huelle und laedt die
Actor-Gewichte aus dem Checkpoint (gleiche Architektur, keine Normalisierung bei A2).
  python export_ckpt.py <policy.pt> <model_N.pt> <ziel.pt>"""
import sys

import torch

base = torch.jit.load(sys.argv[1], map_location="cpu")
sd = torch.load(sys.argv[2], map_location="cpu", weights_only=False)["model_state_dict"]
bsd = base.state_dict()
miss = [k for k in bsd if k not in sd]
if miss:
    raise SystemExit(f"FEHLER: Schluessel fehlen im Checkpoint: {miss[:5]}")
base.load_state_dict({k: sd[k] for k in bsd})
base.save(sys.argv[3])
print("EXPORT_OK", sys.argv[2].split("/")[-1], len(bsd), "Tensoren")
