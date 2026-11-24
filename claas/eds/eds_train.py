"""CD2 / EDS: Hoehenkarten-Schaetzer trainieren (Vorlesung RL_v2, Folien 20, 33-38) - Claas, 23.11.

Stufe 1 (--stage ae_depth / ae_lidar): CNN-Autoencoder je Sensor (strided conv 3x3 + ReLU,
  FC auf Latent 256), Verlust MAE auf gueltigen Pixeln.
Stufe 2 (--stage decoder): Encoder eingefroren; je Sensor Linear 256->128, Roboterzustand 15->64,
  letzte eigene Vorhersage 165->128, Fusion Linear+LayerNorm+ReLU -> 256, GRU (2 Schichten, 256),
  Linear+ReLU 256, Linear -> 165. Ueberwacht gegen die exakte Hoehenkarte (MAE), Sequenzen
  der Laenge --seq (16 wie auf Folie 37), nie ueber einen Reset hinweg.
--modal depth | lidar | fusion waehlt die Sensoren fuer Stufe 2 (Vergleich wie Folie 36/37).

Daten: Ordner von eds_collect.py (shard_*.npz). Ergebnis: <out>/<stage>.pt und <out>/<stage>.json
(MAE in cm auf zurueckgehaltenen Robotern, Kurven je Epoche). Ausgabe des Decoders ist direkt
die Hoehenkarte, die die Policy L3 erwartet (Wert = Sensorhoehe - Boden - 0,5, auf +-1 begrenzt).
"""

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--data", nargs="+", required=True, help="Ordner mit shard_*.npz")
p.add_argument("--out", required=True)
p.add_argument("--stage", choices=["ae_depth", "ae_lidar", "decoder"], required=True)
p.add_argument("--modal", choices=["depth", "lidar", "fusion"], default="fusion")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seq", type=int, default=16)
p.add_argument("--latent", type=int, default=256)
p.add_argument("--val_frac", type=float, default=0.15)
p.add_argument("--max_shards", type=int, default=1000)
p.add_argument("--state_cols", choices=["all", "real"], default="all", help="real: nur was der echte Roboter misst (IMU, Befehl, Roll/Nick)")
args = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)

DW, DH = 160, 120  # Eingang Tiefe (mittig aus 212x120)
D_MAX = 3.0
L_MAX = 6.0


# ---------------------------------------------------------------- Daten
def load(keys):
    files = []
    for d in args.data:
        files += sorted(glob.glob(os.path.join(d, "**", "shard_*.npz"), recursive=True))
    files = files[: args.max_shards]
    if not files:
        raise SystemExit("keine shard_*.npz gefunden")
    parts = {k: [] for k in keys}
    run_id = []
    for i, f in enumerate(files):
        z = np.load(f)
        for k in keys:
            parts[k].append(z[k])
        run_id.append(np.full(z["hm"].shape[:2], i, dtype=np.int32))
        print(f"geladen {f}: {z['hm'].shape}", flush=True)
    return files, {k: v for k, v in parts.items()}, run_id


def crop_depth(d):
    w = d.shape[-1]
    x0 = (w - DW) // 2
    return d[..., :DH, x0:x0 + DW]


# ---------------------------------------------------------------- Netze
class ConvAE(nn.Module):
    """Klassischer CNN-Autoencoder (Folie 33): 4x strided conv 3x3 + ReLU, FC -> Latent."""

    def __init__(self, h, w, latent=256, ch=(16, 32, 64, 128)):
        super().__init__()
        layers, c = [], 1
        for o in ch:
            layers += [nn.Conv2d(c, o, 3, stride=2, padding=1), nn.ReLU()]
            c = o
        self.enc = nn.Sequential(*layers)
        with torch.no_grad():
            s = self.enc(torch.zeros(1, 1, h, w)).shape
        self.s = s[1:]
        n = int(np.prod(self.s))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(n, 512), nn.ReLU(), nn.Linear(512, latent))
        self.fc_dec = nn.Sequential(nn.Linear(latent, 512), nn.ReLU(), nn.Linear(512, n), nn.ReLU())
        dl, c = [], ch[-1]
        for o in list(ch[-2::-1]) + [8]:
            dl += [nn.ConvTranspose2d(c, o, 3, stride=2, padding=1, output_padding=1), nn.ReLU()]
            c = o
        dl += [nn.Conv2d(c, 1, 3, padding=1)]
        self.dec = nn.Sequential(*dl)
        self.hw = (h, w)

    def encode(self, x):
        return self.fc(self.enc(x))

    def forward(self, x):
        z = self.encode(x)
        y = self.dec(self.fc_dec(z).view(-1, *self.s))
        return F.interpolate(y, size=self.hw, mode="bilinear", align_corners=False), z


class Decoder(nn.Module):
    """GRU-Decoder (Folie 36): getrennte Kompression, Fusion, GRU-Kern, Kopf -> 165 Werte."""

    def __init__(self, use_d, use_l, latent=256, n_state=15, n_hm=165):
        super().__init__()
        self.use_d, self.use_l = use_d, use_l
        n = 64 + 128
        if use_d:
            self.cd = nn.Sequential(nn.Linear(latent, 128), nn.ReLU())
            n += 128
        if use_l:
            self.cl = nn.Sequential(nn.Linear(latent, 128), nn.ReLU())
            n += 128
        self.cs = nn.Sequential(nn.Linear(n_state, 64), nn.ReLU())
        self.ch = nn.Sequential(nn.Linear(n_hm, 128), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(n, 256), nn.LayerNorm(256), nn.ReLU())
        self.gru = nn.GRU(256, 256, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, n_hm))

    def step(self, zd, zl, s, prev, h):
        parts = [self.cs(s), self.ch(prev)]
        if self.use_d:
            parts.append(self.cd(zd))
        if self.use_l:
            parts.append(self.cl(zl))
        x = self.fuse(torch.cat(parts, -1)).unsqueeze(1)
        y, h = self.gru(x, h)
        return self.head(y[:, 0]).clamp(-1.0, 1.0), h


def split_envs(run_id_list, n_total_envs):
    """Ganze Roboter (Lauf, Env) fuer die Validierung zurueckhalten."""
    keys = [(r, e) for r in range(len(run_id_list)) for e in range(run_id_list[r].shape[1])]
    rng = np.random.default_rng(0)
    rng.shuffle(keys)
    nv = max(1, int(len(keys) * args.val_frac))
    return set(keys[nv:]), set(keys[:nv])


# ---------------------------------------------------------------- Stufe 1
def train_ae(kind):
    key = "depth" if kind == "ae_depth" else "lidar"
    files, parts, run_id = load([key])
    tr_keys, va_keys = split_envs(run_id, None)
    arrs = []
    for r, a in enumerate(parts[key]):
        a = crop_depth(a) if key == "depth" else a.reshape(a.shape[0], a.shape[1], 40, -1)
        for e in range(a.shape[1]):
            arrs.append(((r, e) in va_keys, a[:, e]))
    tr = np.concatenate([a for v, a in arrs if not v])
    va = np.concatenate([a for v, a in arrs if v])
    print(f"{key}: train {tr.shape}, val {va.shape}", flush=True)
    h, w = tr.shape[-2:]
    scale = D_MAX if key == "depth" else L_MAX
    model = ConvAE(h, w, args.latent).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    hist = []

    def batches(x, shuffle):
        idx = np.random.permutation(len(x)) if shuffle else np.arange(len(x))
        for i in range(0, len(x), args.bs):
            b = torch.from_numpy(x[idx[i:i + args.bs]].astype(np.float32)).to(dev).unsqueeze(1)
            yield b

    def loss_fn(y, b):
        m = (b > 0).float()
        t = (b / scale).clamp(0, 1)
        return (torch.abs(y - t) * m).sum() / m.sum().clamp(min=1.0)

    best = 1e9
    for ep in range(args.epochs):
        model.train()
        t0, tl, nb = time.time(), 0.0, 0
        for b in batches(tr, True):
            y, _ = model((b / scale).clamp(0, 1))
            loss = loss_fn(y, b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
            nb += 1
        sched.step()
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for b in batches(va, False):
                y, _ = model((b / scale).clamp(0, 1))
                vl += loss_fn(y, b).item()
                vn += 1
        mae_cm = 100 * scale * vl / max(vn, 1)
        hist.append({"epoch": ep, "train_mae_cm": 100 * scale * tl / nb, "val_mae_cm": mae_cm, "s": time.time() - t0})
        print(json.dumps(hist[-1]), flush=True)
        if mae_cm < best:
            best = mae_cm
            torch.save({"state": model.state_dict(), "h": h, "w": w, "latent": args.latent, "scale": scale}, os.path.join(args.out, f"{kind}.pt"))
    json.dump({"stage": kind, "files": files, "best_val_mae_cm": best, "hist": hist}, open(os.path.join(args.out, f"{kind}.json"), "w"), indent=1)
    print(f"FERTIG {kind}: beste Val-MAE {best:.2f} cm", flush=True)


# ---------------------------------------------------------------- Stufe 2
def train_decoder():
    use_d = args.modal in ("depth", "fusion")
    use_l = args.modal in ("lidar", "fusion")
    keys = ["hm", "state", "ep"] + (["depth"] if use_d else []) + (["lidar"] if use_l else [])
    files, parts, run_id = load(keys)
    enc = {}
    for k, use in (("depth", use_d), ("lidar", use_l)):
        if use:
            ck = torch.load(os.path.join(args.out, f"ae_{k}.pt"), map_location=dev)
            m = ConvAE(ck["h"], ck["w"], ck["latent"]).to(dev)
            m.load_state_dict(ck["state"])
            m.eval()
            enc[k] = (m, ck["scale"])
    # Latents vorab berechnen (Encoder eingefroren)
    lat = {k: [] for k in enc}
    with torch.no_grad():
        for r in range(len(files)):
            for k, (m, scale) in enc.items():
                a = parts[k][r]
                a = crop_depth(a) if k == "depth" else a.reshape(a.shape[0], a.shape[1], 40, -1)
                T, N = a.shape[:2]
                flat = a.reshape(T * N, *a.shape[2:])
                zs = []
                for i in range(0, len(flat), 512):
                    b = torch.from_numpy(flat[i:i + 512].astype(np.float32)).to(dev).unsqueeze(1)
                    zs.append(m.encode((b / scale).clamp(0, 1)).cpu())
                lat[k].append(torch.cat(zs).view(T, N, -1))
                parts[k][r] = None  # Speicher frei
    tr_keys, va_keys = split_envs(run_id, None)
    # Sequenzen (Lauf, Env, Start) ohne Reset darin
    seqs = {"tr": [], "va": []}
    for r in range(len(files)):
        epa = parts["ep"][r]
        T, N = epa.shape
        for e in range(N):
            tag = "va" if (r, e) in va_keys else "tr"
            for t0 in range(0, T - args.seq + 1, args.seq // 2 if tag == "tr" else args.seq):
                if epa[t0, e] == epa[t0 + args.seq - 1, e]:
                    seqs[tag].append((r, e, t0))
    print(f"Sequenzen: train {len(seqs['tr'])}, val {len(seqs['va'])}", flush=True)
    cols = list(range(15)) if args.state_cols == "all" else [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14]
    parts["state"] = [s[..., cols] for s in parts["state"]]
    sfx = "" if args.state_cols == "all" else "_real"
    st_all = np.concatenate([s.reshape(-1, s.shape[-1]) for s in parts["state"]])
    s_mu, s_sd = st_all.mean(0), st_all.std(0) + 1e-3
    model = Decoder(use_d, use_l, args.latent, n_state=len(cols)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    def batch(sl):
        hm = torch.stack([torch.from_numpy(parts["hm"][r][t0:t0 + args.seq, e]) for r, e, t0 in sl]).to(dev)
        st = torch.stack([torch.from_numpy((parts["state"][r][t0:t0 + args.seq, e] - s_mu) / s_sd) for r, e, t0 in sl]).float().to(dev)
        zd = torch.stack([lat["depth"][r][t0:t0 + args.seq, e] for r, e, t0 in sl]).to(dev) if use_d else None
        zl = torch.stack([lat["lidar"][r][t0:t0 + args.seq, e] for r, e, t0 in sl]).to(dev) if use_l else None
        return hm, st, zd, zl

    def run(hm, st, zd, zl):
        B = hm.shape[0]
        prev = torch.zeros(B, hm.shape[-1], device=dev)
        h = None
        outs = []
        for t in range(args.seq):
            y, h = model.step(zd[:, t] if zd is not None else None, zl[:, t] if zl is not None else None, st[:, t], prev, h)
            outs.append(y)
            prev = y.detach()
        return torch.stack(outs, 1)

    hist, best = [], 1e9
    for ep in range(args.epochs):
        model.train()
        idx = np.random.permutation(len(seqs["tr"]))
        t0, tl, nb = time.time(), 0.0, 0
        for i in range(0, len(idx), args.bs):
            hm, st, zd, zl = batch([seqs["tr"][j] for j in idx[i:i + args.bs]])
            y = run(hm, st, zd, zl)
            loss = torch.abs(y - hm)[:, 2:].mean()  # erste Schritte ohne Vorgeschichte schwaecher gewichten
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            nb += 1
        sched.step()
        model.eval()
        err = []
        with torch.no_grad():
            for i in range(0, len(seqs["va"]), args.bs):
                hm, st, zd, zl = batch(seqs["va"][i:i + args.bs])
                y = run(hm, st, zd, zl)
                err.append(torch.abs(y - hm)[:, -1].cpu())  # Fehler am Sequenzende (mit Vorgeschichte)
        e_all = torch.cat(err)
        mae_cm = 100 * float(e_all.mean())
        per_point = (100 * e_all.mean(0)).tolist()
        hist.append({"epoch": ep, "train_mae_cm": 100 * tl / nb, "val_mae_cm": mae_cm, "s": time.time() - t0})
        print(json.dumps(hist[-1]), flush=True)
        if mae_cm < best:
            best = mae_cm
            torch.save({"state": model.state_dict(), "modal": args.modal, "seq": args.seq, "s_mu": s_mu, "s_sd": s_sd,
                        "latent": args.latent, "cols": cols}, os.path.join(args.out, f"decoder_{args.modal}{sfx}.pt"))
            best_pp = per_point
    json.dump({"stage": "decoder", "modal": args.modal, "seq": args.seq, "files": files, "best_val_mae_cm": best,
               "val_mae_cm_per_point": best_pp, "hist": hist},
              open(os.path.join(args.out, f"decoder_{args.modal}{sfx}.json"), "w"), indent=1)
    print(f"FERTIG decoder {args.modal}: beste Val-MAE {best:.2f} cm (Folie 37: 2,2-3,0 cm)", flush=True)


if args.stage == "decoder":
    train_decoder()
else:
    train_ae(args.stage)
