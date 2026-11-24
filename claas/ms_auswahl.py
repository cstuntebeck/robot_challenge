"""Meilenstein (Claas): besten Lauf auswaehlen (select) bzw. Abschlussbericht schreiben (report).

Nur Python-Standardbibliothek. Liest die Slurm-Logs der Trainingsjobs.

    python ms_auswahl.py select  B1=<job> ... B5=<job> BASELINE=<ckpt>
    python ms_auswahl.py report  B1=<job> B2=<job> B3=<job> C=<job> [V1=<job> ...]

select schreibt runs/claas/ms_auswahl.{env,json,md}; Lauf C liest die .env.
report schreibt runs/claas/MEILENSTEIN.md und .json.
"""

import glob
import json
import os
import re
import sys
from datetime import datetime

BASE = "/bigwork/nhkbrc2a/isaac"
RUNS = f"{BASE}/runs"
OUT = f"{RUNS}/claas"
KEYS = {
    "reward": r"Mean reward:\s*([-\d.]+)",
    "ep_len": r"Mean episode length:\s*([-\d.]+)",
    "falls": r"Episode_Termination/bad_orientation:\s*([-\d.]+)",
    "terrain": r"Curriculum/terrain_levels:\s*([-\d.]+)",
    "cmd": r"Curriculum/lin_vel_cmd_levels:\s*([-\d.]+)",
    "push": r"Curriculum/dr_level:\s*([-\d.]+)",
    "err_xy": r"Metrics/base_velocity/error_vel_xy:\s*([-\d.]+)",
    "track": r"Episode_Reward/track_lin_vel_xy:\s*([-\d.]+)",
}
# Aufgabe des Laufs: Name, Task fuer Lauf C, falls er gewinnt
ROLE = {
    "B1": ("Treppen, Unitree-Rewards", "Unitree-G1-Claas-MS-C-Stairs"),
    "B2": ("Treppen, Tracking-Gewicht 2,0", "Unitree-G1-Claas-MS-C-StairsTrack"),
    "B3": ("eben, adaptive DR", "Unitree-G1-Claas-MS-C-StairsV2"),
    "B4": ("Treppen, Curricula V2", "Unitree-G1-Claas-MS-C-StairsV2"),
    "B5": ("Treppen, Curricula V2, Tracking-Gewicht 2,0", "Unitree-G1-Claas-MS-C-StairsV2Track"),
    "C": ("Lauf C (Weitertraining des Besten)", None),
}


def run_dir(log_text):
    root = re.findall(r"Logging experiment in directory:\s*(\S+)", log_text)
    name = re.findall(r"Exact experiment name requested from command line:\s*(\S+)", log_text)
    return f"{root[-1]}/{name[-1]}" if root and name else None


def last_ckpt(d):
    if not d:
        return None
    files = glob.glob(f"{d}/model_*.pt")
    nums = [(int(m.group(1)), f) for f in files if (m := re.search(r"model_(\d+)\.pt$", f))]
    return max(nums)[1] if nums else None


def metrics(job, window=100):
    log = f"{BASE}/logs/slurm_claas_ms_{job}.out"
    m = {"job": job, "log": log, "ok": False}
    if not os.path.exists(log):
        m["fehler"] = "Log fehlt"
        return m
    text = open(log, errors="replace").read()
    blocks = text.split("Learning iteration ")[1:]
    m["iterationen"] = len(blocks)
    if blocks:
        m["iteration_zuletzt"] = blocks[-1].split()[0]
    tail = blocks[-window:]
    for k, rx in KEYS.items():
        vals = [float(x.group(1)) for b in tail if (x := re.search(rx, b))]
        m[k] = round(sum(vals) / len(vals), 4) if vals else None
    d = run_dir(text)
    m["run_dir"] = d
    m["checkpoint"] = last_ckpt(d)
    m["beendet"] = "TRAIN_DONE" in text
    m["ok"] = bool(m["checkpoint"]) and len(blocks) >= 200
    return m


def score(m):
    """Treppen-Eignung: Gelaende 50 %, Standfestigkeit 35 %, Tracking 15 %."""
    t = min((m.get("terrain") or 0) / 9.0, 1.0)
    f = m.get("falls") if m.get("falls") is not None else 1.0
    e = m.get("err_xy") if m.get("err_xy") is not None else 1.0
    return round(0.5 * t + 0.35 * (1 - f) + 0.15 * max(0.0, 1 - e), 4)


def fmt(v, nd=2):
    return "–" if v is None else f"{v:.{nd}f}".replace(".", ",")


def table(ms):
    rows = ["| Lauf | Ziel | Iterationen | Reward | Stürze | Terrain-Level | Befehl max. m/s | Stoß-Stufe m/s | Tracking-Fehler m/s | Score |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in ms.items():
        rows.append(
            f"| {name} (Job {m['job']}) | {ROLE[name][0]} | {m.get('iterationen', 0)} | {fmt(m.get('reward'), 1)} | "
            f"{fmt(m['falls'] * 100, 0) + ' %' if m.get('falls') is not None else '–'} | {fmt(m.get('terrain'))} | "
            f"{fmt(m.get('cmd'))} | {fmt(m.get('push'))} | {fmt(m.get('err_xy'))} | {fmt(m.get('score'), 3)} |")
    return "\n".join(rows)


def select(args):
    ms = {k: metrics(args[k]) for k in ("B1", "B2", "B3", "B4", "B5") if k in args}
    for m in ms.values():
        m["score"] = score(m) if m["ok"] else None
    stairs = {k: m for k, m in ms.items() if k in ("B1", "B2", "B4", "B5") and m["ok"] and m["falls"] is not None and m["falls"] < 0.6}
    if stairs:
        win = max(stairs, key=lambda k: stairs[k]["score"])
        reason = (f"{win} gewinnt (Score {stairs[win]['score']}): bester Treppen-Lauf mit < 60 % Stuerzen. "
                  f"Lauf C trainiert dessen letzten Checkpoint mit derselben Konfiguration weiter.")
        c_task, c_ckpt = ROLE[win][1], stairs[win]["checkpoint"]
    elif ms.get("B3", {}).get("ok"):
        win = "B3"
        reason = ("Kein Treppen-Lauf unter 60 % Stuerzen. Lauf C startet vom robusten ebenen Lauf B3 "
                  "und lernt Treppen mit der Version-2-Konfiguration (B4).")
        c_task, c_ckpt = ROLE["B3"][1], ms["B3"]["checkpoint"]
    else:
        win = "Baseline"
        reason = "Alle Laeufe B1-B3 ohne verwertbares Ergebnis. Lauf C startet von der Baseline 3000 (Version-2-Konfiguration)."
        c_task, c_ckpt = "Unitree-G1-Claas-MS-C-StairsV2", args["BASELINE"]
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/ms_auswahl.env", "w") as fh:
        fh.write(f'C_TASK="{c_task}"\nC_INIT_CKPT="{c_ckpt}"\nC_WINNER="{win}"\nC_REASON="{reason}"\n')
    res = {"zeit": datetime.now().isoformat(timespec="seconds"), "gewinner": win, "grund": reason,
           "c_task": c_task, "c_init": c_ckpt, "laeufe": ms}
    json.dump(res, open(f"{OUT}/ms_auswahl.json", "w"), indent=2, ensure_ascii=False)
    md = (f"# Meilenstein – Auswahl nach Stufe 1 ({res['zeit']})\n\n{table(ms)}\n\n"
          f"**Gewinner:** {win}\n\n{reason}\n\nStart-Checkpoint für Lauf C: `{c_ckpt}`\n")
    open(f"{OUT}/ms_auswahl.md", "w").write(md)
    print(md)


ORDER = ("StairsEasy", "StairsMid", "FlatPush")


def abgabe_video(c_job):
    """Videos des Endstands (Lauf C) zu einem Abgabe-Video zusammenfuegen (moviepy aus dem Container)."""
    if not c_job:
        return None
    parts = [f"{RUNS}/videos/ms_C_{c_job}_{s}.mp4" for s in ORDER]
    parts = [p for p in parts if os.path.exists(p)]
    if not parts:
        print("Abgabe-Video: keine Videos von Lauf C gefunden")
        return None
    try:
        try:
            from moviepy import VideoFileClip, concatenate_videoclips  # moviepy >= 2
        except ImportError:
            from moviepy.editor import VideoFileClip, concatenate_videoclips  # moviepy 1.x
        out = f"{RUNS}/videos/MEILENSTEIN_Abgabe_{c_job}.mp4"
        clips = [VideoFileClip(p) for p in parts]
        concatenate_videoclips(clips, method="compose").write_videofile(out, fps=50, logger=None)
        print("Abgabe-Video:", out)
        return out
    except Exception as exc:  # noqa: BLE001
        print("Abgabe-Video fehlgeschlagen:", repr(exc))
        return None


def report(args):
    ms = {k: metrics(args[k]) for k in ("B1", "B2", "B3", "B4", "B5", "C") if k in args}
    for m in ms.values():
        m["score"] = score(m) if m["ok"] else None
    sel = {}
    if os.path.exists(f"{OUT}/ms_auswahl.json"):
        sel = json.load(open(f"{OUT}/ms_auswahl.json"))
    vids = sorted(glob.glob(f"{RUNS}/videos/ms_*.mp4"))
    link = "https://login.cluster.uni-hannover.de/pun/sys/dashboard/files/fs/"
    vlines = "\n".join(f"- [{os.path.basename(v)}]({link}/{v.lstrip('/')})" for v in vids) or "- (keine)"
    c = ms.get("C", {})
    md = (f"# Meilenstein 18./19.11. – Abschlussbericht ({datetime.now().isoformat(timespec='minutes')})\n\n"
          f"## Ergebnisse (Mittel der letzten 100 Iterationen)\n\n{table(ms)}\n\n"
          f"## Auswahl nach Stufe 1\n\n{sel.get('grund', '–')}\n\n"
          f"## Endstand – Grundlage für alles Weitere\n\n"
          f"- Checkpoint: `{c.get('checkpoint') or '–'}`\n- Task-Konfiguration: `{sel.get('c_task', '–')}` "
          f"(cpr/claas_milestone.py)\n- Weitertrainieren: `sbatch --export=ALL,TASK=<task>,INIT_CKPT=<checkpoint>,MAX_ITER=<n> "
          f"claas/ms_train.sh`\n\n## Videos\n\n{vlines}\n")
    abgabe = abgabe_video(args.get("C"))
    if abgabe:
        md += f"\n## Abgabe-Video\n\n- [{os.path.basename(abgabe)}]({link}/{abgabe.lstrip('/')})\n"
    open(f"{OUT}/MEILENSTEIN.md", "w").write(md)
    json.dump({"laeufe": ms, "auswahl": sel, "videos": vids}, open(f"{OUT}/MEILENSTEIN.json", "w"), indent=2,
              ensure_ascii=False)
    print(md)


if __name__ == "__main__":
    mode = sys.argv[1]
    kv = dict(a.split("=", 1) for a in sys.argv[2:])
    {"select": select, "report": report}[mode](kv)
