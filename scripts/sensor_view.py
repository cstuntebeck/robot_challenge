"""T-002: zwei RealSense D435 an einer Halterung vor dem G1 - Blickfeld-Video.

Baut die Stock-Umgebung G1 rough (Play) nach, haengt eine Halterung mit einer
oder zwei D435 an `pelvis` oder `torso_link`, laesst die veroeffentlichte
NVIDIA-Policy ueber Treppen laufen und schreibt ein Video:
links der Roboter von der Seite, rechts RGB- und Tiefenbild jeder Kamera
(oben die obere Kamera, unten die untere - so wie sie montiert sind).

Aufruf im Container (ueber submit_sensor_view.sh):
    /isaac-sim/python.sh sensor_view.py --mount pelvis --variant beide \
        --checkpoint <checkpoint.pt> --out <ordner> --headless --enable_cameras

Nur die Geometrie der Halterung, ohne Isaac Sim (lokal):
    python3 sensor_view.py --layout-only --mount torso_link

Geometrie (Lesart aus T-002, vom Nutzer bestaetigt):
- untere Kamera: unterer Bildrand senkrecht nach unten -> Neigung 90 - VFOV/2
- obere Kamera: schliesst mit OVERLAP Grad Ueberlappung oben an
- Gehaeuse 90 x 25 x 25 mm (Intel-Datenblatt), optisches Zentrum in der Mitte
  der Frontflaeche (Annahme: der echte Tiefen-Ursprung sitzt im linken Imager)
- hinter den Kameras eine Platte (PLATE), dahinter CLEARANCE Luft zum Koerper:
  das ist der Platz fuer die CAD-Halterung (Stege, Schrauben, Klemme)
"""

import argparse
import json
import math
import os
import sys

# --- D435 und Halterung -------------------------------------------------------
HFOV = 87.0                 # Grad, Masterarbeit S. 22 / Datenblatt
VFOV_TARGET = 55.2          # Grad, untere Angabe der Masterarbeit S. 22
CAM_W, CAM_H = 320, 176     # quadratische Pixel -> VFOV = 2 atan(tan(HFOV/2) * H/W) = 55,1 Grad
CAM_DEPTH, CAM_HEIGHT, CAM_WIDTH = 0.025, 0.025, 0.090   # Gehaeuse [m]
D_MIN, D_MAX = 0.3, 3.0     # Tiefenbereich [m]
OVERLAP = 2.0               # Grad Ueberlappung der beiden Sichtfelder
GAP = 0.010                 # Luft zwischen den Gehaeusen [m]
PLATE = 0.005               # Plattenstaerke hinter den Kameras [m]

# Ohne Messung (falls die Bounding Box aus dem USD nicht gelesen werden kann).
FALLBACK_FRONT = {"pelvis": 0.10, "torso_link": 0.15}
FALLBACK_TORSO_ABOVE_PELVIS = 0.20


def vfov_deg():
    return math.degrees(2 * math.atan(math.tan(math.radians(HFOV / 2)) * CAM_H / CAM_W))


def pitches():
    v = vfov_deg()
    lower = 90.0 - v / 2
    upper = lower - v + OVERLAP
    return lower, upper


def _box_corners(cx, cz, pitch_deg):
    """Ecken des Gehaeuses in der Seitenansicht (x vor, z oben), (cx, cz) = Front-Mitte."""
    p = math.radians(pitch_deg)
    ax = (math.cos(p), -math.sin(p))          # Blickrichtung
    up = (math.sin(p), math.cos(p))           # Gehaeuse-"oben"
    pts = []
    for back in (0.0, CAM_DEPTH):
        for s in (-0.5, 0.5):
            pts.append((cx - back * ax[0] + s * CAM_HEIGHT * up[0],
                        cz - back * ax[1] + s * CAM_HEIGHT * up[1]))
    return pts


def _in_fov(origin, pitch_deg, point, vfov):
    dx, dz = point[0] - origin[0], point[1] - origin[1]
    ang = -math.degrees(math.atan2(dz, dx))   # positiv = nach unten
    return dx > 0 and (pitch_deg - vfov / 2) <= ang <= (pitch_deg + vfov / 2)


def layout(front_x, z_lower, clearance):
    """Lage beider Kameras und der Platte im Rahmen des Montage-Glieds.

    front_x: vorderste Stelle des Koerpers (x im Glied-Rahmen)
    z_lower: Hoehe des optischen Zentrums der unteren Kamera (Glied-Rahmen)
    """
    lower_p, upper_p = pitches()
    v = vfov_deg()
    lo = _box_corners(0.0, 0.0, lower_p)
    up0 = _box_corners(0.0, 0.0, upper_p)
    dz = max(z for _, z in lo) - min(z for _, z in up0) + GAP
    up = _box_corners(0.0, dz, upper_p)
    # Sieht eine Kamera das andere Gehaeuse?
    conflicts = [f"obere sieht untere ({x:.3f},{z:.3f})" for x, z in lo if _in_fov((0, dz), upper_p, (x, z), v)]
    conflicts += [f"untere sieht obere ({x:.3f},{z:.3f})" for x, z in up if _in_fov((0, 0), lower_p, (x, z), v)]
    rear = min(x for x, _ in lo + up)
    shift = front_x + clearance + PLATE - rear       # hinterste Gehaeusekante -> vor Platte
    z_all = [z for _, z in lo + up]
    plate = {"x": front_x + clearance + PLATE / 2,
             "z": z_lower + (min(z_all) + max(z_all)) / 2,
             "size": (PLATE, CAM_WIDTH + 0.01, max(z_all) - min(z_all) + 0.01)}

    def cam(px, pz, pitch):
        p = math.radians(pitch)
        # Gehaeusemitte liegt CAM_DEPTH/2 hinter der Front-Mitte, entlang der Blickachse
        return {"pitch_deg": pitch, "front": (px, 0.0, pz),
                "box_center": (px - CAM_DEPTH / 2 * math.cos(p), 0.0, pz + CAM_DEPTH / 2 * math.sin(p))}

    return {
        "vfov_deg": v, "front_x": front_x, "clearance": clearance,
        "unten": cam(shift, z_lower, lower_p),
        "oben": cam(shift, z_lower + dz, upper_p),
        "platte": plate,
        "abstand_optische_zentren_mm": round(1000 * math.hypot(0, dz), 1),
        "abstand_front_zu_koerper_mm": round(1000 * (shift - front_x), 1),
        "sichtkonflikte": conflicts,
    }


def pitch_quat(pitch_deg):
    """(w, x, y, z) fuer Neigung um +y: positiv = Blick nach unten (x -> -z)."""
    h = math.radians(pitch_deg) / 2
    return (math.cos(h), 0.0, math.sin(h), 0.0)


# --- Argumente ----------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--mount", choices=["pelvis", "torso_link"], required=True)
parser.add_argument("--variant", choices=["unten", "beide"], default="beide")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--out", default=".")
parser.add_argument("--steps", type=int, default=750, help="Policy-Schritte (50 Hz), Default 15 s")
parser.add_argument("--speed", type=float, default=0.6, help="Vorwaerts-Kommando [m/s]")
parser.add_argument("--clearance", type=float, default=0.02, help="Luft zwischen Koerper und Platte [m]")
parser.add_argument("--torso_height", type=float, default=1.0,
                    help="Hoehe der unteren Kamera ueber Grund bei Montage am torso_link [m]")
parser.add_argument("--layout-only", action="store_true")
parser.add_argument("--front_x", type=float, default=None, help="nur mit --layout-only")

if "--layout-only" in sys.argv:
    a = parser.parse_args()
    fx = a.front_x if a.front_x is not None else FALLBACK_FRONT[a.mount]
    print(json.dumps(layout(fx, 0.0, a.clearance), indent=2))
    sys.exit(0)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import cpr.offline_assets as offline_assets  # noqa: E402

offline_assets.patch_launcher_args(args_cli)        # VOR AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
offline_assets.apply()                              # nach App-Start, vor isaaclab-Modulen

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    G1RoughPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (  # noqa: E402
    G1RoughEnvCfg_PLAY,
)

TASK = "Isaac-Velocity-Rough-G1-Play-v0"


def measure_robot(usd_path, mount):
    """Vorderkante des Montage-Glieds und Lage torso_link/pelvis aus der USD-Datei."""
    info = {"usd": usd_path, "gemessen": False}
    try:
        stage = Usd.Stage.Open(usd_path)
        prims = {p.GetName(): p for p in stage.Traverse() if p.GetName() in ("pelvis", "torso_link")}
        xc = UsdGeom.XformCache()
        bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                               useExtentsHint=False)
        rng = bc.ComputeUntransformedBound(prims[mount]).ComputeAlignedRange()
        if rng.IsEmpty():
            raise RuntimeError(f"leere Bounding Box fuer {mount}")
        m_p = xc.GetLocalToWorldTransform(prims["pelvis"])
        m_t = xc.GetLocalToWorldTransform(prims["torso_link"])
        rel = m_t * m_p.GetInverse()
        info.update({
            "gemessen": True,
            "prim": str(prims[mount].GetPath()),
            "bbox_min": list(rng.GetMin()), "bbox_max": list(rng.GetMax()),
            "torso_in_pelvis": list(rel.ExtractTranslation()),
            "pelvis_rot_ident": bool(m_p.ExtractRotationMatrix().GetOrthonormalized() == m_p.ExtractRotationMatrix()),
        })
    except Exception as exc:  # noqa: BLE001
        info["fehler"] = repr(exc)
    return info


def depth_to_rgb(d):
    """Tiefe [m] -> Farbe. Rot: naeher als 0,3 m (D435 misst dort nicht). Schwarz: > 3 m / nichts."""
    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf)
    out = np.zeros(d.shape + (3,), dtype=np.uint8)
    ok = (d >= D_MIN) & (d <= D_MAX)
    t = np.clip((d - D_MIN) / (D_MAX - D_MIN), 0, 1)
    anchors = np.array([[255, 230, 60], [60, 200, 160], [40, 90, 220], [20, 20, 90]], dtype=float)
    pos = t * (len(anchors) - 1)
    i = np.clip(pos.astype(int), 0, len(anchors) - 2)
    f = (pos - i)[..., None]
    col = anchors[i] * (1 - f) + anchors[i + 1] * f
    out[ok] = col[ok].astype(np.uint8)
    out[(d > 0) & (d < D_MIN)] = (230, 30, 30)
    return out


def label(img, text):
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        ImageDraw.Draw(im).text((4, 3), text, fill=(255, 255, 255))
        return np.asarray(im)
    except Exception:  # noqa: BLE001
        return img


def strip(text, width, h=18):
    return label(np.full((h, width, 3), 40, dtype=np.uint8), text)


def main():
    out = os.path.abspath(args_cli.out)
    os.makedirs(out, exist_ok=True)
    tag = f"{args_cli.mount}_{args_cli.variant}"
    result = {"tag": tag, "mount": args_cli.mount, "variant": args_cli.variant, "ok": False}

    env_cfg = G1RoughEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 42
    # Nur Treppen: Pyramide nach innen (aus der Mitte heraus aufwaerts, im naechsten Feld abwaerts)
    tg = ROUGH_TERRAINS_CFG.replace(num_rows=3, num_cols=3, curriculum=False, difficulty_range=(0.4, 0.6))
    tg.sub_terrains = {"stairs_up": ROUGH_TERRAINS_CFG.sub_terrains["pyramid_stairs_inv"].replace(proportion=1.0)}
    env_cfg.scene.terrain.terrain_generator = tg
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.speed, args_cli.speed)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (0.9, -2.6, 0.7)
    env_cfg.viewer.lookat = (0.9, 0.0, 0.0)
    env_cfg.viewer.resolution = (960, 540)

    # Lage der Halterung aus der Robotergeometrie
    robot = measure_robot(env_cfg.scene.robot.spawn.usd_path, args_cli.mount)
    result["robot_usd"] = robot
    front = robot["bbox_max"][0] if robot["gemessen"] else FALLBACK_FRONT[args_cli.mount]
    if args_cli.mount == "pelvis":
        z_lower = 0.0                                  # Hoehe des Beckens
    else:
        torso_above = robot["torso_in_pelvis"][2] if robot["gemessen"] else FALLBACK_TORSO_ABOVE_PELVIS
        pelvis_z = env_cfg.scene.robot.init_state.pos[2]
        z_lower = args_cli.torso_height - pelvis_z - torso_above
    lay = layout(front, z_lower, args_cli.clearance)
    result["layout"] = lay
    print("[T-002] Halterung:", json.dumps(lay, indent=1))

    link = "{ENV_REGEX_NS}/Robot/" + args_cli.mount
    cams = ["unten"] + (["oben"] if args_cli.variant == "beide" else [])
    for name in cams:
        c = lay[name]
        setattr(env_cfg.scene, f"cam_{name}", TiledCameraCfg(
            prim_path=f"{link}/t002_cam_{name}",
            update_period=0.0, width=CAM_W, height=CAM_H,
            data_types=["rgb", "distance_to_image_plane"],
            # ohne das liefert data.pos_w/quat_w_world nur die Startlage (camera_cfg.py:77-82, F-013)
            update_latest_camera_pose=True,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0,
                horizontal_aperture=2 * 10.0 * math.tan(math.radians(HFOV / 2)),
                clipping_range=(0.05, 20.0)),
            offset=TiledCameraCfg.OffsetCfg(pos=c["front"], rot=pitch_quat(c["pitch_deg"]), convention="world"),
        ))
    # Geschwindigkeitspfeile ueber dem Kopf aus (velocity_env_cfg.py:101)
    env_cfg.commands.base_velocity.debug_vis = False

    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array")
    agent_cfg = G1RoughPPORunnerCfg()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    scene = env.unwrapped.scene
    dev = env.unwrapped.device
    rob = scene["robot"]
    link_id = rob.find_bodies(args_cli.mount)[0][0]
    # ein Ausdruck statt Liste: find_bodies bricht ab, wenn ein Listeneintrag nichts trifft (utils/string.py:230)
    near_ids, near_names = rob.find_bodies(".*(knee|elbow|wrist|hand|ankle_roll).*")
    scanner = scene["height_scanner"]

    # Gehaeuse und Platte: als Kinder des Glieds liefen sie nicht mit (Job 8790560, F-013).
    # Deshalb Marker, deren Weltlage jeden Schritt aus der Glied-Lage gerechnet wird.
    parts = {"unten": (lay["unten"]["box_center"], pitch_quat(lay["unten"]["pitch_deg"]),
                       (CAM_DEPTH, CAM_WIDTH, CAM_HEIGHT), (0.1, 0.35, 0.8))}
    if "oben" in cams:
        parts["oben"] = (lay["oben"]["box_center"], pitch_quat(lay["oben"]["pitch_deg"]),
                         (CAM_DEPTH, CAM_WIDTH, CAM_HEIGHT), (0.9, 0.5, 0.1))
    pl = lay["platte"]
    parts["platte"] = ((pl["x"], 0.0, pl["z"]), (1.0, 0.0, 0.0, 0.0), pl["size"], (0.25, 0.25, 0.25))
    names = list(parts)
    markers = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/T002_Halterung",
        markers={n: sim_utils.CuboidCfg(size=parts[n][2],
                                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=parts[n][3]))
                 for n in names}))
    loc_pos = torch.tensor([parts[n][0] for n in names], device=dev, dtype=torch.float32)
    loc_quat = torch.tensor([parts[n][1] for n in names], device=dev, dtype=torch.float32)
    cam_loc_pos = {n: torch.tensor([lay[n]["front"]], device=dev, dtype=torch.float32) for n in cams}
    cam_loc_quat = {n: torch.tensor([pitch_quat(lay[n]["pitch_deg"])], device=dev, dtype=torch.float32) for n in cams}
    half = torch.tensor([CAM_DEPTH, CAM_WIDTH, CAM_HEIGHT], device=dev) / 2
    fwd = torch.tensor([[1.0, 0.0, 0.0]], device=dev)

    def place_parts():
        lp = rob.data.body_pos_w[0, link_id].unsqueeze(0)
        lq = rob.data.body_quat_w[0, link_id].unsqueeze(0)
        pos = lp + quat_apply(lq.expand(len(names), 4), loc_pos)
        quat = quat_mul(lq.expand(len(names), 4), loc_quat)
        markers.visualize(translations=pos, orientations=quat, marker_indices=list(range(len(names))))
        return lp, lq, pos, quat

    frames = []
    stats = {n: {"pitch": [], "height": [], "pose_diff": []} for n in cams}
    min_dist = {}
    stills = {int(s * 50) for s in (2, 6, 10)}
    obs = env.get_observations()
    place_parts()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        lp, lq, ppos, pquat = place_parts()
        hits = scanner.data.ray_hits_w[0]
        for n in cams:
            # Massgeblich: Glied-Lage + Offset (die Kamera haengt per Konstruktion am Glied)
            cpos = (lp + quat_apply(lq, cam_loc_pos[n]))[0]
            cquat = quat_mul(lq, cam_loc_quat[n])
            f = quat_apply(cquat, fwd)[0]
            stats[n]["pitch"].append(math.degrees(math.asin(max(-1.0, min(1.0, -float(f[2]))))))
            # Boden direkt unter der Kamera: naechster Strahl des height_scanner (Raster 1,6 x 1,0 m)
            k = torch.argmin(torch.linalg.norm(hits[:, :2] - cpos[:2], dim=-1))
            stats[n]["height"].append(float(cpos[2] - hits[k, 2]))
            # Gegenprobe: was die Sensor-API meldet (update_latest_camera_pose=True)
            stats[n]["pose_diff"].append(float(torch.linalg.norm(cpos - scene[f"cam_{n}"].data.pos_w[0])))
            # Abstand Glied-Ursprung -> Gehaeuse (Quader). Grob: Ursprung, nicht Oberflaeche des Glieds
            i = names.index(n)
            rel = quat_apply_inverse(pquat[i:i + 1].expand(len(near_ids), 4),
                                     rob.data.body_pos_w[0, near_ids] - ppos[i:i + 1])
            dist = torch.linalg.norm(torch.clamp(rel.abs() - half, min=0.0), dim=-1)
            for name, dv in zip(near_names, dist.tolist()):
                key = f"{n}:{name}"
                min_dist[key] = min(min_dist.get(key, 9.9), dv)
        if step % 2:
            continue
        view = np.asarray(env.unwrapped.render())[..., :3]
        rows = []
        for n in ("oben", "unten"):
            if n in cams:
                d = scene[f"cam_{n}"].data.output
                rgb = d["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
                dep = depth_to_rgb(d["distance_to_image_plane"][0, ..., 0].cpu().numpy())
                title = (f"{n}e Kamera  Neigung {lay[n]['pitch_deg']:.1f} Grad  Hoehe {stats[n]['height'][-1]:.2f} m"
                         f"  | RGB | Tiefe 0.3-3 m (rot: <0.3 m)")
                rows += [strip(title, 2 * CAM_W), np.concatenate([rgb, dep], axis=1)]
            else:
                rows += [strip("obere Kamera: in dieser Variante nicht montiert", 2 * CAM_W),
                         np.full((CAM_H, 2 * CAM_W, 3), 25, dtype=np.uint8)]
        right = np.concatenate(rows, axis=0)
        pad = view.shape[0] - right.shape[0]
        right = np.concatenate([right, np.full((pad, right.shape[1], 3), 15, dtype=np.uint8)], axis=0)
        info = f"G1 rough, NVIDIA-Policy, {args_cli.speed} m/s | Halterung an {args_cli.mount} | t={step / 50:.1f} s"
        frame = np.concatenate([label(view.copy(), info), right], axis=1)
        frames.append(frame)
        if step in stills:
            try:
                from PIL import Image
                Image.fromarray(frame).save(os.path.join(out, f"{tag}_t{step // 50:02d}s.png"))
            except Exception as exc:  # noqa: BLE001
                print("[T-002] Standbild fehlgeschlagen:", exc)
        if bool(dones[0]):
            result.setdefault("resets", []).append(step)

    video = os.path.join(out, f"{tag}.mp4")
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    ImageSequenceClip(frames, fps=25).write_videofile(video, logger=None)

    def span(xs):
        xs = xs[25:] or xs                      # erste 0,5 s (Absetzen) ausgelassen
        return {"min": round(min(xs), 3), "max": round(max(xs), 3), "mittel": round(sum(xs) / len(xs), 3)}

    # Plausibilitaet (F-013): beim Gehen MUSS sich die Kameralage aendern.
    problems = []
    for n in cams:
        s = span(stats[n]["pitch"])
        h = span(stats[n]["height"])
        if s["max"] - s["min"] < 0.1 or h["max"] - h["min"] < 0.005:
            problems.append(f"{n}: Kameralage aendert sich nicht (Neigung {s}, Hoehe {h})")

    # Weicht die Sensor-API ab, wird das gemeldet, macht den Lauf aber nicht ungueltig:
    # die Bilder zeigen, dass die Kamera dem Glied folgt (Job 8790560).
    result.update({
        "ok": os.path.getsize(video) > 0 and not problems,
        "probleme": problems,
        "video": video, "frames": len(frames), "steps": args_cli.steps,
        "vfov_deg": round(vfov_deg(), 2),
        "kamera_neigung_welt_grad": {n: span(stats[n]["pitch"]) for n in cams},
        "kamera_hoehe_ueber_boden_m": {n: span(stats[n]["height"]) for n in cams},
        "sensor_api_abweichung_max_m": {n: round(max(stats[n]["pose_diff"]), 4) for n in cams},
        "min_abstand_glied_ursprung_zu_gehaeuse_m": {
            k: round(v, 3) for k, v in sorted(min_dist.items(), key=lambda kv: kv[1])[:8]},
    })
    with open(os.path.join(out, f"{tag}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print("[T-002] ERGEBNIS", json.dumps(result))
    # Sonst feuert der Mitlauf-Callback der Ansichtskamera nach dem Schliessen noch einmal
    # (viewport_camera_controller.py:226-231) und wirft einen AttributeError ins Log.
    if env.unwrapped.viewport_camera_controller is not None:
        env.unwrapped.viewport_camera_controller.update_view_to_world()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
