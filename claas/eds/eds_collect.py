"""CD2 / EDS: Datensatz fuer den Hoehenkarten-Schaetzer aufnehmen - Claas, 23.11.

Ein Lehrer (exakte Hoehenkarte) laeuft im Treppen-Gelaende; pro 0,1 s (10 Hz, wie die
Hoehenkarte in der Vorlesung) wird gespeichert:
  depth  (N, H, W)   Tiefenbild der unteren D435 an torso_link [m], float16 (0 = kein Wert)
  lidar  (N, C*K)    Entfernungen eines MID-360-artigen LiDARs am Kopf [m], float16 (0 = kein Treffer)
  hm     (N, 165)    exakte Hoehenkarte im EDS-Raster (Ziel, HM T), wie die Policy sie sieht
  hm_z   (N, 165)    Bodenhoehe je Rasterpunkt relativ zum Becken [m] (Rohwert)
  state  (N, 15)     Roboterzustand: Drehrate(3), Schwerkraft(3), Befehl(3), lin. Geschw.(3), Beckenhoehe ueber Boden(1), Roll/Nick(2)
  joints (N, 46)     Gelenkpositionen relativ (23) und -geschwindigkeiten (23)
  pose   (N, 7)      Beckenlage Welt (x y z, qw qx qy qz) - fuer Drift/Odometrie-Tests
  ep     (N,)        Episodennummer je Roboter (Sequenzen nicht ueber Resets hinweg bilden!)
Dateien: <out>/shard_XXX.npz mit Achsen (Zeit, Roboter, ...), dazu meta.json und preview.png.

Annahmen (mit Dennis abgleichen): Kamera wie KAMERA_D435.md (untere Kamera an torso_link,
unterer Bildrand senkrecht nach unten), Bild 212x120 (quadr. Pixel, 87 x 56,5 Grad, ~ echte D435),
fuer 160x120 mittig zuschneiden; LiDAR 40 Kanaele x 276 Spalten (Folie 34), -52..+7 Grad.
"""

import argparse
import copy
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip  (aus Unitrees scripts/rsl_rl, Pfad setzt unitree_run.py)

parser = argparse.ArgumentParser(description="EDS-Datensatz aufnehmen")
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--out", required=True)
parser.add_argument("--seconds", type=float, default=300.0, help="Simulierte Zeit")
parser.add_argument("--hz", type=float, default=10.0)
parser.add_argument("--shard", type=int, default=500, help="Zeitschritte je Datei")
parser.add_argument("--cam_w", type=int, default=212)
parser.add_argument("--cam_h", type=int, default=120)
parser.add_argument("--no_lidar", action="store_true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

from cpr import claas_g1_23dof_eds as eds  # noqa: E402

HFOV = 87.0
D_MIN, D_MAX = 0.2, 3.0
CAM_LINK = "torso_link"
CAM_POS = (0.20, 0.0, 0.156)  # wie sensor_view.py (Brust, untere Kamera), grob fuer 23 DoF
LIDAR_POS = (0.0, 0.0, 0.42)  # MID-360 im Kopf, ueber torso_link
LIDAR_CH, LIDAR_COLS, LIDAR_VFOV = 40, 276, (-52.0, 7.0)


def vfov_deg(w, h):
    return math.degrees(2 * math.atan(math.tan(math.radians(HFOV / 2)) * h / w))


def pitch_quat(pitch_deg):
    h = math.radians(pitch_deg) / 2
    return (math.cos(h), 0.0, math.sin(h), 0.0)


def main():
    out = os.path.abspath(args_cli.out)
    os.makedirs(out, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=True, entry_point_key="play_env_cfg_entry_point")
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Ziel: exaktes Hoehenraster im EDS-Format
    eds._eds_scanner(env_cfg)
    # Kamera
    W, H = args_cli.cam_w, args_cli.cam_h
    vf = vfov_deg(W, H)
    pitch = 90.0 - vf / 2
    env_cfg.scene.eds_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + CAM_LINK + "/eds_cam",
        update_period=0.0, width=W, height=H, data_types=["distance_to_image_plane"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(focal_length=10.0,
                                         horizontal_aperture=2 * 10.0 * math.tan(math.radians(HFOV / 2)),
                                         clipping_range=(0.05, 20.0)),
        offset=TiledCameraCfg.OffsetCfg(pos=CAM_POS, rot=pitch_quat(pitch), convention="world"),
    )
    # LiDAR
    if not args_cli.no_lidar:
        lid = copy.deepcopy(env_cfg.scene.height_scanner)
        lid.prim_path = "{ENV_REGEX_NS}/Robot/" + CAM_LINK
        lid.offset = RayCasterCfg.OffsetCfg(pos=LIDAR_POS)
        lid.pattern_cfg = patterns.LidarPatternCfg(channels=LIDAR_CH, vertical_fov_range=LIDAR_VFOV,
                                                   horizontal_fov_range=(-180.0, 180.0),
                                                   horizontal_res=360.0 / LIDAR_COLS)
        lid.max_distance = 20.0
        if getattr(lid, "attach_yaw_only", None):
            lid.attach_yaw_only = False
        if hasattr(lid, "ray_alignment"):
            lid.ray_alignment = "base"
        lid.debug_vis = False
        env_cfg.scene.eds_lidar = lid
    env_cfg.commands.base_velocity.debug_vis = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    scene = u.scene
    rob = scene["robot"]
    dev = u.device
    n = u.num_envs
    period = max(1, round(1.0 / (args_cli.hz * u.step_dt)))
    steps = int(args_cli.seconds / u.step_dt)
    sc_eds = scene["height_scanner_eds"]
    ep = torch.zeros(n, dtype=torch.int32, device=dev)

    meta = {"task": args_cli.task, "checkpoint": args_cli.checkpoint, "num_envs": n, "step_dt": u.step_dt,
            "period_steps": period, "hz": 1.0 / (period * u.step_dt), "cam": {"w": W, "h": H, "hfov": HFOV,
            "vfov": vf, "pitch_deg": pitch, "link": CAM_LINK, "pos": CAM_POS, "d_min": D_MIN, "d_max": D_MAX},
            "lidar": None if args_cli.no_lidar else {"channels": LIDAR_CH, "cols": LIDAR_COLS,
            "vfov": LIDAR_VFOV, "pos": LIDAR_POS},
            "hm": {"res": eds.EDS_RES, "size": eds.EDS_SIZE, "center_x": eds.EDS_CENTER_X,
                   "ray_starts_xy": sc_eds.ray_starts[0, :, :2].cpu().tolist()},
            "state": ["angvel_x", "angvel_y", "angvel_z", "grav_x", "grav_y", "grav_z", "cmd_vx", "cmd_vy",
                      "cmd_wz", "linvel_x", "linvel_y", "linvel_z", "pelvis_h", "roll", "pitch"],
            "shards": []}
    buf = {k: [] for k in ("depth", "lidar", "hm", "hm_z", "state", "joints", "pose", "ep")}
    shard_i = 0

    def flush():
        nonlocal shard_i
        if not buf["hm"]:
            return
        path = os.path.join(out, f"shard_{shard_i:03d}.npz")
        np.savez(path, **{k: np.stack(v) for k, v in buf.items() if v})
        meta["shards"].append(os.path.basename(path))
        print(f"[eds_collect] {path}: {len(buf['hm'])} Zeitschritte x {n} Roboter", flush=True)
        for v in buf.values():
            v.clear()
        shard_i += 1

    obs = env.get_observations()
    falls = 0
    for k in range(steps):
        with torch.inference_mode():
            obs, _, dones, extras = env.step(policy(obs))
        d = dones.bool()
        falls += int(d.sum())
        ep += d.int()
        if k % period:
            continue
        with torch.inference_mode():
            dep = scene["eds_cam"].data.output["distance_to_image_plane"][..., 0]
            dep = torch.nan_to_num(dep, nan=0.0, posinf=0.0, neginf=0.0)
            dep = torch.where((dep >= D_MIN) & (dep <= D_MAX), dep, torch.zeros_like(dep))
            buf["depth"].append(dep.half().cpu().numpy())
            if not args_cli.no_lidar:
                ld = scene["eds_lidar"].data
                r = torch.linalg.norm(ld.ray_hits_w - ld.pos_w.unsqueeze(1), dim=-1)
                r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
                r = torch.where(r <= 20.0, r, torch.zeros_like(r))
                buf["lidar"].append(r.half().cpu().numpy())
            buf["hm"].append(eds.height_scan_eds_exact(u, eds.EDS).cpu().numpy().astype(np.float32))
            hz = sc_eds.data.ray_hits_w[..., 2] - rob.data.root_pos_w[:, 2:3]
            buf["hm_z"].append(torch.nan_to_num(hz, nan=-2.0, posinf=-2.0, neginf=-2.0).cpu().numpy())
            ground = torch.nan_to_num(sc_eds.data.ray_hits_w[..., 2], nan=0.0).mean(dim=1)
            roll, pit, _ = euler_xyz_from_quat(rob.data.root_quat_w)
            wrap = lambda a: torch.atan2(torch.sin(a), torch.cos(a))  # noqa: E731
            state = torch.cat([
                rob.data.root_ang_vel_b, rob.data.projected_gravity_b,
                u.command_manager.get_command("base_velocity")[:, :3], rob.data.root_lin_vel_b,
                (rob.data.root_pos_w[:, 2] - ground).unsqueeze(1), wrap(roll).unsqueeze(1), wrap(pit).unsqueeze(1),
            ], dim=1)
            buf["state"].append(state.cpu().numpy().astype(np.float32))
            buf["joints"].append(torch.cat([rob.data.joint_pos - rob.data.default_joint_pos, rob.data.joint_vel],
                                           dim=1).cpu().numpy().astype(np.float32))
            buf["pose"].append(torch.cat([rob.data.root_pos_w, rob.data.root_quat_w], dim=1).cpu().numpy())
            buf["ep"].append(ep.cpu().numpy().copy())
        if k == period * 20:
            try:
                from PIL import Image
                im = buf["depth"][-1][0].astype(np.float32)
                Image.fromarray((np.clip(im / D_MAX, 0, 1) * 255).astype(np.uint8)).save(os.path.join(out, "preview.png"))
                if not args_cli.no_lidar:
                    li = buf["lidar"][-1][0].astype(np.float32)
                    meta["lidar"]["n_rays"] = int(li.shape[0])
                    Image.fromarray((np.clip(li / 6.0, 0, 1) * 255).astype(np.uint8).reshape(LIDAR_CH, -1)
                                    if li.shape[0] % LIDAR_CH == 0 else
                                    (np.clip(li / 6.0, 0, 1) * 255).astype(np.uint8)[None]).save(
                        os.path.join(out, "preview_lidar.png"))
                cp = scene["eds_cam"].data.pos_w[0]
                meta["cam"]["height_above_ground_t2s"] = float(cp[2] - ground[0])
            except Exception as exc:  # noqa: BLE001
                print("[eds_collect] Vorschau fehlgeschlagen:", exc)
        if len(buf["hm"]) >= args_cli.shard:
            flush()
    flush()
    meta["resets"] = falls
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("[eds_collect] FERTIG", json.dumps({k: v for k, v in meta.items() if k != "hm"}), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
