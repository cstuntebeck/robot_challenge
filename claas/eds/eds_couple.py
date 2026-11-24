"""CD2 / ToDo 5: EDS + Lehrer-Policy L3 in der Simulation koppeln - Claas, 24.11.

L3 laeuft im Treppen-Gelaende (eigene Spur je Roboter, Stufen werden hoeher). Die Hoehenkarte
fuer die Policy kommt je nach --mode aus
  exact  der exakten Karte (Obergrenze)
  est    der kuenstlich verrauschten Karte, mit der L3 trainiert wurde
  eds    dem trainierten Encoder-Decoder (Kamera/LiDAR/Zustand -> Karte, 10 Hz)
Gemessen: Stuerze (Abbruch ohne Zeitablauf), zurueckgelegte Strecke je Episode, Fehler der
Karte gegenueber der exakten (cm, nur eds/est). Ergebnis: <out>.json, optional Video.
"""

import argparse
import copy
import json
import math
import os
import re

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--mode", choices=["exact", "est", "eds"], required=True)
parser.add_argument("--decoder", default=None, help="decoder_<modal>[_real].pt (eds)")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seconds", type=float, default=60.0)
parser.add_argument("--out", required=True, help="Pfad ohne Endung")
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=2000)
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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
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

# Netzklassen aus eds_train.py uebernehmen (ohne dessen Argumente zu parsen)
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eds_train.py")).read()
_cls = _src[_src.index("class ConvAE"):_src.index("def split_envs")]
exec(_cls, globals())  # noqa: S102  definiert ConvAE, Decoder

# wie eds_collect.py
HFOV, D_MIN, D_MAX = 87.0, 0.2, 3.0
CAM_LINK, CAM_POS = "torso_link", (0.20, 0.0, 0.156)
LIDAR_POS, LIDAR_CH, LIDAR_COLS, LIDAR_VFOV = (0.0, 0.0, 0.42), 40, 276, (-52.0, 7.0)
W, H, DW, DH = 212, 120, 160, 120

ST = {"model": None}


def _state(env):
    rob = env.scene["robot"]
    sc = env.scene["height_scanner_eds"]
    ground = torch.nan_to_num(sc.data.ray_hits_w[..., 2], nan=0.0).mean(dim=1)
    roll, pit, _ = euler_xyz_from_quat(rob.data.root_quat_w)
    wrap = lambda a: torch.atan2(torch.sin(a), torch.cos(a))  # noqa: E731
    return torch.cat([rob.data.root_ang_vel_b, rob.data.projected_gravity_b,
                      env.command_manager.get_command("base_velocity")[:, :3], rob.data.root_lin_vel_b,
                      (rob.data.root_pos_w[:, 2] - ground).unsqueeze(1), wrap(roll).unsqueeze(1),
                      wrap(pit).unsqueeze(1)], dim=1)


def hm_eds(env, sensor_cfg=eds.EDS, offset: float = 0.5):
    exact = eds.height_scan_eds_exact(env, sensor_cfg, offset)
    if ST["model"] is None:
        return exact
    if "buf" not in ST or ST["buf"].shape != exact.shape:
        ST["buf"] = exact.clone()
        ST["prev"] = torch.zeros_like(exact)
        ST["h"] = None
        ST["err"] = []
    reset = env.episode_length_buf <= 1
    if reset.any():
        ST["prev"][reset] = 0.0
        if ST["h"] is not None:
            ST["h"][:, reset] = 0.0
    if env.common_step_counter % 5 == 0 or ST["h"] is None:
        with torch.no_grad():
            zd = zl = None
            if "depth" in ST["enc"]:
                d = env.scene["eds_cam"].data.output["distance_to_image_plane"][..., 0]
                d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                d = torch.where((d >= D_MIN) & (d <= D_MAX), d, torch.zeros_like(d))
                x0 = (d.shape[-1] - DW) // 2
                d = d[:, :DH, x0:x0 + DW]
                m, sc = ST["enc"]["depth"]
                zd = m.encode((d / sc).clamp(0, 1).unsqueeze(1))
            if "lidar" in ST["enc"]:
                ld = env.scene["eds_lidar"].data
                r = torch.linalg.norm(ld.ray_hits_w - ld.pos_w.unsqueeze(1), dim=-1)
                r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
                r = torch.where(r <= 20.0, r, torch.zeros_like(r)).view(r.shape[0], LIDAR_CH, -1)
                m, sc = ST["enc"]["lidar"]
                zl = m.encode((r / sc).clamp(0, 1).unsqueeze(1))
            s = (_state(env)[:, ST["cols"]] - ST["mu"]) / ST["sd"]
            y, ST["h"] = ST["model"].step(zd, zl, s, ST["prev"], ST["h"])
            ST["prev"] = y
            ST["buf"] = y
            ST["err"].append(float((y - exact).abs().mean()))
    return ST["buf"]


ERR_EST = []


def hm_est_logged(env, sensor_cfg=eds.EDS, offset: float = 0.5):
    y = eds.height_scan_eds_est(env, sensor_cfg, offset)
    if env.common_step_counter % 5 == 0:
        ERR_EST.append(float((y - eds.height_scan_eds_exact(env, sensor_cfg, offset)).abs().mean()))
    return y


def pitch_quat(p):
    h = math.radians(p) / 2
    return (math.cos(h), 0.0, math.sin(h), 0.0)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=True, entry_point_key="play_env_cfg_entry_point")
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    pol = env_cfg.observations.policy
    pol.height_scan.func = {"exact": eds.height_scan_eds_exact, "est": hm_est_logged, "eds": hm_eds}[args_cli.mode]
    if args_cli.mode == "eds":
        vf = math.degrees(2 * math.atan(math.tan(math.radians(HFOV / 2)) * H / W))
        env_cfg.scene.eds_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + CAM_LINK + "/eds_cam", update_period=0.0, width=W, height=H,
            data_types=["distance_to_image_plane"], update_latest_camera_pose=True,
            spawn=sim_utils.PinholeCameraCfg(focal_length=10.0, horizontal_aperture=2 * 10.0 * math.tan(math.radians(HFOV / 2)),
                                             clipping_range=(0.05, 20.0)),
            offset=TiledCameraCfg.OffsetCfg(pos=CAM_POS, rot=pitch_quat(90.0 - vf / 2), convention="world"))
        lid = copy.deepcopy(env_cfg.scene.height_scanner)
        lid.prim_path = "{ENV_REGEX_NS}/Robot/" + CAM_LINK
        lid.offset = RayCasterCfg.OffsetCfg(pos=LIDAR_POS)
        lid.pattern_cfg = patterns.LidarPatternCfg(channels=LIDAR_CH, vertical_fov_range=LIDAR_VFOV,
                                                   horizontal_fov_range=(-180.0, 180.0), horizontal_res=360.0 / LIDAR_COLS)
        lid.max_distance = 20.0
        if getattr(lid, "attach_yaw_only", None):
            lid.attach_yaw_only = False
        if hasattr(lid, "ray_alignment"):
            lid.ray_alignment = "base"
        lid.debug_vis = False
        env_cfg.scene.eds_lidar = lid

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(env, video_folder=os.path.dirname(args_cli.out) + "/video_" + os.path.basename(args_cli.out),
                                       step_trigger=lambda s: s == 0, video_length=args_cli.video_length, disable_logger=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    u = env.unwrapped
    dev = u.device

    info = {"task": args_cli.task, "mode": args_cli.mode, "checkpoint": args_cli.checkpoint, "decoder": args_cli.decoder}
    if args_cli.mode == "eds":
        ck = torch.load(args_cli.decoder, map_location=dev, weights_only=False)
        modal = ck["modal"]
        ST["enc"] = {}
        for k in (["depth"] if modal in ("depth", "fusion") else []) + (["lidar"] if modal in ("lidar", "fusion") else []):
            a = torch.load(os.path.join(os.path.dirname(args_cli.decoder), f"ae_{k}.pt"), map_location=dev, weights_only=False)
            m = ConvAE(a["h"], a["w"], a["latent"]).to(dev)
            m.load_state_dict(a["state"])
            m.eval()
            ST["enc"][k] = (m, a["scale"])
        cols = ck.get("cols", list(range(15)))
        dm = Decoder("depth" in ST["enc"], "lidar" in ST["enc"], ck["latent"], n_state=len(cols)).to(dev)
        dm.load_state_dict(ck["state"])
        dm.eval()
        ST["cols"] = torch.tensor(cols, device=dev)
        ST["mu"] = torch.tensor(ck["s_mu"], device=dev, dtype=torch.float32)
        ST["sd"] = torch.tensor(ck["s_sd"], device=dev, dtype=torch.float32)
        ST["model"] = dm
        info["modal"] = modal
        info["state_cols"] = cols

    rob = u.scene["robot"]
    n = u.num_envs
    start_x = rob.data.root_pos_w[:, 0].clone()
    dists, falls, timeouts = [], 0, 0
    obs = env.get_observations()
    steps = int(args_cli.seconds / u.step_dt)
    for k in range(steps):
        x_before = rob.data.root_pos_w[:, 0].clone()
        with torch.inference_mode():
            obs, _, dones, extras = env.step(policy(obs))
        d = dones.bool()
        if d.any():
            to = extras.get("time_outs", torch.zeros_like(d)).bool()
            falls += int((d & ~to).sum())
            timeouts += int((d & to).sum())
            dists += (x_before[d] - start_x[d]).tolist()
            start_x[d] = rob.data.root_pos_w[d, 0]
    dists_alive = (rob.data.root_pos_w[:, 0] - start_x).tolist()
    err = ST.get("err", []) if args_cli.mode == "eds" else ERR_EST
    info.update({
        "num_envs": n, "seconds": args_cli.seconds, "stuerze": falls, "zeitabläufe": timeouts,
        "episoden_beendet": len(dists),
        "strecke_m_mittel": round(float(np.mean(dists + dists_alive)), 2),
        "strecke_m_lebend_mittel": round(float(np.mean(dists_alive)), 2),
        "karte_fehler_cm": round(100 * float(np.mean(err[5:])), 2) if len(err) > 5 else None,
    })
    json.dump(info, open(args_cli.out + ".json", "w"), indent=1, ensure_ascii=False)
    print("[eds_couple] ERGEBNIS", json.dumps(info, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
