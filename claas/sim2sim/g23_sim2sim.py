"""Sim-to-Sim: in Isaac Lab trainierte 23-DoF-Policy in MuJoCo testen (Claas, 19.11.).

Laeuft mit der Container-Python (torch) + mujoco aus claas/ext/pylib:
    bash claas/sim2sim/run_sim2sim.sh <policy.pt> <deploy.yaml> [tag]

Alles, was die Policy sieht und tut, kommt aus der deploy.yaml des Trainingslaufs
(params/deploy.yaml) - genau die Datei, die auch fuer den echten Roboter gebraucht wird:
  joint_ids_map  Isaac-Gelenk i -> Index in der Unitree-SDK-Reihenfolge
  stiffness/damping (SDK-Reihenfolge), default_joint_pos/offset, action scale,
  Beobachtungen inkl. scale und history_length, step_dt.
MuJoCo-Modell: offizielle g1_23dof_rev_1_0.xml aus unitree_ros (Coach: URDF/Modell passt).

Ausgabe: runs/sim2sim/<tag>.json (Kennzahlen je Abschnitt) und, wenn Rendern geht,
runs/videos/sim2sim_<tag>.mp4.
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch
import yaml
import mujoco

BASE = "/bigwork/nhkbrc2a/isaac"
XML = f"{BASE}/claas/ext/unitree_ros/robots/g1_description/g1_23dof_rev_1_0.xml"

# Unitree-SDK-Reihenfolge (29 Plaetze; beim 23-DoF-Roboter bleiben 13/14, 20/21, 27/28 leer)
SDK_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "", "",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "", "",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "", "",
]


def effort_limit(name):
    if any(k in name for k in ("hip_pitch", "hip_yaw", "waist_yaw")):
        return 88.0
    if any(k in name for k in ("hip_roll", "knee")):
        return 139.0
    return 25.0


# Testablauf: (Dauer s, vx, vy, wz, Name) - innerhalb der Trainingsbereiche aus deploy.yaml
SCENARIO = [
    (2.0, 0.0, 0.0, 0.0, "stehen"),
    (5.0, 0.5, 0.0, 0.0, "vor 0,5"),
    (4.0, 1.0, 0.0, 0.0, "vor 1,0"),
    (3.0, 0.0, 0.3, 0.0, "seitlich 0,3"),
    (3.0, -0.4, 0.0, 0.0, "rueckwaerts 0,4"),
    (4.0, 0.5, 0.0, 0.2, "Kurve"),
    (4.0, 0.5, 0.0, 0.0, "Stoesse"),  # 2 seitliche Stoesse (0,5 m/s) in diesem Abschnitt
    (3.0, 0.0, 0.0, 0.8, "drehen 0,8"),
    (2.0, 0.0, 0.0, 0.0, "stehen"),
]
# Treppen-Test (23.11.): S2S_STAIRS=<Stufenhoehe m> baut 6 Stufen hoch, Podest, 5 Stufen runter
STAIRS_H = float(os.environ.get("S2S_STAIRS", "0") or 0)
if STAIRS_H > 0:
    SCENARIO = [
        (2.0, 0.0, 0.0, 0.0, "stehen"),
        (18.0, 0.5, 0.0, 0.0, "Treppe 0,5"),
        (2.0, 0.0, 0.0, 0.0, "stehen"),
    ]
PUSH_TIMES = (22.0, 24.0)  # Sekunden ab Start, beide im Abschnitt "Stoesse" (21-25 s)
PUSH_VEL = float(os.environ.get("S2S_PUSH", "0.5"))


def quat_rotate_inverse(q_wxyz, v):
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return R.T @ v


def main():
    policy_path, deploy_path = sys.argv[1], sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else "test"
    cfg = yaml.safe_load(open(deploy_path))

    ids_map = cfg["joint_ids_map"]
    names = [SDK_NAMES[i] for i in ids_map]  # Isaac-Reihenfolge
    n = len(names)
    kp = np.array([cfg["stiffness"][i] for i in ids_map])
    kd = np.array([cfg["damping"][i] for i in ids_map])
    q0 = np.array(cfg["default_joint_pos"])
    act_cfg = cfg["actions"]["JointPositionAction"]
    a_scale = np.array(act_cfg["scale"])
    a_offset = np.array(act_cfg["offset"])
    tau_max = np.array([effort_limit(nm) for nm in names])
    step_dt = cfg["step_dt"]

    obs_cfg = cfg["observations"]
    order = list(obs_cfg.keys())
    hist = {k: obs_cfg[k].get("history_length", 1) or 1 for k in order}
    scale = {k: np.array(obs_cfg[k]["scale"]) for k in order}

    if STAIRS_H > 0:
        spec = mujoco.MjSpec.from_file(XML)
        n_st, tread, x0, plat = 6, 0.30, 1.5, 1.5

        def box(xc, top, lx):
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[lx / 2, 1.5, top / 2],
                                    pos=[xc, 0.0, top / 2], rgba=[0.55, 0.6, 0.68, 1.0])

        for i in range(n_st):
            box(x0 + (i + 0.5) * tread, (i + 1) * STAIRS_H, tread)
        top = n_st * STAIRS_H
        xp = x0 + n_st * tread
        box(xp + plat / 2, top, plat)
        xd = xp + plat
        for i in range(n_st - 1):
            box(xd + (i + 0.5) * tread, top - (i + 1) * STAIRS_H, tread)
        model = spec.compile()
        print(f"Treppe: {n_st} Stufen je {STAIRS_H * 100:.0f} cm hoch, Auftritt {tread} m, Podest {plat} m")
    else:
        model = mujoco.MjModel.from_xml_path(XML)
    model.opt.timestep = float(os.environ.get("S2S_DT", "0.002"))
    nopolicy = os.environ.get("S2S_NOPOLICY", "0") == "1"
    decim = int(round(step_dt / model.opt.timestep))
    data = mujoco.MjData(model)

    qadr = np.array([model.jnt_qposadr[model.joint(nm).id] for nm in names])
    vadr = np.array([model.jnt_dofadr[model.joint(nm).id] for nm in names])
    act_id = np.array([model.actuator(nm).id for nm in names])
    print("MJCF armature", np.round(model.dof_armature[vadr], 4).tolist())
    print("MJCF damping", np.round(model.dof_damping[vadr], 3).tolist(), "frictionloss", np.round(model.dof_frictionloss[vadr], 3).tolist())
    print("ctrlrange", model.actuator_ctrllimited[act_id].tolist(), np.round(model.actuator_ctrlrange[act_id], 1).tolist()[:3], "gear", model.actuator_gear[act_id, 0].tolist()[:3])
    # Isaac-Konfiguration nutzt armature=0.01 an allen Gelenken; das MJCF von unitree_ros hat 0.
    # Ohne Armature faellt der Roboter in MuJoCo sofort um (Diagnose 9391173, 19.11.).
    arm = os.environ.get("S2S_ARMATURE", "0.01")
    if arm:
        model.dof_armature[vadr] = float(arm)
        print("Armature gesetzt auf", arm)
    print("Isaac-Reihenfolge", names)
    # Stoerungen fuer den Robustheitstest (22.11.)
    fr = os.environ.get("S2S_FRICTION")
    if fr:
        model.geom_friction[:, 0] = float(fr)
        print("Reibung gesetzt auf", fr)
    ma = os.environ.get("S2S_MASS")
    if ma:
        model.body_mass[model.body("torso_link").id] += float(ma)
        print("Zusatzmasse Torso", ma, "kg")
    delay = int(os.environ.get("S2S_DELAY", "0"))
    if delay:
        print("Aktionslatenz", delay, "Physikschritte =", delay * model.opt.timestep * 1000, "ms")

    # Startpose
    data.qpos[2] = 0.80
    data.qpos[qadr] = q0
    mujoco.mj_forward(model, data)

    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()

    buffers = {k: None for k in order}
    last_action = np.zeros(n)

    def term(k, cmd):
        quat = data.qpos[3:7]
        if k == "base_ang_vel":
            return data.qvel[3:6].copy()
        if k == "projected_gravity":
            return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        if k == "velocity_commands":
            return np.array(cmd)
        if k == "joint_pos_rel":
            return data.qpos[qadr] - q0
        if k == "joint_vel_rel":
            return data.qvel[vadr].copy()
        if k == "last_action":
            return last_action.copy()
        raise KeyError(f"Beobachtung {k} nicht implementiert")

    def build_obs(cmd):
        parts = []
        for k in order:
            v = term(k, cmd) * scale[k]
            if buffers[k] is None:  # Isaac Lab fuellt die Historie beim Reset mit dem ersten Wert
                buffers[k] = [v] * hist[k]
            else:
                buffers[k] = buffers[k][1:] + [v]
            parts.append(np.concatenate(buffers[k]))  # aelteste zuerst
        return np.concatenate(parts).astype(np.float32)

    # Rendern (optional)
    renderer = None
    frames = []
    try:
        W, H = (1280, 720) if STAIRS_H > 0 else (640, 480)
        model.vis.global_.offwidth, model.vis.global_.offheight = max(W, 640), max(H, 480)
        renderer = mujoco.Renderer(model, H, W)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = model.body("pelvis").id
        cam.distance, cam.elevation, cam.azimuth = 3.0, -15.0, 135.0
        if STAIRS_H > 0:
            cam.distance, cam.elevation, cam.azimuth = 3.2, -12.0, 90.0
    except Exception as e:  # noqa: BLE001
        print("Rendern nicht moeglich:", repr(e))
        renderer = None

    targets = [q0.copy()] * 4
    results = []
    t = 0.0
    fell_at = None
    in_dim_checked = False
    pushes = [] if STAIRS_H > 0 else list(PUSH_TIMES)
    t_start = time.time()
    for dur, vx, vy, wz, label in SCENARIO:
        cmd = (vx, vy, wz)
        v_meas, wz_meas, heights = [], [], []
        steps = int(round(dur / step_dt))
        for _ in range(steps):
            obs = build_obs(cmd)
            if nopolicy:
                a = np.zeros(n)
            else:
                with torch.no_grad():
                    a = policy(torch.from_numpy(obs)[None])[0].numpy()
            k_dbg = int(round(t / step_dt))
            if k_dbg < 3 or k_dbg in (25, 50):
                print(f"t={t:.2f} z={data.qpos[2]:.3f} angv={np.round(obs[12:15], 3).tolist()} grav={np.round(obs[27:30], 3).tolist()} a_max={np.abs(a).max():.2f} a[:6]={np.round(a[:6], 2).tolist()}")
            if not in_dim_checked:
                print(f"Beobachtung {obs.shape[0]} Werte, Aktion {a.shape[0]} Werte")
                in_dim_checked = True
            last_action = a.copy()
            q_new = a_offset + a_scale * a
            targets = targets[1:] + [q_new]
            ds, dsub = divmod(delay, decim)
            for _sub in range(decim):
                q_tgt = targets[-1 - ds] if _sub >= dsub else targets[-2 - ds]
                tau = kp * (q_tgt - data.qpos[qadr]) - kd * data.qvel[vadr]
                data.ctrl[act_id] = np.clip(tau, -tau_max, tau_max)
                mujoco.mj_step(model, data)
            t += step_dt
            if pushes and t >= pushes[0]:
                pushes.pop(0)
                quat = data.qpos[3:7]
                yaw = math.atan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
                data.qvel[0] += -math.sin(yaw) * PUSH_VEL  # seitlich zur Laufrichtung
                data.qvel[1] += math.cos(yaw) * PUSH_VEL
            lin_b = quat_rotate_inverse(data.qpos[3:7], data.qvel[0:3])
            v_meas.append(lin_b[:2].copy())
            wz_meas.append(data.qvel[5])
            heights.append(data.qpos[2])
            grav_z = quat_rotate_inverse(data.qpos[3:7], np.array([0.0, 0.0, -1.0]))[2]
            if fell_at is None and (data.qpos[2] < 0.45 or grav_z > -0.6):
                fell_at = round(t, 2)
            if renderer is not None and len(frames) < 60 * 30 and int(round(t / step_dt)) % 2 == 0:
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())
        vm = np.array(v_meas)
        half = len(vm) // 2  # Einschwingen ignorieren
        results.append({
            "abschnitt": label,
            "befehl": [vx, vy, wz],
            "gemessen_vx": round(float(vm[half:, 0].mean()), 3),
            "gemessen_vy": round(float(vm[half:, 1].mean()), 3),
            "gemessen_wz": round(float(np.mean(wz_meas[half:])), 3),
            "hoehe_min": round(float(np.min(heights)), 3),
        })
        print(results[-1])

    out = {
        "tag": tag,
        "policy": policy_path,
        "deploy": deploy_path,
        "mujoco": mujoco.__version__,
        "timestep": model.opt.timestep,
        "decimation": decim,
        "gestuerzt_bei_s": fell_at,
        "dauer_s": round(t, 2),
        "abschnitte": results,
        "treppe_m": STAIRS_H,
        "x_ende_m": round(float(data.qpos[0]), 2),
        "rechenzeit_s": round(time.time() - t_start, 1),
    }
    os.makedirs(f"{BASE}/runs/sim2sim", exist_ok=True)
    json.dump(out, open(f"{BASE}/runs/sim2sim/{tag}.json", "w"), indent=2, ensure_ascii=False)
    print("ERGEBNIS", json.dumps({"gestuerzt_bei_s": fell_at, "dauer_s": round(t, 2)}))

    if frames:
        try:
            import imageio.v2 as imageio

            path = f"{BASE}/runs/videos/sim2sim_{tag}.mp4"
            imageio.mimsave(path, frames, fps=25)
            print("VIDEO_OK", path)
        except Exception as e:  # noqa: BLE001
            print("Video konnte nicht geschrieben werden:", repr(e))


if __name__ == "__main__":
    main()
