# Deploy-Pakete G1 23 DoF (Claas, 22.11.2026)

Vorbereitung für den ersten Test am echten Roboter (gemeinsam mit Dennis, Werkstatt).
Noch **nicht** auf dem Roboter oder in `unitree_mujoco` getestet – nur in Isaac Lab und in MuJoCo mit eigenem Skript (`claas/sim2sim/`).

## Pakete

| Ordner | Policy | Empfehlung |
| --- | --- | --- |
| `A5_robust_drehen/` | eben, Latenz 0–20 ms, Motorstärke ±10 %, Drehen ±1 rad/s | **zuerst testen** |
| `A6_gangbild/` | wie A5 + Gangbild-Rewards | nur, wenn Gang im Video besser |
| `A4_robust/` | eben, robust, dreht kaum | Rückfall |
| `A2_eben/` | eben, ohne Latenz/Motorstreuung | Basis |

Jeder Ordner: `policy.onnx` / `policy.pt` (Actor, exportiert von Isaac Lab), `deploy.yaml` (alles für den Controller), `env.yaml`/`agent.yaml` (Trainingskonfiguration), Checkpoint, `INFO.txt` (Herkunft + Prüfsummen).

## Was der Controller tun muss (aus `deploy.yaml`)

- **Takt:** Policy 50 Hz (`step_dt: 0.02`), Sollwerte an die Motoren über unitree_sdk2 (LowCmd) mit Kp/Kd aus `stiffness`/`damping`.
- **Gelenkreihenfolge:** Die Policy rechnet in Isaac-Reihenfolge; `joint_ids_map[i]` = Motorindex im SDK (29 Plätze, beim 23-DoF-G1 bleiben 13/14 = waist roll/pitch, 20/21 und 27/28 = wrist pitch/yaw leer). `stiffness`/`damping` stehen in SDK-Reihenfolge, `default_joint_pos` in Isaac-Reihenfolge.
- **Beobachtung (390 Werte):** je 5 Zeitschritte, älteste zuerst, Terme in dieser Reihenfolge:
  1. `base_ang_vel` (IMU-Gyro, Körperframe) × 0,2 → 15
  2. `projected_gravity` (Schwerkraft im Körperframe aus IMU-Quaternion) → 15
  3. `velocity_commands` (vx, vy, wz) → 15
  4. `joint_pos_rel` (q − default_joint_pos) → 115
  5. `joint_vel_rel` (dq) × 0,05 → 115
  6. `last_action` → 115
  Beim Start Historie mit dem ersten Wert füllen.
- **Aktion:** `q_soll = default_joint_pos + 0,25 × Aktion`, kein Clipping.
- **Befehlsbereich im Training:** vx −0,5…1,0 m/s, vy ±0,3 m/s, wz ±1,0 rad/s (A5/A6).

## Stolpersteine (aus Sim-to-Sim gelernt)

- Ohne Motorträgheit (Armature 0,01) fiel der Roboter im MuJoCo-Modell von unitree_ros sofort um. In `unitree_mujoco` prüfen, ob Armature gesetzt ist.
- IMU: Policy erwartet Werte im **Becken**-Frame (Root). Der G1 hat IMUs im Becken und im Torso – die Becken-IMU nehmen.
- Vorzeichen/Achsen der Gelenke zwischen SDK und Modell beim ersten Start mit kleiner Bewegung prüfen.

## Ablauf in der Werkstatt (Vorschlag)

1. `unitree_mujoco` auf dem ThinkPad (Linux), Controller gegen die Simulation über DDS → gleiches Verhalten wie im eigenen MuJoCo-Test?
2. Roboter am Gestell, Dämpfungsmodus → langsam (2–3 s) in `default_joint_pos` überblenden → Policy mit Befehl 0 (stehen).
3. Not-Aus bereit, Drehmoment- und Winkelgrenzen aktiv.
4. Langsam vorwärts (0,2 m/s), dann drehen, dann vom Gestell.
5. Abbruch bei Schwingen, Zittern oder Vorzeichenfehlern.
