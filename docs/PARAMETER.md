# Parameter der Policy (MDP und PPO)

Dieses Dokument beschreibt das Lernproblem so, wie es im Code steht. Zuerst kommt die Basis, die für alle 23-DoF-Läufe gilt. Danach folgen die Änderungen je Lauf als Differenz zur Basis.

**Quellen** (Stand 24.11.2026, Pfade relativ zu `/bigwork/nhkbrc2a/isaac/`):

| Ebene | Datei | Rolle |
| --- | --- | --- |
| 1 | `vendor/unitree_rl_lab/.../g1/29dof/velocity_env_cfg.py` | Unitrees Basis-Umgebung (`RobotEnvCfg`), unverändert |
| 2 | `cpr/claas_milestone.py` | adaptive Curricula (V1/V2), DR-Erweiterungen, Treppen-Gelände, numerische Schutzmechanismen |
| 3 | `cpr/claas_g1_23dof.py` | 23-DoF-Roboter, Läufe A1–A11 und T1–T4, Robustheitstest E0–E6 |
| 3 | `cpr/claas_g1_23dof_teacher.py` | Lehrer L1/L2 mit Höhenraster |
| 3 | `cpr/claas_g1_23dof_eds.py` | Lehrer L3 im EDS-Format (CD2) |
| – | `cpr/cd1_robust.py` | Treppen-Gelände `STAIRS_TERRAINS_CFG` |
| – | `vendor/unitree_rl_lab/.../agents/rsl_rl_ppo_cfg.py` | PPO-Hyperparameter, unverändert |
| – | `claas/unitree_run.py` | Registrierung der Tasks (Task-Name → Konfigurationsklasse) |

Unter `vendor/` wurde nichts geändert. Alle Anpassungen erben von Unitrees Klassen und überschreiben einzelne Felder.

---

## 1. Roboter: Unitree G1, 23 DoF

- **Modell:** offizielle URDF `g1_23dof_rev_1_0.urdf` aus `unitree_ros`. Der Betreuer hat am 19.11. bestätigt, dass sie zum Roboter der Challenge passt. Beim ersten Start wird sie in USD umgewandelt (`claas/ext/usd_g1_23dof`).
- **Gelenke:** 6 je Bein, 1 Taille (`waist_yaw`), 5 je Arm (Schulter pitch/roll/yaw, Ellbogen, `wrist_roll`). Gegenüber 29 DoF fehlen `waist_roll/pitch` und `wrist_pitch/yaw`.
- **Startpose:** Becken 0,80 m. Hüfte pitch −0,1, Knie 0,3, Knöchel pitch −0,2, Schulter pitch 0,3, Schulter roll ±0,25, Ellbogen 0,97, Handgelenk roll ±0,15 rad.

**Aktoren** (implizite PD-Regler in PhysX, Werte wie Unitrees `UNITREE_G1_29DOF_CFG`):

| Motor | Gelenke | τ_max [Nm] | ω_max [rad/s] | Kp | Kd | Armature |
| --- | --- | --- | --- | --- | --- | --- |
| 7520-14 | Hüfte pitch/yaw, Taille yaw | 88 | 32 | 100 (Taille 200) | 2 (Taille 5) | 0,01 |
| 7520-22 | Hüfte roll, Knie | 139 | 20 | 100 / Knie 150 | 2 / Knie 4 | 0,01 |
| 5020-16 | Schultern, Ellbogen, Handgelenk, Knöchel | 25 | 37 | 40 | 1 (Knöchel 2) | 0,01 |

Die Armature 0,01 ist für den Sim-to-Sim-Transfer wesentlich. Das MuJoCo-MJCF hat Armature 0, und ohne sie fällt die Policy dort sofort (Befund 19.11.).

## 2. Zeitskalen

| Größe | Wert |
| --- | --- |
| Physik-Zeitschritt `sim.dt` | 5 ms (200 Hz) |
| Decimation | 4 → Policy-Takt 20 ms (**50 Hz**) |
| Episodenlänge | 20 s = 1000 Policy-Schritte |
| Parallele Umgebungen | 4096 |

## 3. Beobachtungen

**Policy (Actor), blind:** 6 Terme mit je 5 Zeitschritten Historie, also **390 Eingänge**.

| Term | Dim. | Skala | Rauschen (uniform ±) |
| --- | --- | --- | --- |
| Drehrate Becken `base_ang_vel` | 3 | 0,2 | 0,2 |
| projizierte Schwerkraft | 3 | 1 | 0,05 |
| Geschwindigkeitsbefehl (vx, vy, ωz) | 3 | 1 | – |
| Gelenkwinkel relativ zur Startpose | 23 | 1 | 0,01 |
| Gelenkgeschwindigkeiten | 23 | 0,05 | 1,5 |
| letzte Aktion | 23 | 1 | – |

**Critic:** dieselben Terme ohne Rauschen, zusätzlich die lineare Beckengeschwindigkeit (privilegiert), mit 5 Schritten Historie. In Unitrees Basis ist der Höhenscanner im Critic auskommentiert. Er ist nur in den Lehrer-Läufen aktiv.

**Lehrer L1/L2:** Policy und Critic bekommen zusätzlich ein exaktes Höhenraster mit 17 × 11 = 187 Punkten, Auflösung 0,1 m, Fläche 1,6 × 1,0 m. Das Raster ist um 0,5 m nach vorne versetzt (ca. −0,3 … +1,3 m vor dem Becken) und hat keine Historie. Strahlen ohne Treffer werden auf −1 gesetzt.

**Lehrer L3 (CD2, EDS-Format):** Raster mit 11 × 15 = 165 Punkten, Abstand 1/14 m, quer ±0,36 m, längs −0,3 … +0,7 m. Die Policy sieht eine „geschätzte“ Karte, die so aufbereitet ist, wie das EDS sie später liefern soll:
- nur mit 10 Hz aktualisiert
- 3×3-geglättet
- Rauschen 2,5 cm
- Drift 2 cm je Roboter

Der Critic sieht die exakte Karte.

## 4. Aktionen

- 23 Gelenk-Sollwinkel: `q_soll = q_default + 0,25 · a`. Sie gehen an die PD-Regler aus Abschnitt 1.
- **Ab A4:** Aktionslatenz zufällig 0–4 Physikschritte (0–20 ms), pro Umgebung bei jedem Reset neu gezogen (`DelayedJointPositionAction`).

## 5. Befehle

`UniformLevelVelocityCommand`: neue Befehle alle 10 s, 2 % stehende Umgebungen, kein Heading-Modus.

| Achse | Startbereich | Grenze (`limit_ranges`) |
| --- | --- | --- |
| vx [m/s] | ±0,1 (ab null) bzw. voll (ab Checkpoint) | −0,5 … 1,0 |
| vy [m/s] | ±0,1 bzw. voll | ±0,3 |
| ωz [rad/s] | ±0,1 (A1–A3); ±0,2 (A4); ±0,5 (A5 ff.) | ±0,2 (A1–A3); **±1,0 ab A4** |

## 6. Belohnungen

Basis: Unitrees `RewardsCfg`. Die Gewichte gelten pro Sekunde; Isaac Lab multipliziert sie intern mit dt.

| Term | Funktion | Basis | A2 | A5 (Kandidat) | A6 |
| --- | --- | --- | --- | --- | --- |
| `track_lin_vel_xy` | exp(−‖v_xy − v_cmd‖²/0,25) im Gier-System | 1,0 | **2,0** | 2,0 | 2,0 |
| `track_ang_vel_z` | exp(−(ω_z − ω_cmd)²/0,25) | 0,5 | 0,5 | **1,5** | 1,5 |
| `alive` | konstant | 0,15 | | | |
| `base_linear_velocity` | v_z² | −2,0 | | | |
| `base_angular_velocity` | ω_x² + ω_y² | −0,05 | | | |
| `joint_vel` | ‖q̇‖² | −0,001 | | | |
| `joint_acc` | ‖q̈‖² | −2,5·10⁻⁷ | | | |
| `action_rate` | ‖a_t − a_{t−1}‖² | −0,05 | | | |
| `dof_pos_limits` | Gelenkgrenzen | −5,0 | | | |
| `energy` | Σ\|τ·q̇\| | −2·10⁻⁵ | | | |
| `joint_deviation_arms` | L1 zur Startpose, Arme | −0,1 | | | **−0,05** |
| `joint_deviation_waists` | L1, Taille | −1,0 | | | |
| `joint_deviation_legs` | L1, Hüfte roll/yaw | −1,0 | | | |
| `flat_orientation_l2` | Neigung des Beckens | −5,0 | | | |
| `base_height` | (h − 0,78 m)² | −10 | | | |
| `gait` | Bodenkontakt im 0,8-s-Takt, Phasenversatz 0,5 | 0,5 | | | **1,5** |
| `feet_slide` | Fußgeschwindigkeit bei Kontakt | −0,2 | | | **−0,5** |
| `feet_clearance` | Fußhöhe 10 cm in der Schwungphase | 1,0 | | | **2,0** |
| `undesired_contacts` | Kontakt außer Füßen (> 1 N) | −1,0 | | | |
| `feet_air_time` | Schwungdauer > 0,4 s | – | | | **+1,0** (neu) |

Leere Felder bedeuten „wie links“. A7/A7c verwenden `gait` 1,0, `feet_clearance` 1,5 und `feet_slide` −0,3.

**Auf Treppen und unebenem Gelände** (`ms._stairs`) werden `base_height`, `feet_clearance` und der Abbruch `base_height` relativ zum Boden unter dem Roboter berechnet (Höhenscanner). Ab dem 19.11. kommt dafür `base_height_l2_safe` zum Einsatz: Es mittelt nur über gültige Strahlen und begrenzt die Strafe auf 1,0. Siehe Abschnitt 11.

## 7. Abbruchbedingungen

| Term | Bedingung |
| --- | --- |
| `time_out` | 20 s erreicht (zählt nicht als Sturz) |
| `base_height` | Becken unter 0,2 m (auf Gelände relativ zum Boden) |
| `bad_orientation` | Neigung > 0,8 rad |

**Sturzrate** = Anteil der beendeten Episoden, die durch `base_height` oder `bad_orientation` enden.

## 8. Domain Randomization

| Größe | Modus | A1–A3 | A4–A8 | A9 | A11 | Test E0–E6 |
| --- | --- | --- | --- | --- | --- | --- |
| Reibung statisch | Start, 64 Buckets | 0,3–1,25 | 0,3–1,25 | 0,15–1,25 | 0,1–0,9 | s. u. |
| Reibung dynamisch | Start | 0,3–1,2 | 0,3–1,2 | 0,12–1,2 | 0,08–0,85 | 0,9 × statisch |
| Zusatzmasse Torso | Start | −1 … +3 kg | gleich | gleich | gleich | |
| Schwerpunkt Torso | Start | ±3 cm je Achse | gleich | gleich | gleich | |
| PD-Gains Kp, Kd | Reset | ±5 … ±20 % (adaptiv) | gleich | | | ±20 % |
| Stöße (Geschwindigkeit) | alle 5–12 s | 0,2 … 1,0 m/s (adaptiv) | gleich | | | 1,0 m/s |
| Aktionslatenz | Reset | – | 0–20 ms | gleich | gleich | |
| Motorstärke (τ_max) | Reset | – | ±10 % | gleich | gleich | |
| Startposition / Gier | Reset | ±0,5 m / ±π | gleich | | | |

**Robustheitstest E0–E6** (`EVAL_SZENARIEN`): Die Policy wird nicht aktualisiert, es werden nur Kennzahlen erfasst. Alle Curricula sind aus, alle Befehle im vollen Bereich. Szenarien:

- E0: nominal = Maximalstufe des Trainings
- E1: Reibung 0,15–0,3
- E2: Stöße 1,5 m/s alle 2–4 s
- E3: Latenz 30–40 ms
- E4: +4–6 kg
- E5: Motor 70–80 %
- E6: kombiniert (Reibung 0,3–0,5, Stöße 1,2 m/s, Latenz 25–30 ms, +2–3 kg, Motor 85–90 %)

## 9. Curricula

Alle Curricula werden einmal pro Episodenlänge (1000 Schritte) ausgewertet.

| Curriculum | V1 (B1–B3) | **V2 (ab B4, alle 23-DoF-Läufe)** |
| --- | --- | --- |
| DR-Stufe (9 Stufen: Stöße 0,2→1,0 m/s, Gains ±5→20 %) | hoch, wenn mittlere Episodenlänge ≥ 75 %; runter < 45 % | hoch, wenn Sturzrate (EMA, α = 0,02) < 30 %; runter > 50 % |
| lineare Befehle (Schritt ±0,1 m/s) | wachsen, wenn Tracking-Reward > 50 % des Gewichts | > **70 %** |
| Drehbefehle (Schritt ±0,1 rad/s, ab A4) | – | > 50 % (A4), > **40 %** (A5 ff.) |
| Gelände (10 Stufen) | hoch ab 0,3 × 8 m = 2,4 m Strecke; runter unter 50 % der befohlenen Strecke | runter erst unter **25 %** |

Zum Vergleich Unitrees Original: Befehle wachsen erst ab 80 % Tracking-Reward. Deshalb blieb die Baseline bei 0,3 m/s.

## 10. Gelände

**Treppen-Mix** (`STAIRS_TERRAINS_CFG`): Kacheln 8 × 8 m, 10 Schwierigkeitsstufen × 20 Spalten. Stufenhöhe 2 → 18 cm, Stufentiefe 0,30 m, Podest 3 m.

| Typ | Anteil | Parameter |
| --- | --- | --- |
| eben | 10 % | |
| Treppe abwärts (Pyramide) | 25 % | 2–18 cm |
| Treppe aufwärts (Grube) | 25 % | 2–18 cm |
| Kästen | 10 % | 2–10 cm, Raster 0,45 m |
| zufällig rau | 15 % | 1–6 cm |
| Rampe auf / ab | je 7,5 % | 0–0,3 rad |

**Leicht uneben** (`ROUGH_LIGHT_CFG`, A8/A11):

| Typ | Anteil | Parameter |
| --- | --- | --- |
| eben | 15 % | |
| rau | 30 % | 1–6 cm |
| Rampe auf / ab | je 15 % | 0–0,3 rad |
| Kanten | 25 % | 1–5 cm |

Keine Treppen.

## 11. Numerische Schutzmechanismen

Diese drei Eingriffe ändern das Lernproblem nicht, sie verhindern nur den Abbruch des Trainings. Sie sind in allen Läufen ab dem 20.11. aktiv.

1. **`base_height_l2_safe`** (19.11.): Höhenscanner-Strahlen ohne Treffer lieferten `inf`. Der Reward wurde dadurch `−inf`, die Policy-Std `NaN`, und das Training brach ab mit `normal expects all elements of std >= 0.0` (B2, B5, A3, T1). Jetzt wird nur über gültige Strahlen gemittelt und die Strafe auf 1,0 begrenzt.
2. **Reward- und Beobachtungsschutz** (`_reward_guard`): Reward pro Schritt auf [−10, 10], Beobachtungen auf [−100, 100], `nan/inf` werden ersetzt. Anlass waren T2b/T3b: Physik-Ausreißer auf Treppen ließen den Value Loss gegen ∞ laufen.
3. **Std-Begrenzung** (`_std_absichern`): Die Policy-Std wird vor jeder Verteilungsberechnung auf [0,02; 2,0] begrenzt. Das Checkpoint-Format bleibt unverändert.

> Korrektur zur Ursachenanalyse: Am 19.11. abends wurde zunächst „zu schnelles Gelände-Curriculum + Entropie-Term“ als Ursache vermutet. Die Diagnose am selben Abend hat den `inf`-Reward als eigentliche Ursache gefunden (Punkt 1). Das Curriculum bleibt aber ein inhaltliches Problem für blinde Policies auf Treppen, siehe [EXPERIMENTE.md](EXPERIMENTE.md).

## 12. PPO (rsl_rl, `BasePPORunnerCfg`, unverändert)

| Parameter | Wert |
| --- | --- |
| Actor / Critic | MLP [512, 256, 128], ELU, getrennte Netze |
| Rauschen | Gauß, `init_noise_std` 1,0, Std als freier Parameter (`scalar`) |
| Schritte je Umgebung und Iteration | 24 → 4096 × 24 = 98 304 Samples/Iteration |
| Epochen / Minibatches | 5 / 4 |
| Lernrate | 1·10⁻³, adaptiv über KL (Ziel 0,01) |
| Clip ε | 0,2 (auch Value-Clipping) |
| γ / λ (GAE) | 0,99 / 0,95 |
| Entropie-Koeffizient | 0,01 |
| Value-Loss-Koeffizient | 1,0 |
| max. Gradientennorm | 1,0 |
| Beobachtungs-Normierung | aus |
| Seed | Standard der Runner-Konfiguration; A5s1/A5s2 mit Seed 1/2 |
| Checkpoint | alle 100 Iterationen |
| Symmetrie (nur A7/A7b) | Daten-Augmentierung links/rechts (`symmetry_cfg`), kein Mirror-Loss |

**Weitertrainieren:** Mit `--resume` wird ab einem Checkpoint fortgesetzt, inklusive Optimizer-Zustand. Ausnahme sind die Lehrer: Das Netz bekommt zusätzliche Eingänge, die alten Gewichte werden übernommen und die neuen Eingänge mit 0 initialisiert. Am Anfang verhält sich der Lehrer deshalb exakt wie das Startmodell. Der Optimizer startet dabei neu.

## 13. Deploy

`params/deploy.yaml` im Run-Ordner enthält:
- die Zuordnung der Isaac-Gelenke auf die 29 Motor-Plätze des Unitree-SDK (die beim 23-DoF-Modell fehlenden Plätze sind leer, Kp = Kd = 0)
- Kp/Kd
- die Startpose
- Action-Scale 0,25
- den Beobachtungsaufbau mit Historie 5
- `step_dt` 0,02

Die exportierten TorchScript-Policies haben 390 Eingänge und 23 Ausgänge (geprüft 22.11., `claas/deploy/PRUEFUNG.txt`).
