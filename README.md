# RobotChallenge – Gruppe A: Fortbewegung für den Unitree G1 (23 DoF)

Diese Arbeit entsteht im Rahmen der RobotChallenge 2026/27. Ziel ist eine Fortbewegungs-Policy für den humanoiden Roboter **Unitree G1 mit 23 Freiheitsgraden**:

- gelernt mit Reinforcement Learning (PPO) in **Isaac Lab**
- geprüft per Sim-to-Sim in **MuJoCo**
- vorbereitet für den Einsatz am realen Roboter

| Challenge Day | Termin | Ziel | Stand |
| --- | --- | --- | --- |
| CD1 – Robuste Simulation | 01.12.2026 | robuste Fortbewegung, Videos, Curriculum- und Reward-Tuning dokumentiert | Kandidaten A8 und A5 stehen fest, Deploy-Pakete gebaut |
| CD2 – Terrain-Aware & Blind Policy | 12.01.2027 | Policy mit Höhenkarte, Wahrnehmung (EDS) | Lehrer L1–L3 trainiert, EDS in Arbeit |
| CD3 – Vision Policy auf G1 | 26.01.2027 | Vision-Policy auf dem realen Roboter | – |

## Ergebnis auf einen Blick (Stand 24.11.2026)

- **A8** (empfohlen für den Realtest): blinde Policy, trainiert auf leicht unebenem Boden mit Latenz, Motorstärke-Streuung und Drehbefehlen.
  - Isaac: 0,05 % Stürze nominal, 2,7 % bei Stößen von 1,5 m/s, 10 % bei kombinierten Störungen
  - MuJoCo: kein Sturz; folgt 1,0 m/s mit 1,04 m/s, dreht 0,73 rad/s bei 0,8 befohlen, überquert 5-cm-Stufen
- **A5** (Vergleich/Rückfall): dasselbe Rezept auf ebenem Boden. Mit drei Seeds reproduziert.
- **Grenzen:** Blind sind Treppen über ~5–8 cm nicht zuverlässig lösbar, und Reibung unter 0,3 ist kritisch. Ein Lehrer mit Höhenraster (L1) erreicht auf derselben Gelände-Stufe 0,4 % statt 13–15 % Stürze. Das ist der Weg für CD2.

Details: [docs/EXPERIMENTE.md](docs/EXPERIMENTE.md) · alle Parameter: [docs/PARAMETER.md](docs/PARAMETER.md)

## Wie die Policy aufgebaut ist

```
Befehl (vx, vy, ωz) ─┐
IMU (Drehrate, Schwerkraft) ─┤   5 Schritte Historie    MLP 512-256-128   23 Gelenk-Sollwinkel
Gelenkwinkel, -geschw. ─┤ ─────────────────────► (ELU) ─────────────► q_soll = q0 + 0,25·a
letzte Aktion ─┘          390 Eingänge                                  → PD-Regler im Motor
                                                                         Policy 50 Hz, Physik 200 Hz
```

- **Lernverfahren:** PPO (rsl_rl) mit asymmetrischem Actor-Critic. Der Critic sieht zusätzlich die Beckengeschwindigkeit. 4096 parallele Roboter.
- **Belohnung:** Hauptterme sind das Folgen der Soll-Geschwindigkeit (Gewicht 2,0) und der Soll-Drehrate (1,5). Dazu kommen Strafterme für Energie, Ruck, Haltung, Fußschleifen und unerwünschte Kontakte (Unitrees Reward-Satz).
- **Robustheit durch Domain Randomization:**
  - Reibung, Masse und Schwerpunkt
  - PD-Gains, Stöße
  - Aktionslatenz 0–20 ms, Motorstärke ±10 %
- **Adaptive Curricula:** Befehlsbereich, Störungsstärke und Gelände wachsen erst, wenn die Policy die aktuelle Stufe beherrscht. Die Störungsstärke wird über die Sturzrate gesteuert, die Befehle über die Tracking-Güte.

## Wie die Policy entstanden ist

Jeder Schritt hat eine Hypothese und eine Änderung und wurde an derselben Metrik gemessen. Das Laborbuch steht in [docs/EXPERIMENTE.md](docs/EXPERIMENTE.md).

1. **Baseline** (Unitrees Task, 29 DoF): läuft eben, aber 28 % Stürze. Das Befehls-Curriculum bleibt bei 0,3 m/s hängen.
2. **Adaptive Curricula V2** (Steuerung über die Sturzrate) und **Tracking-Gewicht 2,0** gegen das Treten auf der Stelle.
3. **Umstellung auf 23 DoF** (A2): eben 0,2 % Stürze bei voller DR. Das ist die Basis aller weiteren Läufe.
4. **Numerische Stabilität:** `inf`-Werte des Höhenscanners verursachten alle Trainingsabbrüche auf Treppen. Behoben am 19./20.11.
5. **Sim-to-Sim zeigt Lücken:** A2 dreht in MuJoCo nicht. A4 ergänzt Latenz, Motorstärke und Drehbefehle, A5 korrigiert das Gewichtsverhältnis (Drehen 1,5).
6. **Robustheitstest E0–E6** ohne Policy-Updates: A5 ist robust bis auf glatten Boden. A8 (unebener Boden) ist in allen Szenarien besser.
7. **Ablationen:** Links/Rechts-Symmetrie verhindert das Drehen (A7b), Gangbild-Rewards kosten Tempo und Drehen (A6/A7c), blinde Treppen scheitern (T4, A10).
8. **Wahrnehmung:** Lehrer mit exaktem Höhenraster (L1/L2), dann im Format des EDS (L3). Das ist die Vorlage für die Schüler-Policy in CD2.

## Repository-Struktur

Code und Konfiguration liegen hier. Große Daten (Checkpoints, Videos, Logs, Container) bleiben auf dem Cluster unter `/bigwork/nhkbrc2a/isaac/`, siehe `.gitignore`.

```
cpr/                        Umgebungs-Konfigurationen (erben von Unitree, ändern nur Felder)
  claas_g1_23dof.py           23-DoF-Roboter, Läufe A1–A11, T1–T4, Robustheitstest E0–E6
  claas_milestone.py          adaptive Curricula V1/V2, DR, Treppen-Gelände, Schutzmechanismen
  claas_g1_23dof_teacher.py   Lehrer L1/L2 mit Höhenraster
  claas_g1_23dof_eds.py       Lehrer L3 im EDS-Format (CD2)
  cd1_robust.py               Treppen-Gelände (STAIRS_TERRAINS_CFG)
  unitree_g1_perceptive.py    Höhenraster-Bausteine (Team)
claas/                      Ausführung
  unitree_run.py              Registrierung: Task-Name → Konfigurationsklasse
  ms_train.sh / ms_play.sh    Training bzw. Videos mit Start-Checkpoint (Slurm)
  g23_kette*.sh               automatische Ketten: Smoke → Training → Video → MuJoCo
  sim2sim/                    MuJoCo-Transfer und Checkpoint-Vergleich
  deploy/                     Export (TorchScript/ONNX) und Prüfung der Deploy-Pakete
  eds/                        Höhenkarten-Schätzung (CD2)
scripts/                    Container-Wrapper, Basis-Submit-Skripte
docs/                       EXPERIMENTE.md (Laborbuch), PARAMETER.md (MDP + PPO)
```

Nicht im Repository:
- `vendor/`: `unitree_rl_lab` (Commit 4960b84) und Isaac Lab, unverändert
- `runs/`, `logs/`: Checkpoints, Videos, Slurm-Logs

## Reproduzieren

Umgebung: Isaac Sim 5.1, Isaac Lab 2.3.2 im Apptainer-Container auf dem LUIS-Cluster, GPU H200 (Partition `gpu` bzw. Reservierung `imes.gpu`).

```bash
cd /bigwork/nhkbrc2a/isaac

G="-p gpu --exclude=gpu-20-n[005-008]"   # V100-Knoten laufen mit dem Container nicht

# Training (Beispiel A8: ab A5-Checkpoint, 4000 Iterationen)
sbatch $G --time=08:00:00 \
  --export=ALL,TASK=Unitree-G1-Claas-23-Rough,INIT_CKPT=<A5-Checkpoint>,MAX_ITER=4000,VIDEO=0 \
  claas/ms_train.sh
# Seed setzen (Reproduzierbarkeit): zusätzlich EXTRA_ARGS="--seed 2"

# Robustheitstest eines Checkpoints (Szenario E0…E6; 60 Iterationen ohne Policy-Update)
sbatch $G --time=00:40:00 \
  --export=ALL,TASK=Unitree-G1-Claas-23-Eval-E1_glatt,INIT_CKPT=<Checkpoint>,MAX_ITER=60,VIDEO=0 \
  claas/ms_train.sh
```

Die vollständigen Abläufe (Smoke-Test → Training → Videos → MuJoCo, per Slurm-Abhängigkeit verkettet) stehen in `claas/g23_kette*.sh`. Ein Beispiel ist `claas/g23_kette4.sh` für den Robustheitstest sowie A7, A8 und die Seeds.

**Task-Namen der Läufe** (`claas/unitree_run.py`):

| Lauf | Task | Klasse |
| --- | --- | --- |
| A1 / A2 | `Unitree-G1-Claas-23-Flat` / `-FlatTrack` | `G23FlatEnvCfg` / `G23FlatTrackEnvCfg` |
| A3 | `Unitree-G1-Claas-23-StairsTrack` | `G23StairsTrackEnvCfg` |
| A4 | `Unitree-G1-Claas-23-Robust` | `G23RobustEnvCfg` |
| A5, A5s1/s2 | `Unitree-G1-Claas-23-RobustTurn` | `G23RobustTurnEnvCfg` |
| A6 | `Unitree-G1-Claas-23-Gait` | `G23GaitEnvCfg` |
| A7 / A7b / A7c | `-Sym` / `-SymOnly` / `-GaitMod` | `G23SymEnvCfg` / `G23SymOnlyEnvCfg` / `G23GaitModEnvCfg` |
| A8 | `Unitree-G1-Claas-23-Rough` | `G23RoughEnvCfg` |
| A9 | `Unitree-G1-Claas-23-Slip` | `G23SlipEnvCfg` |
| A10 / A11 | `-StairsA10` / `-RoughSlip` | `G23StairsA10EnvCfg` / `G23RoughSlipEnvCfg` |
| T1–T3 | `-Stairs` / `-StairsTrack` | `G23StairsEnvCfg` / `G23StairsTrackEnvCfg` |
| T4 | `-StairsRobust` | `G23StairsRobustEnvCfg` |
| L1 / L2 | `-Teacher` / `-TeacherRobust` | `G23TeacherEnvCfg` / `G23TeacherRobustEnvCfg` |
| E0–E6 | `Unitree-G1-Claas-23-Eval-E*` | `G23Eval_E*` |

Checkpoints liegen unter `runs/logs/rsl_rl/<experiment>/<Datum_Uhrzeit>/model_<It>.pt`. Die Zuordnung Lauf → Ordner → Job-ID steht in Notion („Experimente“).

## Zusammenarbeit

- Ergebnisse und Entscheidungen: Notion, Teamspace RobotChallenge, Datenbank „Experimente“
- Code: dieses Repository. Auf dem Cluster `git pull`, dann `sbatch`.
- Neue Läufe: neue Klasse in `cpr/`, Task in `claas/unitree_run.py` registrieren, Eintrag in `docs/EXPERIMENTE.md` mit Frage, Änderung, Start-Checkpoint, Ergebnis und Schluss.
