# Zwei RealSense D435 am G1: Simulation auf eurem Cluster-Account

Auf eurem Account beim LUIS-Cluster liegt eine fertige Simulation. Sie zeigt,
was zwei Intel RealSense D435 vorne am Unitree G1 sehen, während der Roboter
Treppen hinaufläuft. Ihr wählt, wo die Halterung sitzt (Hüfte oder Brust) und
ob nur die untere oder beide Kameras dran sind, schickt einen Job ab und
bekommt nach etwa 20 Minuten Videos.

Für den ersten Challenge Day braucht ihr das nicht: Die Policy dort ist blind.
Die Simulation ist zum Ausprobieren und um die Montage einzuschätzen.

## Was simuliert wird

- **Roboter:** Unitree G1 mit 29 Gelenken, in Isaac Lab 2.3.2 / Isaac Sim 5.1.
- **Laufen:** die fertige Policy von NVIDIA für den G1 (`Isaac-Velocity-Rough-G1`),
  vorwärts mit 0,6 m/s, eine Treppe hinauf. Die Policy ist blind; sie nutzt die
  Kameras nicht.
- **Kameras:** D435 mit 87° × 55° Sichtfeld (horizontal × vertikal),
  Tiefenbereich 0,3 bis 3,0 m, Bildgröße 320 × 176 Pixel.
- **Halterung:** zwei Kameras übereinander auf einer Platte vor dem Körper,
  2 cm Luft zum Körper. Die Gehäuse sind einfache Quader in D435-Größe
  (90 × 25 × 25 mm), die Platte ebenso.
  - **Untere Kamera:** 62,4° nach unten geneigt. Ihr unterer Bildrand zeigt
    senkrecht nach unten, also auf den Boden direkt vor den Füßen.
  - **Obere Kamera:** 9,3° nach unten geneigt. Die Sichtfelder überlappen sich um 2°.
  - Zusammen sehen beide Kameras von senkrecht unten bis etwa 18° über die Waagrechte.

## Die vier Einstellungen

| | nur untere Kamera | beide Kameras |
|---|---|---|
| **Hüfte** (`pelvis`) | `MOUNTS=pelvis VARIANTS=unten` | `MOUNTS=pelvis VARIANTS=beide` |
| **Brust** (`torso_link`) | `MOUNTS=torso_link VARIANTS=unten` | `MOUNTS=torso_link VARIANTS=beide` |

Beim Gehen sitzt die untere Kamera im Mittel **0,64 m** (Hüfte) bzw.
**0,91 m** (Brust) über dem Boden; gemessen in einem Probelauf mit dieser
Simulation. „Nur die obere Kamera“ gibt es nicht als Einstellung.

## Starten

1. **Anmelden.** Im LUH-Netz oder über VPN:
   ```
   ssh <account>@login.cluster.uni-hannover.de
   ```
2. **Job abschicken.** Ohne weitere Angaben werden alle vier Kombinationen
   gerendert, je ein Video von 15 s:
   ```
   sbatch $BIGWORK/isaac/scripts/submit_sensor_view.sh
   ```
   Nur eine Auswahl, zum Beispiel Brust mit beiden Kameras:
   ```
   MOUNTS=torso_link VARIANTS=beide sbatch $BIGWORK/isaac/scripts/submit_sensor_view.sh
   ```
   Beide Montageorte, jeweils nur die untere Kamera:
   ```
   MOUNTS="pelvis torso_link" VARIANTS=unten sbatch $BIGWORK/isaac/scripts/submit_sensor_view.sh
   ```
   Weitere Stellschrauben, ebenfalls vor `sbatch` zu setzen:

   | Variable | Bedeutung | Standard |
   |---|---|---|
   | `STEPS` | Länge des Videos in Steuerschritten (50 pro Sekunde) | `750` (15 s) |
   | `SPEED` | Vorwärtsgeschwindigkeit in m/s | `0.6` |
   | `CLEARANCE` | Luft zwischen Körper und Halterungsplatte in m | `0.02` |

3. **GPU.** Kamerabilder brauchen eine Grafikkarte mit RT-Kernen. Der Job
   fordert eine L40S in der Partition `imes.gpu` an.
   - Werktags von 8 bis 20 Uhr ist diese Partition reserviert. Bleibt der Job
     tagsüber hängen, hängt `--reservation=imes.gpu` an:
     ```
     sbatch --reservation=imes.gpu $BIGWORK/isaac/scripts/submit_sensor_view.sh
     ```
   - Ausweichen auf die A40: `sbatch -p phdgpu …`
   - Rechenkarten ohne RT-Kerne (A100, H100, H200) liefern hier keine Kamerabilder.
   - Alle vier Kombinationen dauern auf der L40S etwa 22 Minuten. Das Zeitlimit
     des Jobs ist 1 Stunde. Für längere Videos (`STEPS` größer) gebt mehr Zeit
     mit, z. B. `sbatch --time=02:00:00 …`.

4. **Verfolgen.**
   ```
   squeue -u $USER                                               # wartet / läuft
   tail -f $BIGWORK/isaac/logs/slurm_sensor_view_<jobid>.out     # Log live
   sacct -j <jobid>                                              # nach dem Ende
   ```
   Fertig und erfolgreich ist der Job, wenn das Log mit `SENSOR_VIEW_DONE rc=0`
   endet. Für jede Kombination steht vorher eine Zeile `SENSOR_VIEW_OK`.
   Abstürze in den ersten drei Minuten wiederholt der Job selbst.

## Ergebnisse

Alles liegt in `$BIGWORK/isaac/runs/sensor_view/<jobid>/`, je Kombination
(z. B. `torso_link_beide`):

| Datei | Inhalt |
|---|---|
| `torso_link_beide.mp4` | links der Roboter von der Seite, rechts Farb- und Tiefenbild jeder Kamera (oben die obere, unten die untere, wie montiert) |
| `torso_link_beide_t02s.png`, `_t06s.png`, `_t10s.png` | Standbilder nach 2, 6 und 10 s |
| `torso_link_beide.json` | während des Laufs gemessen: Neigung und Höhe jeder Kamera über dem Boden, kleinster Abstand der Ellbogen zum Gehäuse, `"ok": true`, `"probleme"` |

Herunterladen auf den eigenen Rechner, über den Transfer-Knoten:
```
scp '<account>@transfer.cluster.uni-hannover.de:/bigwork/<account>/isaac/runs/sensor_view/<jobid>/*.mp4' .
```

## Was das (noch) nicht ist

- **Keine Eingabe für eine Policy.** Die Kameras nehmen nur auf. Eine Policy,
  die mit den Kamerabildern läuft, gibt es noch nicht.
- **Keine eigene Policy.** Das Beispiel läuft fest mit der Policy von NVIDIA.
- **Vereinfachte Halterung.** In der Simulation sind Gehäuse und Platte
  Quader. Die Bilder kommen aus der Mitte der Gehäusefront; bei der echten
  D435 liegt der Tiefen-Ursprung im linken Bildsensor, 17,5 mm seitlich.
- **Verdeckung nicht beziffert.** Wie viel der eigene Körper vom Bild
  verdeckt, zeigen nur die Videos, keine Zahl.

## Halterung zum Drucken

Für die **Brust**montage gibt es eine gedruckte Halterung als CAD: STEP, STL
und eine Druckdatei für den Bambu P2S in PETG. Die Kameras sitzen darin wie
in der Simulation. Die obere Kamera sitzt 5 mm höher, damit die Gehäuse sich
nicht berühren. Die Dateien bekommt ihr auf Nachfrage.

## Wenn etwas nicht klappt

- **Job startet nicht:** `squeue -u $USER` zeigt den Grund in der letzten
  Spalte. Tagsüber meistens die Reservierung, siehe Schritt 3.
- **Im Log steht `FEHLER: … ohne Video`:** Die Zeilen davor zeigen, woran es
  lag. Schickt uns die Job-ID.
- **Bitte die Dateien unter `$BIGWORK/isaac/scripts/` nicht ändern.** Wir
  pflegen sie für beide Gruppen gleich. Zum Experimentieren eine Kopie anlegen.
