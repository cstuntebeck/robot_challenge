#!/bin/bash
#SBATCH --job-name=extract_assets
#SBATCH --partition=mpp.share
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_extract_%j.out

set -u
D=/bigwork/nhkbrc2a/isaac
cd "$D/downloads" || exit 1

# Die drei Teile sind ein Split-Zip: Teil 1+2 sind Rohsegmente ohne Central
# Directory, nur Teil 3 hat es. Die Zip-Split-Namenskonvention verlangt
# .z01/.z02/.zip - deshalb erst umbenennen. Idempotent, damit ein Neustart
# des Jobs nicht scheitert.
for pair in "001:assets.z01" "002:assets.z02" "003:assets.zip"; do
  src="isaac-sim-assets-complete-5.1.0.${pair%%:*}.zip"
  dst="${pair##*:}"
  [ -f "$src" ] && mv -v "$src" "$dst"
done

for f in assets.z01 assets.z02 assets.zip; do
  if [ ! -f "$f" ]; then
    echo "FEHLT: $D/downloads/$f - Download unvollstaendig, Abbruch."
    exit 1
  fi
done
ls -lh assets.z01 assets.z02 assets.zip

python3 $D/scripts/extract_split_zip.py \
  $D/downloads/assets.z01 $D/downloads/assets.z02 $D/downloads/assets.zip \
  $D/assets
rc=$?
echo "EXTRACT_EXIT: $rc"

# Erfolgskriterium der Anleitung direkt mitpruefen
ls $D/assets/Assets/Isaac/5.1/ 2>/dev/null
ls $D/assets/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/Unitree/ 2>/dev/null
echo "EXTRACT_JOB_DONE"
# Exit-Code weiterreichen - sonst meldet Slurm immer COMPLETED 0:0.
exit $rc
