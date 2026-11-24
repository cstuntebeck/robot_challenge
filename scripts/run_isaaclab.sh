#!/bin/bash
# Wrapper: ein Kommando im Isaac-Lab-Container ausfuehren, vollstaendig offline.
# Usage: run_isaaclab.sh /isaac-sim/python.sh <script.py> [args...]
#
# JEDE Zeile hier ist die Lehre aus einem konkreten Absturz - nicht kuerzen.
set -u
BASE=/bigwork/nhkbrc2a/isaac
SIF=$BASE/containers/isaac-lab-2.3.2.sif
BIGWORK_ROOT=$(dirname "$BASE")

# Relativ geschriebene Ausgaben (train.py: logs/rsl_rl/..., play.py:
# .pretrained_checkpoints/...) landen im Arbeitsverzeichnis. Liegt das unter
# $HOME, schreiben sie in das HOME des Containers - und das ist unten ein
# Wegwerf-Verzeichnis, das der trap beim Beenden loescht. So ging der Checkpoint
# des ersten Smoke-Tests (8178097) verloren, ohne dass etwas fehlschlug.
# Lieber laut abbrechen: Job-Skripte machen vorher `cd $BASE/runs`.
case "$PWD/" in
  "$HOME"/*)
    echo "[run_isaaclab] FEHLER: Arbeitsverzeichnis $PWD liegt unter \$HOME -" \
         "im Container fluechtig, Ausgaben gingen verloren. Vorher: cd $BASE/runs" >&2
    exit 2 ;;
esac

# Asset-Wurzel finden (der Ordner, der Isaac/ und NVIDIA/ enthaelt)
ASSET_ROOT=$(dirname "$(find "$BASE/assets" -maxdepth 4 -type d -name NVIDIA | head -1)")

# Ein per `sbatch --export=ALL` geerbtes DISPLAY laesst kit beim Start
# XOpenDisplay versuchen -> sporadischer Segfault. Bedingungslos toeten.
unset DISPLAY
export APPTAINERENV_DISPLAY=

export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_ACCEPT_EULA=Y
export APPTAINERENV_PRIVACY_CONSENT=N
export APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES
export APPTAINERENV_ISAACSIM_ASSET_ROOT="$ASSET_ROOT"
export APPTAINERENV_ISAAC_ASSET_ROOT="$ASSET_ROOT"

# Kit-Caches: PRO PROZESS auf knoten-lokalem Scratch. Teilen sich mehrere Jobs
# einen Cache, korrumpiert das den kvdb-Store -> sofortiger Segfault beim Start.
# Haelt ausserdem die kleine Home-Quota unangetastet.
CROOT="${TMPDIR:-/tmp}/kitcache_$$_${SLURM_JOB_ID:-x}"
mkdir -p "$CROOT/kitcache" "$CROOT/kitdata" "$CROOT/home" "$CROOT/shm"
trap 'rm -rf "$CROOT"' EXIT

# --home: vollstaendig isoliertes HOME pro Job. Teilen sich parallele kit-Prozesse
# das echte Home (Telemetrie, kvdb, Caches), schiessen sie sich gegenseitig ab.
# Privates /dev/shm: liegengebliebene Shared-Memory-Segmente hart gekillter
# kit-Prozesse lassen neue Instanzen beim Start abstuerzen.
apptainer exec --nv \
  --home "$CROOT/home:$HOME" \
  --bind "$BIGWORK_ROOT:$BIGWORK_ROOT" \
  --bind "$CROOT/kitcache:/isaac-sim/kit/cache" \
  --bind "$CROOT/kitdata:/isaac-sim/kit/data" \
  --bind "$CROOT/shm:/dev/shm" \
  "$SIF" "$@"
