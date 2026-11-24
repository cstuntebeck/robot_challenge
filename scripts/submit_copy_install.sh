#!/bin/bash
#SBATCH --job-name=copy_install
#SBATCH --partition=mpp.share
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_copy_install_%j.out

# T-006: SIF und Offline-Assets von einem anderen Account derselben Unix-Gruppe
# uebernehmen, statt 150 GB neu zu laden und zu entpacken (das dauerte im
# September Tage). Quelle per  --env SRC=<CLUSTER_BASE des Quell-Accounts>.
# Der Quell-Account muss vorher `cluster share grant` ausgefuehrt haben.
#
# Erfolg entscheidet nicht rsync, sondern der Vergleich am Ende: Dateizahl,
# Bytes, SIF bitgleich, keine Symlinks zurueck in die Quelle.
set -u
BASE=/bigwork/nhkbrc2a/isaac
RUNS=$BASE/runs
HELPER=$BASE/scripts/job_result.sh
[ -f "$HELPER" ] || { echo "FEHLER: $HELPER fehlt - erst Skripte hochladen"; exit 2; }
source "$HELPER"
cd "$RUNS" || exit 2

SRC="${SRC:-}"
SIF=isaac-lab-2.3.2.sif
RESULT="$RUNS/results/${SLURM_JOB_ID}.json"

fail() {
  echo "FEHLER: $1"
  job_write_result "$RESULT" kind=copy_install job_id="$SLURM_JOB_ID" src="$SRC" \
    exit_code="$2" reason="$1" node="$(hostname)"
  exit "$2"
}

[ -n "$SRC" ] || fail "SRC nicht gesetzt (--env SRC=...)" 2
[ "$SRC" != "$BASE" ] || fail "SRC ist der eigene Account" 2
[ -r "$SRC/containers/$SIF" ] || fail "SIF nicht lesbar: $SRC/containers/$SIF - Freigabe beim Quell-Account?" 3
[ -d "$SRC/assets/Assets" ] || fail "Assets nicht lesbar: $SRC/assets/Assets" 3

mkdir -p "$BASE/containers" "$BASE/assets"
echo "== $(date -Is) SIF"
rsync -a --info=stats1 "$SRC/containers/$SIF" "$BASE/containers/"; rc_sif=$?
echo "== $(date -Is) Assets"
rsync -a --info=stats1 "$SRC/assets/" "$BASE/assets/"; rc_assets=$?
echo "== $(date -Is) Vergleich"

files() { find "$1" -type f | wc -l; }
bytes() { find "$1" -type f -printf "%s\n" | awk '{s+=$1} END {printf "%d", s}'; }
n_src=$(files "$SRC/assets");  n_dst=$(files "$BASE/assets")
b_src=$(bytes "$SRC/assets");  b_dst=$(bytes "$BASE/assets")
back_links=$(find "$BASE/assets" -type l -lname "$SRC*" | wc -l)
if cmp -s "$SRC/containers/$SIF" "$BASE/containers/$SIF"; then sif_same=1; else sif_same=0; fi
echo "Dateien $n_src -> $n_dst | Bytes $b_src -> $b_dst | SIF gleich: $sif_same | Rueck-Links: $back_links"

rc=0
[ "$rc_sif" -eq 0 ] && [ "$rc_assets" -eq 0 ] || rc=1
[ "$n_src" -eq "$n_dst" ] && [ "$b_src" -eq "$b_dst" ] || rc=1
[ "$sif_same" -eq 1 ] && [ "$back_links" -eq 0 ] || rc=1
[ -d "$BASE/assets/Assets/Isaac/5.1/Isaac/IsaacLab" ] || rc=1

job_write_result "$RESULT" kind=copy_install job_id="$SLURM_JOB_ID" src="$SRC" \
  exit_code="$rc" rsync_sif="$rc_sif" rsync_assets="$rc_assets" \
  files_src="$n_src" files_dst="$n_dst" bytes_src="$b_src" bytes_dst="$b_dst" \
  sif_identical="$sif_same" back_links="$back_links" node="$(hostname)"
echo "COPY_INSTALL_DONE rc=$rc"
exit $rc
