#!/bin/bash
#SBATCH --job-name=isaac_sif
#SBATCH --partition=mpp.share
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_sif_%j.out

set -u
BASE=/bigwork/nhkbrc2a/isaac
WORK=${TMPDIR:-/tmp}/sifbuild
mkdir -p "$WORK"
export APPTAINER_TMPDIR=$WORK/tmp
mkdir -p "$APPTAINER_TMPDIR"

echo "=== $(hostname): untar sandbox"
tar -C "$WORK" -xf "$BASE/containers/sandbox.tar"
echo "UNTAR_EXIT: $?"

echo "=== building SIF"
apptainer build "$WORK/isaac-lab-2.3.2.sif" "$WORK/isaac-lab-sandbox"
rc=$?
echo "BUILD_EXIT: $rc"
if [ $rc -eq 0 ]; then
  mv "$WORK/isaac-lab-2.3.2.sif" "$BASE/containers/"
  echo "MOVE_EXIT: $?"
  rm -f "$BASE/containers/sandbox.tar"
fi
rm -rf "$WORK"
echo "SIF_JOB_DONE"
# Exit-Code weiterreichen - sonst meldet Slurm immer COMPLETED 0:0.
exit $rc
