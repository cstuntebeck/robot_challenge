#!/bin/bash
# Login-Knoten: Sandbox bauen (nur entpacken), nach BIGWORK tarren, SIF-Job einreihen.
# Warum dieser Umweg: mksquashfs scheitert auf Lustre an den xattrs, und der
# SIF-Bau reisst das 30-min-CPU-Limit des Login-Knotens.
set -u
export https_proxy=http://proxy.luis.uni-hannover.de:3128
export http_proxy=http://proxy.luis.uni-hannover.de:3128
BASE=/bigwork/nhkbrc2a/isaac
LOG=$BASE/logs/sandbox.log
export APPTAINER_CACHEDIR=$BASE/apptainer_cache
export APPTAINER_TMPDIR=/tmp/${USER}_apt_tmp
SBX=/tmp/${USER}_sandbox
mkdir -p "$APPTAINER_TMPDIR"
rm -rf "$SBX"
mkdir -p "$SBX"        # Parent MUSS existieren, sonst "failed to create build parent dir"

echo "=== $(date) sandbox build on $(hostname) ===" >> "$LOG"
apptainer build --sandbox "$SBX/isaac-lab-sandbox" docker://nvcr.io/nvidia/isaac-lab:2.3.2 >> "$LOG" 2>&1
rc=$?
echo "SANDBOX_EXIT: $rc" >> "$LOG"
if [ $rc -ne 0 ]; then rm -rf "$SBX" "$APPTAINER_TMPDIR"; echo "CHAIN_FAILED" >> "$LOG"; exit 1; fi

echo "=== $(date) tar to bigwork ===" >> "$LOG"
tar -C "$SBX" -cf "$BASE/containers/sandbox.tar" isaac-lab-sandbox >> "$LOG" 2>&1
echo "TAR_EXIT: $?" >> "$LOG"
rm -rf "$SBX" "$APPTAINER_TMPDIR"

echo "=== $(date) submitting SIF conversion job ===" >> "$LOG"
sbatch "$BASE/scripts/submit_convert.sh" >> "$LOG" 2>&1
echo "CHAIN_DONE $(date)" >> "$LOG"
