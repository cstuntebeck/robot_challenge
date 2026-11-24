#!/bin/bash
# Isaac-Sim-5.1-Assetpacks herunterladen. NUR Login-Knoten (Compute hat kein Netz).
# Start ueber `cluster tmux-run isaacdl scripts/download_all.sh` - Dauer ~2 h.
set -u
export https_proxy=http://proxy.luis.uni-hannover.de:3128
export http_proxy=http://proxy.luis.uni-hannover.de:3128

BASE=/bigwork/nhkbrc2a/isaac
cd "$BASE/downloads" || exit 1

for i in 001 002 003; do
  echo "=== $(date) downloading part $i ===" >> "$BASE/logs/wget.log"
  wget -c -nv "https://downloads.isaacsim.nvidia.com/isaac-sim-assets-complete-5.1.0.$i.zip" \
       >> "$BASE/logs/wget.log" 2>&1
  echo "part $i exit: $?" >> "$BASE/logs/wget.log"
done
echo "ALL_ASSET_DOWNLOADS_DONE $(date)" >> "$BASE/logs/wget.log"
