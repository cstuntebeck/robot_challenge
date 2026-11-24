#!/bin/bash
# Kurzauswertung: letzte Metriken je Trainingsjob (Claas)
cd /bigwork/nhkbrc2a/isaac
O=runs/claas/kurz.txt; : > $O
for N in A7:9419436 A8:9419440 A5s2:9419448 A5s1b:9420433 A7b:9420437 A7c:9420441; do
  K=${N%%:*}; J=${N##*:}; F=logs/slurm_claas_ms_$J.out
  echo "== $K $J $(grep -c TRAIN_DONE $F 2>/dev/null)" >> $O
  tail -c 8000 $F 2>/dev/null | grep -E "Learning iteration|Mean reward|track_ang_vel|error_vel|terrain_levels|bad_orientation" | tail -7 | tr -s ' ' >> $O
done
sacct -j 9419437,9419438,9419441,9419442,9419449,9419450,9420434,9420435,9420438,9420439,9420442,9420443 -X --format=JobID,JobName%20,State,Elapsed -n >> $O
squeue -u $USER >> $O
ls runs/sim2sim/ | grep -E "A7|A8|A5s" >> $O
ls runs/videos/ | grep -E "A7|A8|A5s" >> $O
