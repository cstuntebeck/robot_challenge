#!/bin/bash
cd /bigwork/nhkbrc2a/isaac
O=runs/claas/kurz6.txt
python3 claas/mittel30.py A5s2_E0_nominal=logs/slurm_claas_ms_9425268.out A5s2_E1_glatt=logs/slurm_claas_ms_9425269.out A5s2_E2_stoesse=logs/slurm_claas_ms_9425270.out A5s2_E3_latenz=logs/slurm_claas_ms_9425271.out A5s2_E4_last=logs/slurm_claas_ms_9425273.out A5s2_E5_motor=logs/slurm_claas_ms_9425274.out A5s2_E6_kombi=logs/slurm_claas_ms_9425275.out A8_E0_nominal=logs/slurm_claas_ms_9425276.out A8_E1_glatt=logs/slurm_claas_ms_9425277.out A8_E2_stoesse=logs/slurm_claas_ms_9425278.out A8_E3_latenz=logs/slurm_claas_ms_9425279.out A8_E4_last=logs/slurm_claas_ms_9425280.out A8_E5_motor=logs/slurm_claas_ms_9425281.out A8_E6_kombi=logs/slurm_claas_ms_9425282.out A9_train=logs/slurm_claas_ms_9425284.out > $O 2>&1
tail -c 8000 logs/slurm_claas_ms_9425284.out | grep -E "Learning iteration|track_ang_vel|error_vel|bad_orientation" | tail -5 >> $O
sacct -j 9425283,9425284,9425285,9425286 -X --format=JobID,JobName%20,State,Elapsed -n >> $O
cat claas/deploy/PRUEFUNG.txt >> $O 2>&1
ls runs/sim2sim runs/videos | grep A9 >> $O
