#!/bin/bash
#SBATCH --job-name=g23_s2sdiag_claas
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/bigwork/nhkbrc2a/isaac/logs/slurm_claas_s2s_%j.out
# Diagnose Sim-to-Sim (Claas, 19.11.): 1) nur PD-Regelung auf Grundpose, 2) Policy, 3) Policy + Armature 0,01
cd /bigwork/nhkbrc2a/isaac
for V in "stand:S2S_NOPOLICY=1" "policy:S2S_NOPOLICY=0" "policy_arm:S2S_ARMATURE=0.01" "policy_arm_dt5:S2S_ARMATURE=0.01 S2S_DT=0.005"; do
  name=${V%%:*}; envs=${V#*:}
  echo "===== VARIANTE $name ($envs)"
  env $envs bash claas/sim2sim/run_sim2sim.sh "$POLICY" "$DEPLOY" "diag_${name}" 2>&1 | grep -v -i warn | grep -E "MJCF|ctrlrange|Armature|t=|ERGEBNIS|Error|error|abschnitt': 'stehen'|abschnitt': 'vor 0,5'"
done
