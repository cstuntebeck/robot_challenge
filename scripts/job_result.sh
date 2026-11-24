#!/bin/bash
# Gemeinsame Funktionen fuer Job-Skripte - per `source` einbinden, nicht ausfuehren.
#
# Warum es das gibt: Der Smoke-Test 8178097 galt als bestanden, obwohl sein
# Checkpoint nie auf BIGWORK ankam. train.py schreibt relativ zum
# Arbeitsverzeichnis; das war $HOME, und $HOME ist im Container ein
# Wegwerf-Verzeichnis (run_isaaclab.sh: --home + trap). Ausserdem endeten die
# Skripte mit `echo`, Slurm meldete deshalb immer COMPLETED 0:0 - auch bei einem
# Absturz. Ab jetzt prueft jeder Job selbst, ob seine Ausgaben existieren, und
# sagt das Ergebnis zweimal: als Exit-Code und als runs/results/<jobid>.json
# (gelesen von scripts/verify.py).

# Laufverzeichnis aus dem Slurm-Log. train.py druckt beide Teile:
#   [INFO] Logging experiment in directory: <root>
#   Exact experiment name requested from command line: <name>
# Bei Retries zaehlt der letzte Versuch, deshalb jeweils tail -n 1.
job_run_dir() {
  local log="$1" root name
  root=$(grep -a 'Logging experiment in directory:' "$log" | tail -n 1 | sed 's/.*directory: *//')
  name=$(grep -a 'Exact experiment name requested from command line:' "$log" | tail -n 1 | sed 's/.*command line: *//')
  [ -n "$root" ] && [ -n "$name" ] && echo "$root/$name"
}

# Neuester Checkpoint model_<N>.pt (sort -V sortiert numerisch: model_10 nach model_9).
job_last_checkpoint() {
  ls -1 "$1"/model_*.pt 2>/dev/null | sort -V | tail -n 1
}

# Iterationsnummer aus .../model_<N>.pt
job_checkpoint_iter() {
  local n="${1##*model_}"
  echo "${n%.pt}"
}

# Ergebnis als JSON: erst die Zieldatei, dann key=value-Paare.
# Ganzzahlen bleiben Zahlen, alles andere wird ein String.
job_write_result() {
  local out="$1"; shift
  local first=1 kv k v
  mkdir -p "$(dirname "$out")"
  {
    printf '{'
    for kv in "$@"; do
      k=${kv%%=*}; v=${kv#*=}
      [ $first -eq 1 ] || printf ', '
      first=0
      if [[ "$v" =~ ^-?[0-9]+$ ]]; then
        printf '"%s": %s' "$k" "$v"
      else
        v=${v//\\/\\\\}; v=${v//\"/\\\"}; v=${v//$'\n'/ }
        printf '"%s": "%s"' "$k" "$v"
      fi
    done
    printf '}\n'
  } > "$out"
  echo "RESULT_JSON: $out"
  cat "$out"
}
