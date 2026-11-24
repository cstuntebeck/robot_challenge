#!/usr/bin/env bash
# Einmaliger Test: kann der Besitzer auf BIGWORK Gruppen-Leserechte setzen
# (chmod, ACL)? Arbeitet NUR in einem frisch angelegten Testordner und loescht
# ihn am Ende wieder. /bigwork/<user> selbst wird nur gelesen (stat, getfacl),
# nicht veraendert - dessen 0700 ist heute der einzige Schutz fuer alles darunter.
# Laeuft auf dem Login-Knoten per `cluster tmux-run`, Ausgabe in logs/perm_test.log.
set -u
BASE="/bigwork/nhkbrc2a/isaac"
LOG="$BASE/logs/perm_test.log"
T="$BASE/perm_test_tmp"
exec > "$LOG" 2>&1

echo "=== $(date -Is) $(hostname)"
echo "=== id / Gruppe (wer bekaeme Leserechte):"
id
getent group "$(id -gn)" || echo "getent group: nicht verfuegbar"

echo "=== Ist-Zustand oberster Ordner (nur lesen):"
stat -c '%A %a %U:%G %n' "$(dirname "$BASE")" "$BASE" "$BASE/assets"
getfacl -p "$(dirname "$BASE")" 2>&1 || echo "getfacl: nicht verfuegbar"

echo "=== Test im eigenen Ordner $T"
[ -e "$T" ] && { echo "ABBRUCH: $T existiert schon"; exit 1; }
mkdir -m 0700 "$T" && mkdir -m 0700 "$T/sub" && echo test > "$T/sub/f" && chmod 0600 "$T/sub/f"
stat -c '%a %n' "$T" "$T/sub" "$T/sub/f"

echo "--- chmod: Ordner 0710 (Gruppe durchgehen, nicht auflisten), Inhalt g+rX"
chmod 0710 "$T" && chmod -R g+rX "$T/sub"; echo "rc=$?"
stat -c '%a %n' "$T" "$T/sub" "$T/sub/f"

echo "--- ACL: Gruppe lesen per setfacl (inkl. default fuer neue Dateien)"
if command -v setfacl >/dev/null; then
  setfacl -R -m g::rX,d:g::rX "$T/sub"; echo "rc=$?"
  getfacl -p "$T/sub" "$T/sub/f"
else
  echo "setfacl: nicht verfuegbar"
fi

echo "--- Rueckbau"
rm -rf -- "$T" && echo "geloescht: $T"
[ -e "$T" ] && echo "FEHLER: $T noch da"
echo "=== fertig $(date -Is)"
