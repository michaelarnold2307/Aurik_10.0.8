#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_tests_safe.sh — Crash-sicherer Pytest-Launcher für Aurik 9
# ══════════════════════════════════════════════════════════════════════════════
#
# PROBLEM: Wenn pytest unter VS Code (Snap) läuft, ist der Python-Prozess ein
# Kind-Prozess des VS Code-Prozessbaums. Ein OOM-Kill des Python-Prozesses durch
# den Linux-Kernel killt mitunter den gesamten VS Code-Prozessbaum → Absturz.
#
# LÖSUNG: Dieser Wrapper isoliert den Test-Prozess vollständig aus dem
# VS Code-Prozessbaum via systemd-run cgroup (bevorzugt) oder setsid+ulimit.
# Speicher-Cap: Python wird gekillt, NICHT VS Code.
#
# VERWENDUNG:
#   ./run_tests_safe.sh [pytest-argumente]
#
# BEISPIELE:
#   ./run_tests_safe.sh tests/unit -q --timeout=30
#   ./run_tests_safe.sh tests/unit tests/musical_goals --maxfail=5
#   AURIK_MEM_GB=12 ./run_tests_safe.sh tests/ -m "not ml and not e2e"
#
# UMGEBUNGSVARIABLEN:
#   AURIK_MEM_GB=8          Speicher-Cap in GB (Default: 8)
#   AURIK_TEST_RSS_LIMIT_MB=7000   RSS-Watchdog-Limit in conftest.py
#   AURIK_LOG_FILE=...      Pfad für Log-Datei (Default: logs/pytest_safe.log)
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${SCRIPT_DIR}/.venv_aurik/bin/python"
MEM_GB="${AURIK_MEM_GB:-8}"
MEM_BYTES=$(( MEM_GB * 1024 * 1024 * 1024 ))
MEM_MB=$(( MEM_GB * 1024 ))
MEM_KB=$(( MEM_GB * 1024 * 1024 ))
LOG_FILE="${AURIK_LOG_FILE:-${SCRIPT_DIR}/logs/pytest_safe.log}"
RSS_LIMIT_MB="${AURIK_TEST_RSS_LIMIT_MB:-$(( MEM_GB * 1024 * 85 / 100 ))}"

# Robuste Terminal-Capability-Umgebung fuer Snap/VS-Code-Subprozesse
export TERM="${TERM:-xterm-256color}"
export TERMINFO="${TERMINFO:-/usr/share/terminfo}"
export TERMINFO_DIRS="${TERMINFO_DIRS:-/usr/share/terminfo:/lib/terminfo:/etc/terminfo}"

# Konftest-Watchdog-Limit aus Speicher-Cap ableiten (85 % des Caps)
export AURIK_TEST_RSS_LIMIT_MB="$RSS_LIMIT_MB"

mkdir -p "$(dirname "$LOG_FILE")"

echo "══════════════════════════════════════════════════════"
echo " Aurik Safe Test Runner"
echo " Speicher-Cap : ${MEM_GB} GB"
echo " RSS-Watchdog : ${RSS_LIMIT_MB} MB"
echo " Log          : ${LOG_FILE}"
echo " Argumente    : $*"
echo "══════════════════════════════════════════════════════"

# ── Methode 1: systemd-run (beste Isolation via cgroup) ──────────────────────
# Erstellt eine eigene cgroup mit hartem Speicher-Limit. Der Python-Prozess
# wird vom Kernel in seiner eigenen cgroup gekillt — VS Code ist vollständig
# getrennt.
# Prüfe ob systemd-run --user --scope --collect unterstützt wird (systemd ≥ 236).
_SYSTEMD_OK=0
if command -v systemd-run &>/dev/null; then
    if systemd-run --user --scope --collect -- true 2>/dev/null; then
        _SYSTEMD_OK=1
    fi
fi

if [[ "$_SYSTEMD_OK" -eq 1 ]]; then
    echo "[safe-runner] Methode: systemd-run cgroup (MemoryMax=${MEM_GB}G)"
    set +e
    systemd-run \
        --user \
        --scope \
        --collect \
        --quiet \
        --setenv=TERM="$TERM" \
        --setenv=TERMINFO="$TERMINFO" \
        --setenv=TERMINFO_DIRS="$TERMINFO_DIRS" \
        -p "MemoryMax=${MEM_GB}G" \
        -p "MemorySwapMax=512M" \
        -p "CPUWeight=50" \
        -p "TasksMax=512" \
        -- \
        "$PYTHON" -m pytest "$@" \
        --override-ini="addopts=--strict-markers --import-mode=importlib" \
        -p no:xdist \
        --disable-warnings \
        --no-header \
        2>&1 | tee "$LOG_FILE"
    _rc=${PIPESTATUS[0]}
    set -e
    exit "$_rc"
fi

# ── Methode 2: setsid + ulimit (Fallback ohne systemd) ───────────────────────
# setsid: Trennt den Prozess von VS Codes Session → eigene Prozessgruppe.
# ulimit -v: Virtuelle Memory Cap. Wenn Python dieses Limit überschreitet,
# erhält es ENOMEM → Python beendet sich, VS Code überlebt.
echo "[safe-runner] Methode: setsid + ulimit -v ${MEM_MB}M"

(
    # Eigene Session → eigene Prozessgruppe → kein Signal-Forwarding zu VS Code
    exec setsid bash -c "
        ulimit -v $MEM_KB 2>/dev/null || true
        ulimit -m $MEM_KB 2>/dev/null || true
        exec '$PYTHON' -m pytest \"\$@\" \
            --override-ini='addopts=--strict-markers --import-mode=importlib' \
            -p no:xdist \
            --disable-warnings \
            --no-header
    " -- "$@" 2>&1 | tee "$LOG_FILE"
)
