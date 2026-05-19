#!/usr/bin/env bash
# Aurik 9.12.9-hotfix.1 — Startskript mit venv-Python (.venv_aurik, Python 3.10.12)
# GPU-Modus: ROCm-venv auf ext4 (~/.local/share/aurik/venv_rocm) + /dev/kfd vorhanden
# Verwendung: ./run_aurik.sh [Argumente]
#   AURIK_FORCE_CPU=1  ./run_aurik.sh  — erzwingt CPU-only (deaktiviert ROCm)
#
# Hinweis: Das ROCm-venv liegt absichtlich auf ext4 (~/.local/share/aurik/venv_rocm),
# da ROCm GPU Code Objects per mmap() aus ELF-Sektionen geladen werden und
# FUSE/fuseblk (NTFS) dieses mmap nicht unterstützt → hipErrorInvalidDeviceFunction.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_CPU="$SCRIPT_DIR/.venv_aurik/bin/python"
# ROCm-venv liegt auf ext4 (Home), nicht auf dem FUSE/NTFS-Workspace-Laufwerk
VENV_ROCM="$HOME/.local/share/aurik/venv_rocm/bin/python"
PID_FILE="$SCRIPT_DIR/temp_repro/aurik_gui.pid"
LOG_FILE="$SCRIPT_DIR/logs/aurik_frontend.out"

# GPU-Erkennung: ROCm-venv (ext4) + KFD-Device vorhanden und nicht explizit deaktiviert
if [[ "${AURIK_FORCE_CPU:-0}" != "1" && -x "$VENV_ROCM" && -e "/dev/kfd" ]]; then
    VENV_PYTHON="$VENV_ROCM"
    _GPU_MODE="ROCm (AMD GPU)"
    # ORT's libonnxruntime_providers_rocm.so benötigt libhipblas.so.2, libhipfft.so etc.
    # Diese liegen im PyTorch-lib-Verzeichnis des ROCm-venv (ext4).
    _TORCH_LIB="$HOME/.local/share/aurik/venv_rocm/lib/python3.10/site-packages/torch/lib"
    if [[ -d "$_TORCH_LIB" ]]; then
        export LD_LIBRARY_PATH="${_TORCH_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    # .pth-Bridge: aurik_bridge.pth im ROCm-venv-Site-Packages verweist auf venv_aurik-Pakete.
    # .pth-Dateien werden NACH den eigenen Site-Packages geladen → ROCm-torch hat Vorrang.
else
    VENV_PYTHON="$VENV_CPU"
    _GPU_MODE="CPU-only"
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "FEHLER: venv-Python nicht gefunden: $VENV_PYTHON" >&2
    echo "Bitte zuerst: bash scripts/install_aurik.sh" >&2
    echo "Alternativ: python3 -m venv .venv_aurik && .venv_aurik/bin/pip install -r requirements/requirements_aurik.txt" >&2
    exit 1
fi

mkdir -p "$SCRIPT_DIR/temp_repro" "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"

echo "Aurik GPU-Modus: ${_GPU_MODE} (Python: ${VENV_PYTHON})"

# Numba-JIT deaktivieren: verhindert Circular-Import-Crash in ROCm-venv-Threads
# (numba >= 0.57 entfernt is_nonelike aus numba.core.cgutils; librosa triggert numba)
export NUMBA_DISABLE_JIT=1

# Kein Doppelstart: verhindert UI-Konflikte und wiederholte Force-Quit-Dialoge.
if pgrep -f "[A]urik910/main.py" >/dev/null 2>&1; then
    _pid="$(pgrep -f "[A]urik910/main.py" | head -n 1)"
    echo "Aurik läuft bereits (PID ${_pid})."
    exit 0
fi

# In VS Code-Terminals detach starten, damit VS Code den GUI-Prozess nicht verwaltet.
if [[ "${TERM_PROGRAM:-}" == "vscode" ]]; then
    nohup "$VENV_PYTHON" Aurik910/main.py "$@" >>"$LOG_FILE" 2>&1 &
    _pid="$!"
    echo "$_pid" >"$PID_FILE"
    echo "Aurik detached gestartet (PID ${_pid}). Log: $LOG_FILE"
    exit 0
fi

exec "$VENV_PYTHON" Aurik910/main.py "$@"
