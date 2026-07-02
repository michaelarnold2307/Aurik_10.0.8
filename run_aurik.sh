#!/usr/bin/env bash
# Aurik 9.15.0 — Startskript mit venv-Python (.venv_aurik, Python 3.10.12)
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
PIP_ROCM="$HOME/.local/share/aurik/venv_rocm/bin/pip"
PID_FILE="$SCRIPT_DIR/temp_repro/aurik_gui.pid"
LOG_FILE="$SCRIPT_DIR/logs/aurik_frontend.out"

# Release-Default: MIOpen meldet harmlose Workspace-Fallbacks sonst als WARNING
# direkt auf stderr. Fehler bleiben sichtbar; AURIK_DEBUG kann die Stufe anheben.
export MIOPEN_LOG_LEVEL="${MIOPEN_LOG_LEVEL:-1}"

check_rocm_torchaudio_abi() {
    "$VENV_ROCM" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"ROCM_STACK_ERR torch import failed: {exc}")
    raise SystemExit(10)

try:
    import torchaudio
except Exception as exc:
    print(f"ROCM_STACK_ERR torchaudio import failed: {exc}")
    raise SystemExit(11)

torch_ver = str(getattr(torch, "__version__", ""))
audio_ver = str(getattr(torchaudio, "__version__", ""))
torch_build = torch_ver.split("+", 1)[1] if "+" in torch_ver else ""
audio_build = audio_ver.split("+", 1)[1] if "+" in audio_ver else ""

if torch_build and audio_build and torch_build != audio_build:
    print(
        "ROCM_STACK_ERR build mismatch: "
        f"torch={torch_ver} torchaudio={audio_ver}"
    )
    raise SystemExit(12)

print(f"ROCM_STACK_OK torch={torch_ver} torchaudio={audio_ver}")
PY
}

repair_rocm_torchaudio() {
    if [[ ! -x "$PIP_ROCM" ]]; then
        echo "ROCM_STACK_ERR pip im ROCm-venv fehlt: $PIP_ROCM" >&2
        return 1
    fi

    local torch_version rocm_tag
    torch_version="$($VENV_ROCM - <<'PY'
import torch
print(getattr(torch, "__version__", ""))
PY
)"

    if [[ -z "$torch_version" || "$torch_version" != *+rocm* ]]; then
        echo "ROCM_STACK_ERR keine ROCm-Torch-Version erkannt: $torch_version" >&2
        return 1
    fi

    rocm_tag="${torch_version#*+}"
    echo "ROCM_STACK_REPAIR installiere torchaudio==$torch_version via $rocm_tag ..."
    "$PIP_ROCM" install --upgrade --index-url "https://download.pytorch.org/whl/$rocm_tag" \
        "torchaudio==$torch_version"
}

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
    set +e
    check_rocm_torchaudio_abi
    _rocm_stack_rc=$?
    set -e
    if [[ "$_rocm_stack_rc" -ne 0 ]]; then
        echo "Warnung: ROCm-Audio-Stack inkonsistent (torch/torchaudio), rc=${_rocm_stack_rc}." >&2
        if [[ "${AURIK_DISABLE_TORCHAUDIO_AUTO_REPAIR:-0}" != "1" ]] && repair_rocm_torchaudio; then
            set +e
            check_rocm_torchaudio_abi
            _rocm_stack_rc=$?
            set -e
        fi
        if [[ "$_rocm_stack_rc" -eq 0 ]]; then
            echo "ROCM_STACK_REPAIR erfolgreich." >&2
        elif [[ "$_rocm_stack_rc" -eq 11 || "$_rocm_stack_rc" -eq 12 ]]; then
            echo "Warnung: torchaudio bleibt defekt/inkompatibel; GPU bleibt AKTIV, torchaudio-abhängige Phasen fallen auf CPU/DSP zurück." >&2
            echo "Hinweis: Für erneuten Reparaturversuch AURIK_DISABLE_TORCHAUDIO_AUTO_REPAIR=0 setzen." >&2
            export AURIK_TORCHAUDIO_DEGRADED=1
            _GPU_MODE="ROCm (AMD GPU, torchaudio degraded → selective CPU/DSP fallback)"
        else
            echo "Warnung: ROCm-Basisstack defekt (torch nicht nutzbar). Fallback auf CPU-venv." >&2
            VENV_PYTHON="$VENV_CPU"
            _GPU_MODE="CPU-only (ROCm-Stack defekt)"
        fi
    fi
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
