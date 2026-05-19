#!/usr/bin/env bash
# =============================================================================
# Aurik 9 – Installationsskript (Out-of-the-Box, §13.1 / §13.4)
# =============================================================================
# Verwendung:
#   bash scripts/install_aurik.sh              # Frische Installation
#   bash scripts/install_aurik.sh --venv PATH  # Eigenes venv-Verzeichnis
#   bash scripts/install_aurik.sh --no-venv    # Aktuelles venv/Python nutzen
#
# Anforderungen:
#   - Python 3.10 oder 3.11
#   - System-Pakete (Ubuntu/Debian): libportaudio2 portaudio19-dev ffmpeg
# =============================================================================

set -euo pipefail

# --- Farben für Ausgabe ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[Aurik]${NC} $*"; }
success() { echo -e "${GREEN}[Aurik]${NC} ✅ $*"; }
warn()    { echo -e "${YELLOW}[Aurik]${NC} ⚠️  $*"; }
error()   { echo -e "${RED}[Aurik]${NC} ❌ $*"; exit 1; }

# --- Standardwerte ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv_aurik"
USE_EXISTING_VENV=false
PYTHON_BIN=""

# --- Argumente parsen ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --venv)   VENV_DIR="$2"; shift 2 ;;
        --no-venv) USE_EXISTING_VENV=true; shift ;;
        -h|--help)
            echo "Verwendung: $0 [--venv PFAD] [--no-venv]"
            echo "  --venv PFAD   Venv in angegebenem Pfad anlegen (Standard: .venv_aurik)"
            echo "  --no-venv     Aktuell aktives Python / venv verwenden"
            exit 0 ;;
        *) warn "Unbekanntes Argument: $1"; shift ;;
    esac
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Aurik 9 – Intelligente Musik-Restaurierung         ║"
echo "║                    Installations-Skript                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
# SCHRITT 0: System-Abhängigkeiten prüfen
# =============================================================================
info "Schritt 0/5: System-Abhängigkeiten prüfen..."

check_system_dep() {
    local pkg=$1 cmd=$2
    if ! command -v "$cmd" &>/dev/null && ! ldconfig -p 2>/dev/null | grep -q "$pkg"; then
        warn "System-Paket '$pkg' nicht gefunden. Bitte installieren:"
        case "$(uname -s)" in
            Linux)
                if command -v apt-get &>/dev/null; then
                    warn "  sudo apt-get install -y $pkg"
                elif command -v dnf &>/dev/null; then
                    warn "  sudo dnf install -y $pkg"
                fi ;;
        esac
    fi
}

ldconfig -p 2>/dev/null | grep -q libportaudio && success "libportaudio2 vorhanden" \
    || warn "libportaudio2 fehlt → sudo apt-get install -y libportaudio2 portaudio19-dev"
command -v ffmpeg &>/dev/null && success "ffmpeg vorhanden" \
    || warn "ffmpeg fehlt → sudo apt-get install -y ffmpeg"

# =============================================================================
# SCHRITT 1: Python-Version ermitteln
# =============================================================================
info "Schritt 1/5: Python-Interpreter suchen..."

if $USE_EXISTING_VENV; then
    PYTHON_BIN=$(command -v python3 || command -v python)
else
    for py in python3.11 python3.10 python3; do
        if command -v "$py" &>/dev/null; then
            VER=$("$py" -c "import sys; print(sys.version_info[:2])")
            if [[ "$VER" == "(3, 10)" || "$VER" == "(3, 11)" ]]; then
                PYTHON_BIN=$(command -v "$py")
                break
            fi
        fi
    done
fi

[[ -z "$PYTHON_BIN" ]] && error "Python 3.10 oder 3.11 nicht gefunden. Bitte installieren."
PY_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
success "Python $PY_VERSION gefunden: $PYTHON_BIN"

# =============================================================================
# SCHRITT 2: Virtuelle Umgebung anlegen (falls gewünscht)
# =============================================================================
if ! $USE_EXISTING_VENV; then
    info "Schritt 2/5: Virtuelle Umgebung anlegen in: $VENV_DIR"

    if [[ -d "$VENV_DIR" ]]; then
        warn "venv existiert bereits – wird wiederverwendet: $VENV_DIR"
    else
        "$PYTHON_BIN" -m venv "$VENV_DIR"
        success "Virtuelle Umgebung erstellt"
    fi

    PYTHON_BIN="$VENV_DIR/bin/python"
    PIP_BIN="$VENV_DIR/bin/pip"
else
    info "Schritt 2/5: Vorhandenes Python verwenden (--no-venv)"
    PIP_BIN=$(command -v pip3 || command -v pip)
    success "pip: $PIP_BIN"
fi

PIP_BIN="$PYTHON_BIN -m pip"

# pip upgraden
$PYTHON_BIN -m pip install --upgrade pip --quiet
success "pip aktuell"

# =============================================================================
# SCHRITT 3: PyTorch CPU-only installieren (§13.4)
# =============================================================================
info "Schritt 3/5: PyTorch CPU-only installieren (§13.4)..."
info "Verwende: torch==2.7.0+cpu  torchaudio==2.7.0+cpu"
info "(CPU-only, kein CUDA nötig – funktioniert auf jeder Desktop-Hardware)"
info "Sicherheitsfix: torch>=2.6.0 behebt CVE-2025-32434 (critical RCE/deserialization)"

TORCH_INSTALLED=$($PYTHON_BIN -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [[ "$TORCH_INSTALLED" == "2.7.0+cpu" ]]; then
    success "PyTorch 2.7.0+cpu bereits installiert – überspringe"
else
    if [[ -n "$TORCH_INSTALLED" ]]; then
        warn "Andere torch-Version gefunden: $TORCH_INSTALLED → wird auf 2.7.0+cpu aktualisiert"
    fi
    $PYTHON_BIN -m pip install \
        torch==2.7.0+cpu \
        torchaudio==2.7.0+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --quiet
    success "PyTorch 2.7.0+cpu installiert"
fi

# =============================================================================
# SCHRITT 4: Hauptabhängigkeiten installieren
# =============================================================================
info "Schritt 4/5: Aurik-Abhängigkeiten installieren..."
info "Datei: $PROJECT_DIR/requirements/requirements_aurik.txt"

$PYTHON_BIN -m pip install \
    -r "$PROJECT_DIR/requirements/requirements_aurik.txt" \
    --quiet

success "Alle Abhängigkeiten installiert"

# =============================================================================
# SCHRITT 5: Modell-Integrität prüfen (§13.3)
# =============================================================================
info "Schritt 5/5: Lokale ML-Modelle prüfen (§13.3)..."

MANIFEST="$PROJECT_DIR/models/manifest.json"
if [[ -f "$MANIFEST" ]]; then
    BUNDLED_COUNT=$($PYTHON_BIN -c "
import json, pathlib
manifest = json.loads(pathlib.Path('$MANIFEST').read_text())
models = manifest.get('models', [])
ok = sum(1 for m in models if m.get('bundled') and pathlib.Path(m.get('bundled_path', '')).exists())
total = sum(1 for m in models if m.get('bundled'))
print(f'{ok}/{total}')
" 2>/dev/null || echo "?/?")
    success "Lokal gebündelte Modelle verfügbar: $BUNDLED_COUNT"
else
    warn "models/manifest.json nicht gefunden"
fi

# =============================================================================
# ABSCHLUSS
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  Installation abgeschlossen                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
success "Aurik 9 ist einsatzbereit!"
echo ""
info "Aurik starten:"
if ! $USE_EXISTING_VENV; then
    echo "  source $VENV_DIR/bin/activate"
fi
echo "  ./run_aurik.sh"
echo "  # Legacy-Kompatibilität: python start_aurik_90.py"
echo ""
info "Tests ausführen:"
if ! $USE_EXISTING_VENV; then
    echo "  source $VENV_DIR/bin/activate"
fi
echo "  ./run_tests_safe.sh tests/unit --maxfail=5 --tb=short"
echo ""
warn "Optionale Groß-Pakete (für SOTA-Upgrades, §13.3):"
echo "  pip install madmom>=0.16.0       # Beat-Tracking (GrooveMetric)"
echo "  pip install transformers>=4.40.0 # CLAP / EraClassifier SOTA-Upgrade"
echo "  pip install audiosr>=1.0.0       # Bandbreiten-Erweiterung (5.9 GB!)"
echo ""
