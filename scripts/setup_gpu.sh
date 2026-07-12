#!/usr/bin/env bash
# setup_gpu.sh — Aurik GPU acceleration setup (Linux)
# ===================================================
# Detects GPU (NVIDIA CUDA / AMD ROCm / none) and creates
# .venv_gpu with the appropriate PyTorch + ONNX Runtime packages.
#
# Usage:
#   ./scripts/setup_gpu.sh              # auto-detect GPU
#   ./scripts/setup_gpu.sh --cuda       # force NVIDIA CUDA
#   ./scripts/setup_gpu.sh --rocm       # force AMD ROCm
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_GPU="$REPO_ROOT/.venv_gpu"
VENV_CPU="$REPO_ROOT/.venv_aurik"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${GREEN}=== Aurik GPU Setup (Linux) ===${NC}"

# ── Parse args ──────────────────────────────────────────────────────────
FORCE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda) FORCE="cuda"; shift ;;
        --rocm) FORCE="rocm"; shift ;;
        --cpu)  FORCE="cpu"; shift ;;
        *) echo "Usage: $0 [--cuda|--rocm|--cpu]"; exit 1 ;;
    esac
done

# ── Detect GPU ──────────────────────────────────────────────────────────
detect_gpu() {
    if [[ -n "$FORCE" ]]; then
        echo "$FORCE"
        return
    fi
    # NVIDIA detection
    if lspci 2>/dev/null | grep -qi "nvidia\|3D controller.*NVIDIA"; then
        echo "cuda"
        return
    fi
    # AMD detection (with ROCm-capable kernel driver)
    if lspci 2>/dev/null | grep -qi "amd\|advanced micro devices" && [[ -e /dev/kfd ]]; then
        echo "rocm"
        return
    fi
    echo "cpu"
}

GPU_TYPE=$(detect_gpu)
echo -e "Detected GPU: ${YELLOW}$GPU_TYPE${NC}"

if [[ "$GPU_TYPE" == "cpu" ]]; then
    echo -e "${YELLOW}No NVIDIA/AMD GPU detected. GPU acceleration skipped.${NC}"
    echo "Aurik will run with CPU-only mode (.venv_aurik)."
    exit 0
fi

# ── Check CPU venv exists ───────────────────────────────────────────────
if [[ ! -f "$VENV_CPU/bin/python" ]]; then
    echo -e "${RED}ERROR: .venv_aurik not found at $VENV_CPU${NC}"
    echo "Run the main Aurik setup first to create the CPU venv."
    exit 1
fi

# ── Create GPU venv ─────────────────────────────────────────────────────
echo -e "${GREEN}Creating GPU virtual environment at .venv_gpu ...${NC}"
"$VENV_CPU/bin/python" -m venv --clear "$VENV_GPU"
PIP="$VENV_GPU/bin/pip"
PYTHON="$VENV_GPU/bin/python"

# Upgrade pip
"$PIP" install --upgrade pip setuptools wheel -q

# ── Install GPU packages ────────────────────────────────────────────────
if [[ "$GPU_TYPE" == "cuda" ]]; then
    echo -e "${GREEN}Installing NVIDIA CUDA packages ...${NC}"
    "$PIP" install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
    "$PIP" install onnxruntime-gpu

elif [[ "$GPU_TYPE" == "rocm" ]]; then
    echo -e "${GREEN}Installing AMD ROCm packages ...${NC}"
    "$PIP" install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/rocm6.1
    "$PIP" install onnxruntime-rocm

fi

# ── Verify ──────────────────────────────────────────────────────────────
echo -e "${GREEN}Verifying GPU installation ...${NC}"
if "$PYTHON" -c "
import torch
assert torch.cuda.is_available(), 'CUDA/ROCm not available'
print(f'OK: PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}')
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'OK: ONNX Runtime providers: {providers}')
" 2>&1; then
    echo -e "${GREEN}GPU setup complete! .venv_gpu is ready.${NC}"
    echo ""
    echo "To use GPU acceleration, run Aurik normally:"
    echo "  ./run_aurik.sh"
    echo "The runtime selector will auto-detect .venv_gpu."
else
    echo -e "${RED}GPU verification failed. Check logs above.${NC}"
    echo "Falling back to CPU-only (.venv_aurik)."
fi
