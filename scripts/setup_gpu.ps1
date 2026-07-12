<#
setup_gpu.ps1 — Aurik GPU acceleration setup (Windows)
=======================================================
Detects GPU (NVIDIA CUDA / AMD DirectML / none) and installs
appropriate acceleration packages.

Windows strategy:
  NVIDIA GPU → .venv_gpu with CUDA PyTorch + onnxruntime-gpu
  AMD GPU    → DirectML in .venv_aurik (no separate venv needed!)
  No GPU     → CPU-only (.venv_aurik)

Usage:
  powershell -ExecutionPolicy Bypass -File scripts/setup_gpu.ps1
#>
param(
    [switch]$Cuda,
    [switch]$DirectML,
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$VenvGpu = Join-Path $RepoRoot ".venv_gpu"
$VenvCpu = Join-Path $RepoRoot ".venv_aurik"

Write-Host "=== Aurik GPU Setup (Windows) ===" -ForegroundColor Green

# ── Detect GPU ──────────────────────────────────────────────────────────
function Detect-GPU {
    if ($Cuda) { return "cuda" }
    if ($DirectML) { return "directml" }
    if ($Cpu) { return "cpu" }

    try {
        $gpu = Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.Name -match "NVIDIA|AMD|Radeon" } | Select-Object -First 1
        if ($gpu.Name -match "NVIDIA") {
            return "cuda"
        } elseif ($gpu.Name -match "AMD|Radeon") {
            return "directml"
        }
    } catch {
        Write-Host "GPU detection failed: $_" -ForegroundColor Yellow
    }
    return "cpu"
}

$GPUType = Detect-GPU
Write-Host "Detected GPU: $GPUType" -ForegroundColor Yellow

if ($GPUType -eq "cpu") {
    Write-Host "No compatible GPU detected. Aurik runs CPU-only." -ForegroundColor Yellow
    exit 0
}

# ── Check CPU venv exists ───────────────────────────────────────────────
$pythonCpu = Join-Path $VenvCpu "Scripts" "python.exe"
if (-not (Test-Path $pythonCpu)) {
    Write-Host "ERROR: .venv_aurik not found. Run main setup first." -ForegroundColor Red
    exit 1
}

# ── NVIDIA: Create GPU venv ─────────────────────────────────────────────
if ($GPUType -eq "cuda") {
    Write-Host "Creating GPU venv for NVIDIA CUDA ..." -ForegroundColor Green
    & $pythonCpu -m venv --clear $VenvGpu
    $pip = Join-Path $VenvGpu "Scripts" "pip.exe"
    $python = Join-Path $VenvGpu "Scripts" "python.exe"

    & $pip install --upgrade pip setuptools wheel

    Write-Host "Installing PyTorch CUDA + ONNX Runtime GPU ..." -ForegroundColor Green
    & $pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
    & $pip install onnxruntime-gpu

    # Verify
    & $python -c @"
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'OK: PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}')
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'OK: ONNX Runtime providers: {providers}')
"@

    Write-Host "GPU setup complete! .venv_gpu is ready." -ForegroundColor Green
}

# ── AMD: Install DirectML in CPU venv ───────────────────────────────────
if ($GPUType -eq "directml") {
    Write-Host "Installing DirectML for AMD GPU in .venv_aurik ..." -ForegroundColor Green
    $pip = Join-Path $VenvCpu "Scripts" "pip.exe"
    $python = Join-Path $VenvCpu "Scripts" "python.exe"

    & $pip install onnxruntime-directml

    & $python -c @"
import onnxruntime as ort
providers = ort.get_available_providers()
assert 'DmlExecutionProvider' in providers, 'DirectML not available'
print(f'OK: ONNX Runtime providers: {providers}')
"@

    Write-Host "DirectML installed in .venv_aurik. No separate venv needed." -ForegroundColor Green
}

Write-Host ""
Write-Host "Aurik GPU acceleration is ready!" -ForegroundColor Green
Write-Host "The runtime selector will auto-detect GPU support."
