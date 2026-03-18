"""PyInstaller Runtime-Hook — Thread-Limit-Konfiguration.

Setzt OS-Level-Umgebungsvariablen für OpenMP, OpenBLAS und MKL,
bevor irgendeine ML-Bibliothek (ONNX Runtime, PyTorch, scipy) den
Thread-Pool initialisiert.

Referenz: §2.37 CPU-Aware Pipeline Scheduling (copilot-instructions.md)
OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS = cpu_count()
"""

import os

# Use ALL logical CPUs for maximum throughput on Ryzen 7 (8C/16T).
# For ONNX sessions, intra/inter_op_num_threads is set additionally
# via make_session_options() — these vars are the OS-level safety net.
_n = str(os.cpu_count() or 1)

os.environ.setdefault("OMP_NUM_THREADS", _n)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _n)
os.environ.setdefault("MKL_NUM_THREADS", _n)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _n)

# Disable CUDA probing — Aurik is CPU-only (§8 — no GPU).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("ONNXRUNTIME_PROVIDERS", "CPUExecutionProvider")
