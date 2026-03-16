#!/usr/bin/env bash
# §4.4+§10.2 Compliance-Pytest Runner
set -e
cd "/media/michael/Software 4TB/Aurik_Standalone"
.venv_aurik/bin/python -m pytest tests/unit \
  --timeout=30 \
  --tb=short \
  -q \
  --disable-warnings \
  --no-header \
  2>&1 | tail -60
