#!/bin/bash
# SOTA-Build-&-Test-Skript für Aurik (ohne CI-Server)
set -e

# 1. Linting
if command -v flake8 &> /dev/null; then
  echo "[SOTA] Linting mit flake8..."
  flake8 .
else
  echo "[SOTA] flake8 nicht gefunden, überspringe Linting."
fi

# 2. Unit- und Integrationstests
if command -v pytest &> /dev/null; then
  echo "[SOTA] Starte Tests mit pytest..."
  pytest --maxfail=1 --disable-warnings --tb=short
else
  echo "[SOTA] pytest nicht gefunden, überspringe Tests."
fi

# 3. Docker-Builds
if command -v docker &> /dev/null; then
  echo "[SOTA] Baue alle Docker-Images..."
  docker compose build
else
  echo "[SOTA] Docker nicht gefunden, überspringe Docker-Builds."
fi

# 4. Healthchecks (optional)
echo "[SOTA] Starte Healthchecks..."
docker compose up -d
sleep 10
HEALTH_URL="http://localhost:8000/api/health"
if curl -sf "$HEALTH_URL" | grep '"status": "ok"'; then
  echo "[SOTA] Healthcheck erfolgreich."
else
  echo "[SOTA] Healthcheck FEHLGESCHLAGEN!"
  exit 1
fi

echo "[SOTA] Build-&-Test-Skript erfolgreich abgeschlossen."
