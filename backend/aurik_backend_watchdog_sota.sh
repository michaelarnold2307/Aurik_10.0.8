#!/bin/bash
# Watchdog-Skript für Aurik-Backend Self-Healing (SOTA)
# Prüft regelmäßig den neuen Health-Check-Endpunkt und startet das Backend bei Fehlern neu

HEALTH_URL="http://localhost:8000/health"
SERVICE_NAME="aurik-backend"

RESPONSE=$(curl -sf "$HEALTH_URL")
if [[ $? -ne 0 ]] || ! echo "$RESPONSE" | grep 'ok'; then
    echo "[SOTA-Watchdog] Fehler erkannt – $SERVICE_NAME wird neu gestartet..." | tee -a /var/log/aurik_watchdog.log
    systemctl restart "$SERVICE_NAME"
    logger -t aurik-watchdog "Aurik-Backend wurde durch Watchdog neu gestartet."
else
    echo "[SOTA-Watchdog] Aurik-Backend läuft fehlerfrei." | tee -a /var/log/aurik_watchdog.log
fi
