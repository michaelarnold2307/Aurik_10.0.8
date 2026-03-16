#!/bin/bash
# SOTA-Test-Wrapper für Aurik
# Führt alle Tests mit korrektem PYTHONPATH aus dem Projekt-Root aus

export PYTHONPATH="aurik6"
../../.venv_aurik/bin/python -m pytest testing --maxfail=1 --disable-warnings --tb=short
