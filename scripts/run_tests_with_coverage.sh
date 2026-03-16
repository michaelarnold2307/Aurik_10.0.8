#!/bin/bash
# run_tests_with_coverage.sh
# Führt Tests mit Coverage-Report aus

set -e

echo "=== Aurik Test Suite mit Coverage ==="
echo ""

# Cleanup alte Coverage-Daten
rm -rf .coverage htmlcov/ coverage.xml 2>/dev/null || true

# Führe Tests mit Coverage aus
echo "📊 Starte Tests mit Coverage-Tracking..."
../../.venv_aurik/bin/python -m pytest tests/ \
    --maxfail=5 \
    --disable-warnings \
    --tb=short \
    --cov=. \
    --cov-config=.coveragerc \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-report=xml \
    -n auto \
    || true

echo ""
echo "✅ Tests abgeschlossen!"
echo ""
echo "📈 Coverage-Reports:"
echo "  - HTML: htmlcov/index.html"
echo "  - XML: coverage.xml"
echo ""
echo "Öffne HTML-Report mit:"
echo "  xdg-open htmlcov/index.html"
