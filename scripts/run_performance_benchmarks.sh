#!/bin/bash
# run_performance_benchmarks.sh
# Führt Performance-Benchmarks aus und generiert Reports

set -e

echo "=== Aurik Performance Benchmarks ==="
echo ""

# Erstelle Benchmark-Output-Verzeichnis
mkdir -p benchmarks/results

echo "🚀 Starte DSP Performance Benchmarks..."
../../.venv_aurik/bin/python -m pytest benchmarks/test_dsp_performance.py \
    --benchmark-only \
    --benchmark-autosave \
    --benchmark-save-data \
    --benchmark-json=benchmarks/results/dsp_benchmarks.json \
    --benchmark-histogram=benchmarks/results/dsp_histogram \
    -v

echo ""
echo "🚀 Starte Pipeline Performance Benchmarks..."
../../.venv_aurik/bin/python -m pytest benchmarks/test_pipeline_performance.py \
    --benchmark-only \
    --benchmark-autosave \
    --benchmark-save-data \
    --benchmark-json=benchmarks/results/pipeline_benchmarks.json \
    --benchmark-histogram=benchmarks/results/pipeline_histogram \
    -v

echo ""
echo "✅ Benchmarks abgeschlossen!"
echo ""
echo "📊 Ergebnisse:"
echo "  - DSP Benchmarks: benchmarks/results/dsp_benchmarks.json"
echo "  - Pipeline Benchmarks: benchmarks/results/pipeline_benchmarks.json"
echo "  - Histogramme: benchmarks/results/*_histogram.svg"
echo ""
echo "Vergleiche Benchmarks mit:"
echo "  pytest-benchmark compare benchmarks/results/*.json"
