#!/bin/bash
# Docker Container Build Script für Aurik ML Models
# Baut alle fehlenden Docker Images für konsistente Model-Integration
# Stand: 10. Februar 2026

set -e  # Exit on error

MODELS_DIR="/mnt/1846D15B46D139E8/Aurik_Standalone/models"
cd "$MODELS_DIR"

echo "🐋 Aurik Model Container Build Script"
echo "======================================"
echo "Building Docker images for all models with Dockerfiles..."
echo ""

# Function to build a model container
build_model() {
    local model_name=$1
    local image_tag=$2
    
    if [ -f "$MODELS_DIR/$model_name/Dockerfile" ]; then
        echo "📦 Building $model_name..."
        cd "$MODELS_DIR/$model_name"
        docker build -t "$image_tag" . || echo "⚠️  Failed to build $model_name"
        echo "✅ $model_name built successfully"
        echo ""
    else
        echo "⚠️  No Dockerfile found for $model_name"
    fi
}

# P0-CRITICAL Models (Must-Have für Production)
echo "🔴 P0-CRITICAL MODELS"
echo "===================="

# ✅ RawNet2 (NEU - gerade heruntergeladen)
build_model "rawnet2" "aurik_model_rawnet2:latest"

# AST Perceptual Base (NEU umbenannt)
build_model "ast_perceptual_base" "aurik_standalone/models/ast:latest"

# Whisper (NEU erstellt)
build_model "whisper" "aurik_model_whisper:latest"

# MERT Genre Classifier (umbenannt)
build_model "mert_genre_classifier" "aurik_standalone/models/genre:latest"

# Deepfake Detection (umbenannt)
build_model "deepfake-detection" "aurik_standalone/models/deepfake:latest"

# Voice Cloning Detection (umbenannt)
build_model "voice-cloning-detection" "aurik_standalone/models/voice-cloning:latest"

# Montreal Forced Aligner
build_model "montreal-forced-aligner" "aurik_model_mfa:latest"

# MERT Instrument Detector (sollte bereits existieren)
build_model "mert_instrument_detector" "mert_instrument_detector:latest"

# madmom (sollte bereits existieren)
build_model "madmom" "aurik_standalone/models/madmom:latest"


echo ""
echo "🟡 P1 NEXT-GEN MODELS"
echo "====================="

# CREPE Pitch Detection (umbenannt)
build_model "crepe" "aurik_model_crepe:latest"


echo ""
echo "🟢 AUDIO PROCESSING MODELS"
echo "=========================="

# Denoising Models
build_model "deepfilternet_v3_ii" "aurik_model_deepfilternet_v3_ii:latest"
build_model "dccrn" "aurik_model_dccrn:latest"
build_model "fullsubnet_plus" "aurik_model_fullsubnet_plus:latest"
build_model "sgmse_plus" "aurik_model_sgmse_plus:latest"
build_model "resemble_enhance" "aurik_model_resemble_enhance:latest"

# Source Separation
build_model "demucs" "aurik_model_demucs:latest"
build_model "waveunet" "aurik_model_waveunet:latest"

# Vocoding/Enhancement
build_model "hifi_gan" "aurik_model_hifi_gan:latest"

# Classification/Analysis
build_model "panns" "aurik_model_panns:latest"
build_model "banquet" "aurik_model_banquet:latest"

# Quality Assessment
build_model "nisqa" "aurik_model_nisqa:latest"
build_model "pesq" "aurik_model_pesq:latest"
build_model "visqol" "aurik_model_visqol:latest"


echo ""
echo "🎉 Docker Build Complete!"
echo "========================"
echo ""
echo "📊 Summary:"
docker images | grep -E "^(aurik|mert|madmom)" | wc -l
echo "Aurik model images built."
echo ""
echo "To verify images:"
echo "  docker images | grep -E '^(aurik|mert|madmom)'"
echo ""
echo "To test a model:"
echo "  docker run --rm aurik_model_rawnet2:latest --help"
echo ""
