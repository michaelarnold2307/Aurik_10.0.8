#!/bin/bash
# Docker Build Script für ALLE 28 ML-Modelle - AURIK 6.0
# =========================================================
# Stand: 2026-02-05
# Baut alle Docker-Images für die ML-Pipeline

set -e  # Exit on error

MODELS_DIR="/mnt/1846D15B46D139E8/Aurik_Standalone/models"
LOG_FILE="/mnt/1846D15B46D139E8/Aurik_Standalone/docker_build.log"

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "AURIK 8.0 - Docker Image Builder"
echo "Building ALL 28 ML Model Images"
echo "=========================================="
echo ""

# Log-Datei initialisieren
echo "Build started at $(date)" > "$LOG_FILE"

# Zähler
TOTAL=28
SUCCESS=0
FAILED=0

# Funktion zum Bauen eines Images
build_image() {
    local model_dir=$1
    local dockerfile=$2
    local image_name=$3

    echo -e "${YELLOW}[Building]${NC} $image_name..."

    cd "$MODELS_DIR/$model_dir"

    if docker build -f "$dockerfile" -t "$image_name:latest" . >> "$LOG_FILE" 2>&1; then
        echo -e "${GREEN}[✓ SUCCESS]${NC} $image_name"
        ((SUCCESS++))
    else
        echo -e "${RED}[✗ FAILED]${NC} $image_name (check $LOG_FILE)"
        ((FAILED++))
    fi

    cd "$MODELS_DIR"
}

# Processing Plugins (16)
echo ""
echo "=== PROCESSING PLUGINS (16) ==="
echo ""

build_image "deepfilternet_v3_ii" "Dockerfile.deepfilternet_v3_ii" "deepfilternet_v3_ii"
build_image "resemble_enhance" "Dockerfile.resemble_enhance" "resemble_enhance"
build_image "demucs" "Dockerfile.demucs" "demucs"
build_image "mdx23c" "Dockerfile.mdx23c" "mdx23c"
build_image "sgmse_plus" "Dockerfile.sgmse_plus" "sgmse_plus"
build_image "fullsubnet_plus" "Dockerfile.fullsubnet_plus" "fullsubnet_plus"
build_image "dccrn" "Dockerfile.dccrn" "dccrn"
build_image "uvr_mdx_net" "Dockerfile.uvr_mdx_net" "uvr_mdx_net"
build_image "banquet" "Dockerfile.banquet" "banquet"
build_image "hifi_gan" "Dockerfile.hifi_gan" "hifi_gan"
build_image "conv-tasnet" "Dockerfile.conv-tasnet" "conv_tasnet"
build_image "diffwave" "Dockerfile.diffwave" "diffwave"
build_image "waveunet" "Dockerfile.waveunet" "waveunet"
build_image "crepe" "Dockerfile.crepe" "crepe"
echo -e "${YELLOW}[SKIP]${NC} sota_universal_enhancer (meta-plugin, no Docker)"

# Metriken Plugins (4)
echo ""
echo "=== METRIKEN PLUGINS (4) ==="
echo ""

build_image "dnsmos" "Dockerfile.dnsmos" "dnsmos"
build_image "nisqa" "Dockerfile.nisqa" "nisqa"
build_image "pesq" "Dockerfile.pesq" "pesq"
build_image "visqol" "Dockerfile.visqol" "visqol"

# Neue Plugins (8)
echo ""
echo "=== NEUE PLUGINS (7) ==="
echo ""

build_image "audioldm2" "Dockerfile.audioldm2" "audioldm2"
build_image "audiosr" "Dockerfile.audiosr" "audiosr"
build_image "cdpam" "Dockerfile.cdpam" "cdpam"
build_image "gacela" "Dockerfile.gacela" "gacela"
build_image "matchering2.0" "Dockerfile.matchering2.0" "matchering"
build_image "silero" "Dockerfile.silero" "silero"
build_image "vampnet" "Dockerfile.vampnet" "vampnet"

# Zusammenfassung
echo ""
echo "=========================================="
echo "BUILD SUMMARY"
echo "=========================================="
echo -e "${GREEN}SUCCESS:${NC} $SUCCESS/$TOTAL images"
echo -e "${RED}FAILED:${NC} $FAILED/$TOTAL images"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All Docker images built successfully!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some builds failed. Check $LOG_FILE for details.${NC}"
    exit 1
fi
