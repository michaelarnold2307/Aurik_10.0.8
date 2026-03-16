#!/bin/bash
# Baut alle Model-Container gemäß models/model_registry.json
set -e
REGISTRY="models/model_registry.json"

if ! command -v jq &> /dev/null; then
  echo "Fehler: jq ist erforderlich (sudo apt install jq)" >&2
  exit 1
fi

COUNT=$(jq '.models | length' "$REGISTRY")
echo "Baue $COUNT Model-Container gemäß $REGISTRY..."

for i in $(seq 0 $((COUNT-1))); do
  NAME=$(jq -r ".models[$i].name" "$REGISTRY")
  DOCKERFILE=$(jq -r ".models[$i].dockerfile" "$REGISTRY")
  CONTEXT=$(dirname "$DOCKERFILE")
  TAG="aurik_model_${NAME}:latest"
  echo "\n=== Baue $NAME ($DOCKERFILE) als $TAG ==="
  docker build -f "$DOCKERFILE" -t "$TAG" "$CONTEXT"
done

echo "Alle Model-Container wurden gebaut!"
