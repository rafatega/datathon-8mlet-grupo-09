#!/usr/bin/env bash
# Build da imagem, sobe o container e valida /health e /recommend (inclusive PORT via env).
set -euo pipefail

IMAGE=datathon-bandit-api
HOST_PORT=${HOST_PORT:-8000}

docker build -t "$IMAGE" .

check() {
  local port=$1
  for _ in $(seq 1 30); do
    if curl -sf "http://localhost:${port}/health" >/dev/null; then break; fi
    sleep 1
  done
  curl -sf "http://localhost:${port}/health"; echo
  curl -sf -X POST "http://localhost:${port}/recommend" \
    -H "Content-Type: application/json" \
    -d '{"age": 25, "poutcome": "nonexistent"}'; echo
}

CID=$(docker run -d -p "${HOST_PORT}:8000" "$IMAGE")
trap 'docker rm -f "$CID" >/dev/null' EXIT
check "$HOST_PORT"
docker rm -f "$CID" >/dev/null

CID=$(docker run -d -e PORT=9000 -p "${HOST_PORT}:9000" "$IMAGE")
check "$HOST_PORT"

echo "SMOKE TEST OK"
