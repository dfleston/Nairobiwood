#!/usr/bin/env bash
set -euo pipefail

# Registry target: 'ghcr' (default) or 'dockerhub'
TARGET="${1:-ghcr}"
IMAGE_NAME="nairobiwood-wan22"
TAG="${2:-latest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "$TARGET" == "ghcr" ]]; then
  GH_USER="${GH_USER:-$(gh api user -q .login 2>/dev/null || echo 'dfleston')}"
  FULL_IMAGE="ghcr.io/${GH_USER}/${IMAGE_NAME}:${TAG}"
  echo "Target Registry: GitHub Container Registry (${FULL_IMAGE})"
  echo "Logging into ghcr.io..."
  gh auth token | docker login ghcr.io -u "$GH_USER" --password-stdin
elif [[ "$TARGET" == "dockerhub" ]]; then
  DOCKER_USER="${DOCKER_USER:-$(docker info 2>/dev/null | grep -i 'Username:' | awk '{print $2}' || echo '')}"
  if [[ -z "$DOCKER_USER" ]]; then
    read -rp "Enter Docker Hub username: " DOCKER_USER
  fi
  FULL_IMAGE="${DOCKER_USER}/${IMAGE_NAME}:${TAG}"
  echo "Target Registry: Docker Hub (${FULL_IMAGE})"
else
  echo "Unknown target: $TARGET. Choose 'ghcr' or 'dockerhub'."
  exit 1
fi

echo "Building lean image ${FULL_IMAGE} (linux/amd64)..."
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.wan22 \
  -t "$FULL_IMAGE" \
  --push \
  .

echo "Successfully built and pushed: ${FULL_IMAGE}"
