#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec uv --directory "${REPO_ROOT}/mcp-server" run nairobiwood-cli up "$@"
