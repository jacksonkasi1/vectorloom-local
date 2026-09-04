#!/bin/sh
set -eu

# A mounted volume makes this expensive first download a one-time operation.
# Blocking bootstrap is useful for batch jobs; web deployments instead start the
# UI immediately and download in the background.
if [ "${VECTOR_BOOTSTRAP_MODELS:-0}" = "1" ]; then
  /app/vectorloom-local --bootstrap-models
fi

export VECTOR_AUTO_DOWNLOAD="${VECTOR_AUTO_DOWNLOAD:-all}"
exec /app/vectorloom-local
