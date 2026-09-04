#!/bin/sh
set -eu

# A mounted volume makes this expensive first download a one-time operation.
# The server starts only after both selectable models are complete.
if [ "${VECTOR_BOOTSTRAP_MODELS:-1}" = "1" ]; then
  /app/vectorloom-local --bootstrap-models
fi

exec /app/vectorloom-local
