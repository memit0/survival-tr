#!/bin/zsh
# Runs a model offline on 127.0.0.1. Ctrl+C stops it. Delete the runtime with: rm -rf .local
# Defaults to Gemma-2-2B on port 8099. Embeddings: MODEL=embeddinggemma-300m-qat-Q8_0.gguf PORT=8098 ./scripts/start_model.sh --embedding -c 2048
R="${0:A:h:h}"; L="$R/.local"; B="$(dirname "$(find "$L/bin" -name llama-server -type f | head -1)")"
exec sandbox-exec -D LOCAL="$L" -f "$R/scripts/offline.sb" /usr/bin/env HOME="$L/home" LLAMA_CACHE="$L/home/cache" \
  "$B/llama-server" -m "$L/models/${MODEL:-gemma-2-2b-it-Q4_K_M.gguf}" --host 127.0.0.1 --port "${PORT:-8099}" -c 4096 -ngl 99 --jinja "$@"
