#!/bin/zsh
# Runs Kumru offline on 127.0.0.1:8099. Ctrl+C stops it. Delete the runtime with: rm -rf .local
R="${0:A:h:h}"; L="$R/.local"; B="$(dirname "$(find "$L/bin" -name llama-server -type f | head -1)")"
exec sandbox-exec -D LOCAL="$L" -f "$R/scripts/offline.sb" /usr/bin/env HOME="$L/home" LLAMA_CACHE="$L/home/cache" \
  "$B/llama-server" -m "$L/models/kumru-2b-Q4_K_M.gguf" --host 127.0.0.1 --port 8099 -c 4096 -ngl 99 --jinja "$@"
