#!/usr/bin/env bash
# Seed a fresh running server: seed.sh [BASELINES_DIR], default ./baselines.
# Run from the directory containing server-data, with the server and worker already running.
# Fixed tokens are for local testing only.
set -euo pipefail

baseline_dir=${1:-./baselines}
server_url=${SERVER_URL:-http://localhost:8000}
server_url=${server_url%/}
shopt -s nullglob
files=("$baseline_dir"/*.wasm)
if (( ${#files[@]} == 0 )); then
    printf 'No .wasm files in %s\n' "$baseline_dir" >&2
    exit 1
fi
curl --fail --silent --show-error --max-time 10 "$server_url/" >/dev/null

python -m viobench.tools.server.teams add --type participant --token test-participant participant
baseline=$(python -m viobench.tools.server.teams add --type baseline --unlimited --token test-baseline baseline)
printf '%s\n' "$baseline"
python -m viobench.tools.server.teams add --type admin --token test-admin admin
baseline_token=${baseline%% *}

for file in "${files[@]}"; do
    name=${file##*/}
    printf 'Uploading %s\n' "$name"
    curl --fail-with-body --silent --show-error -H "Authorization: Bearer $baseline_token" \
        -F "file=@$file" --form-string "label=${name%.wasm}" "$server_url/submissions"
    printf '\n'
done
