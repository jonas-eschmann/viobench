#!/usr/bin/env bash
# Copy the Compose server's data to a local snapshot, then optionally rsync the snapshots elsewhere.
set -euo pipefail
umask 077

usage() {
    cat <<'EOF'
Usage: backup.sh [BACKUP_DIR [RSYNC_DEST]]

Back up uploads, reports and SQLite (including team tokens).
BACKUP_DIR defaults to ~/viobench-backups. RSYNC_DEST is an optional local
directory or user@host:/path/ destination for all completed snapshots.

Uses the active Docker context. Stops app and worker during the local copy,
then restarts whichever services were running, including on copy failure.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if (( $# > 2 )); then
    usage >&2
    exit 2
fi
command -v docker >/dev/null
if (( $# == 2 )); then
    command -v rsync >/dev/null
fi

HERE=$(cd "$(dirname "$0")" && pwd)
dc() {
    docker compose -f "$HERE/compose.yaml" "$@"
}

# Remember the running services so an offline backup does not start the server.
services=$(dc ps --status running --services app worker)
running=()
while IFS= read -r service; do
    if [[ -n $service ]]; then
        running+=("$service")
    fi
done <<< "$services"

backup_root=${1:-$HOME/viobench-backups}
mkdir -p "$backup_root"
backup_root=$(cd "$backup_root" && pwd)
partial=$(mktemp -d "$backup_root/.partial.XXXXXX")
snapshot="$backup_root/$(date +%Y%m%d-%H%M%S)-${partial##*.}"

restart_services() {
    status=$?
    if (( ${#running[@]} )); then
        dc start "${running[@]}" || status=1
    fi
    exit "$status"
}

(
    trap restart_services EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    dc stop app worker
    dc cp app:/srv/server-data/. "$partial/"
)

# Failed copies remain hidden and are excluded from subsequent rsync runs.
mv "$partial" "$snapshot"
printf 'Backup saved to %s\n' "$snapshot"

if (( $# == 2 )); then
    rsync -avh --progress --exclude='/.partial.*' -- "$backup_root/" "$2"
fi
