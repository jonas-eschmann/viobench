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
Also works after compose down if the server-data volume still exists;
this requires python3 and the locally built app image on the Docker host.
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

cleanup() {
    local status=$?
    if [[ -n $helper ]]; then
        docker rm "$helper" >/dev/null || status=1
    fi
    if (( ${#running[@]} )); then
        dc start "${running[@]}" || status=1
    fi
    exit "$status"
}

(
    helper=""
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    dc stop app worker
    source_container=$(dc ps --all --quiet app)
    if [[ -z $source_container ]]; then
        # compose down removes containers but normally retains named volumes.
        # Resolve the actual name, including COMPOSE_PROJECT_NAME overrides.
        command -v python3 >/dev/null
        volume=$(dc config --format json | python3 -c '
import json, sys
print(json.load(sys.stdin)["volumes"]["server-data"]["name"])
')
        if ! docker volume inspect "$volume" >/dev/null; then
            printf 'Cannot back up: existing data volume %s is unavailable.\n' "$volume" >&2
            exit 1
        fi
        image=$(dc config --images app)
        docker image inspect "$image" >/dev/null
        # Never start this container or populate/change the source volume.
        helper=$(docker create --network none \
            --mount "type=volume,src=$volume,dst=/srv/server-data,readonly,volume-nocopy" \
            --entrypoint /bin/true "$image")
        source_container=$helper
    fi
    docker cp "$source_container:/srv/server-data/." "$partial/"
)

# Failed copies remain hidden and are excluded from subsequent rsync runs.
mv "$partial" "$snapshot"
printf 'Backup saved to %s\n' "$snapshot"

if (( $# == 2 )); then
    rsync -avh --progress --exclude='/.partial.*' -- "$backup_root/" "$2"
fi
