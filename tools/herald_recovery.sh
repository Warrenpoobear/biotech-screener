#!/usr/bin/env bash
# herald_recovery.sh — operator wrapper for F-2026-005 Herald pipeline recovery.
#
# Usage:
#   ./tools/herald_recovery.sh                    # today, minimal steps from health
#   ./tools/herald_recovery.sh 2026-06-24         # specific date
#   ./tools/herald_recovery.sh --full --digest    # full pipeline + evening digest
#   ./tools/herald_recovery.sh --dry-run          # print planned commands only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
AS_OF=""
EXTRA=()

for arg in "$@"; do
  case "$arg" in
    --dry-run|--full|--digest)
      EXTRA+=("$arg")
      ;;
    *)
      if [[ -z "$AS_OF" && "$arg" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        AS_OF="$arg"
      else
        EXTRA+=("$arg")
      fi
      ;;
  esac
done

AS_OF="${AS_OF:-$(TZ=America/Detroit date +%Y-%m-%d)}"

echo "[$(date -Iseconds)] herald_recovery: as_of=${AS_OF} extra=${EXTRA[*]:-}"

exec "$PYTHON" tools/herald_recovery.py --as-of-date "$AS_OF" "${EXTRA[@]}"
