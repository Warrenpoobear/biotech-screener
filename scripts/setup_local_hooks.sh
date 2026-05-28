#!/usr/bin/env bash
# setup_local_hooks.sh — Install local git hooks for this repo.
#
# These hooks run on your local machine only (not in CI).
# They require a local codegraph index (.codegraph/codegraph.db).
#
# Usage:
#   chmod +x scripts/setup_local_hooks.sh
#   ./scripts/setup_local_hooks.sh
#
# What is installed:
#   pre-commit:  runs 'codegraph affected' on staged files and prints
#                the test files that cover those changes, so you know
#                exactly which tests to run before pushing.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

# ---------------------------------------------------------------------------
# Guard: codegraph must be available
# ---------------------------------------------------------------------------
if ! command -v codegraph &>/dev/null; then
    echo "ERROR: codegraph not found in PATH."
    echo "Install it first: npm install -g @colbymchenry/codegraph@0.9.6"
    exit 1
fi

# Guard: must be run from repo root (or git repo)
if [ ! -d "$HOOKS_DIR" ]; then
    echo "ERROR: .git/hooks not found. Run this from the repo root."
    exit 1
fi

# ---------------------------------------------------------------------------
# pre-commit hook: codegraph affected
# ---------------------------------------------------------------------------
PRECOMMIT="$HOOKS_DIR/pre-commit"

cat > "$PRECOMMIT" <<'HOOK'
#!/usr/bin/env bash
# Local pre-commit hook: show tests affected by staged changes.
# Installed by scripts/setup_local_hooks.sh — do not edit manually.

set -uo pipefail

# Only run if codegraph index exists
if [ ! -f "$(git rev-parse --show-toplevel)/.codegraph/codegraph.db" ]; then
    exit 0
fi

STAGED=$(git diff --cached --name-only --diff-filter=ACM 2>/dev/null | grep '\.py$' || true)

if [ -z "$STAGED" ]; then
    exit 0
fi

AFFECTED=$(echo "$STAGED" | codegraph affected --stdin --quiet 2>/dev/null || true)

if [ -n "$AFFECTED" ]; then
    echo ""
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│  codegraph: tests affected by this commit                   │"
    echo "└─────────────────────────────────────────────────────────────┘"
    echo "$AFFECTED" | sed 's/^/  /'
    echo ""
    echo "  Run before pushing:"
    echo "    pytest $AFFECTED -q"
    echo ""
fi

# Hook is advisory only — does not block the commit.
exit 0
HOOK

chmod +x "$PRECOMMIT"
echo "✓ pre-commit hook installed at $PRECOMMIT"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "Local hooks installed. Summary:"
echo "  pre-commit  →  codegraph affected (advisory, non-blocking)"
echo ""
echo "To uninstall:"
echo "  rm $PRECOMMIT"
