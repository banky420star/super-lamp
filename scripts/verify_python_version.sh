#!/usr/bin/env bash
# verify_python_version.sh — Check that the active Python matches .python-version
# Exit 0 on match, exit 1 on mismatch with a clear error message.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PINNED_FILE="$PROJECT_ROOT/.python-version"

if [ ! -f "$PINNED_FILE" ]; then
    echo "ERROR: .python-version not found at $PINNED_FILE"
    echo "Create it with:  echo '3.14.5' > .python-version"
    exit 1
fi

PINNED=$(tr -d '[:space:]' < "$PINNED_FILE")

# Get the active Python major.minor.patch (strip any trailing '+' or extra info)
ACTIVE=$(python --version 2>&1 | sed -E 's/^Python //; s/[+].*$//; s/[[:space:]].*$//')

if [ "$ACTIVE" != "$PINNED" ]; then
    echo "=========================================="
    echo "  PYTHON VERSION MISMATCH"
    echo "=========================================="
    echo "  Pinned  (.python-version) : $PINNED"
    echo "  Active  (python --version): $ACTIVE"
    echo "=========================================="
    echo ""
    echo "Fix: install the pinned version and re-activate your environment."
    echo "  pyenv install $PINNED"
    echo "  pyenv local $PINNED"
    echo ""
    echo "Or, if the pinned version is wrong, update .python-version:"
    echo "  echo '$ACTIVE' > .python-version"
    exit 1
fi

echo "✓ Python version matches .python-version: $ACTIVE"
