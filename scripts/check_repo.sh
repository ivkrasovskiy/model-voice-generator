#!/usr/bin/env bash
# Repo health check — run before declaring any stage done.
# Exit 1 if any check fails; prints what failed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

FAIL=0

echo "=== ruff ==="
if uv run ruff check scripts/ accent_coach/; then
    echo "  ✓ ruff clean"
else
    echo "  ✗ ruff errors"
    FAIL=1
fi

echo ""
echo "=== line-count guard (≤400 code lines per file) ==="
while IFS= read -r -d '' f; do
    rel="${f#$REPO_ROOT/}"
    lines=$(grep -c . "$f" || true)
    if [ "$lines" -gt 400 ]; then
        echo "  ✗ $rel has $lines lines (limit 400)"
        FAIL=1
    fi
done < <(find scripts accent_coach -name "*.py" -print0)
echo "  ✓ line-count check done"

echo ""
echo "=== pytest ==="
if uv run pytest tests/ -q; then
    echo "  ✓ tests pass"
else
    echo "  ✗ tests failed"
    FAIL=1
fi

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "ALL CHECKS PASSED"
else
    echo "CHECKS FAILED — see above"
    exit 1
fi
