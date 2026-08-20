#!/bin/bash
set -e  # Exit immediately if any command fails

# `temp/` holds the precompiled `decimo.mojoc` that `tools/ensure_decimo.sh`
# builds. Linamo does not compile without it --- the matrix types name
# `decimo.Numeric` --- so this is a hard requirement, not a convenience.
if [[ ! -d temp ]]; then
    echo "decimo is missing. Run 'pixi run decimo' first." >&2
    exit 1
fi
INCLUDES=(-I src -I temp)

# Find and run all test files recursively in the tests directory
find tests -name "test_*.mojo" -type f | sort | while read f; do
    echo "=========================================="
    echo "Running: $f"
    echo "=========================================="
    pixi run mojo run "${INCLUDES[@]}" -D ASSERT=all "$f"
done

echo ""
echo "=========================================="
echo "All tests passed!"
echo "=========================================="