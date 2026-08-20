#!/bin/bash
set -e  # Exit immediately if any command fails

# `temp/` holds the precompiled `decimo.mojoc` that `tools/ensure_decimo.sh`
# builds. Only tests/decimo needs it, but passing the include path
# unconditionally costs nothing and keeps one command for the whole suite.
INCLUDES=(-I src)
[[ -d temp ]] && INCLUDES+=(-I temp)

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