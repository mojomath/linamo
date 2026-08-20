#!/bin/bash
# Runs every example end to end. The examples exercise the public API only, so
# a failure here means a user-visible break even when the test suite is green.
set -e

# `temp/` holds the precompiled decimo package (tools/ensure_decimo.sh).
# Linamo does not compile without it.
if [[ ! -d temp ]]; then
    echo "decimo is missing. Run 'pixi run decimo' first." >&2
    exit 1
fi
INCLUDES=(-I src -I temp)

for f in examples/*.mojo; do
    echo "=========================================="
    echo "Running: $f"
    echo "=========================================="
    pixi run mojo run "${INCLUDES[@]}" "$f"
done

echo ""
echo "=========================================="
echo "All examples ran successfully!"
echo "=========================================="
