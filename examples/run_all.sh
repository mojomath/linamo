#!/bin/bash
# Runs every example end to end. The examples exercise the public API only, so
# a failure here means a user-visible break even when the test suite is green.
set -e

for f in examples/*.mojo; do
    echo "=========================================="
    echo "Running: $f"
    echo "=========================================="
    pixi run mojo run -I src "$f"
done

echo ""
echo "=========================================="
echo "All examples ran successfully!"
echo "=========================================="
