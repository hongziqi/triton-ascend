#!/usr/bin/env bash
# Stage Python-only triton-ascend wheel for entry_points puncture (L1).
set -euo pipefail
EXT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$EXT"

mkdir -p python/triton/backends
ln -sfn "$(realpath backend)" python/triton/backends/ascend

# L2b: language/cann (requires native bindings + triton.extension.buffer)
# mkdir -p python/triton/language/extra
# ln -sfn "$(realpath language/cann)" python/triton/language/extra/cann

rm -rf dist build *.egg-info
pip wheel . -w dist --no-deps
ls -la dist/

echo
echo "Install:"
echo "  pip install triton-3.8.0+gitf893845b-*.whl  # TRITON_EXT host (nightly)"
echo "  pip install dist/triton_ascend-*.whl --force-reinstall --no-deps"
