#!/usr/bin/env bash
# Puncture 1: Layer C plugin .so loading
# - Prefer Python (same process as libtriton) — works with static LLVM builds
# - Optional: triton-opt CLI (needs LLVM shared libs for pass registry to unify)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

PLUGIN_SO="${TRITON_PASS_PLUGIN_PATH:-}"
if [[ -z "${PLUGIN_SO}" ]]; then
  for cand in \
    "$ROOT/build/lib.linux-aarch64-cpython-311/triton/plugins/libTritonPluginsTestLib.so" \
    "$(python -c 'import pathlib,triton; print(pathlib.Path(triton.__file__).parent/"plugins"/"libTritonPluginsTestLib.so")' 2>/dev/null || true)"; do
    if [[ -n "$cand" && -f "$cand" ]]; then PLUGIN_SO="$cand"; break; fi
  done
fi
export TRITON_PASS_PLUGIN_PATH="$PLUGIN_SO"

echo "== Puncture 1a: Python loads plugin into libtriton =="
echo "PLUGIN_SO=$PLUGIN_SO"
python - <<'PY'
import os, sys
from triton._C.libtriton import passes
ok = hasattr(passes, "plugin") and hasattr(passes.plugin, "add_plugin")
print("passes.plugin.add_plugin:", ok)
if not ok:
    print("FAIL: plugin not bound; is TRITON_PASS_PLUGIN_PATH set before import?", file=sys.stderr)
    sys.exit(1)
print("PASS: Layer C .so bound as passes.plugin.add_plugin")
PY

TRITON_OPT="${TRITON_OPT:-$ROOT/build/cmake.linux-aarch64-cpython-3.11/bin/triton-opt}"
MLIR="${1:-$ROOT/test/Plugins/test-plugin.mlir}"
echo
echo "== Puncture 1b: triton-opt CLI (optional; needs shared LLVM) =="
if [[ -x "$TRITON_OPT" && -f "$MLIR" ]]; then
  if TRITON_PASS_PLUGIN_PATH="$PLUGIN_SO" "$TRITON_OPT" --help 2>&1 | grep -q 'tritongpu-plugin'; then
    OUT=$(TRITON_PASS_PLUGIN_PATH="$PLUGIN_SO" "$TRITON_OPT" -tritongpu-plugin "$MLIR")
    echo "$OUT" | head -20
    echo "$OUT" | grep -q 'func @foo' && echo "PASS: triton-opt ran plugin" || echo "FAIL: no @foo"
  else
    echo "SKIP: tritongpu-plugin not in CLI help (typical when LLVM is statically linked)."
    echo "      Python path (1a) is the supported puncture for this build."
  fi
else
  echo "SKIP: triton-opt or mlir missing"
fi
