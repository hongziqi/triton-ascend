#!/usr/bin/env bash
# Build a TRITON_EXT-enabled bare triton host from triton-ascend-main (no ascend plugin).
set -euo pipefail

SRC="${TRITON_ASCEND_SRC:-/home/coder/workspace/main/triton-ascend-main}"
# Prefer `triton` env: LLVM cache + prior ascend build. `plugin` env lacks triton after failed installs.
ENV="${TRITON_EXT_ENV:-/home/coder/miniconda/envs/triton}"

if [[ ! -d "$SRC/.git" ]]; then
  echo "TRITON_ASCEND_SRC=$SRC not found" >&2
  exit 1
fi

echo "[L3 host] Installing build deps"
"$ENV/bin/pip" install nanobind pybind11 setuptools wheel cmake ninja 2>&1 | tail -5

echo "[L3 host] Building TRITON_EXT triton from $SRC into $ENV"

# Do NOT set TRITON_PLUGIN_DIRS="" — empty string breaks setup.py copy_externals().
# bin/CMakeLists.txt hardcodes ascend/BiShengIR for triton-opt et al.; without the ascend
# plugin subtree those tools fail to link. For L3 host puncture we still build ascend in-tree;
# out-of-tree ascend is validated separately via triton-ascend wheel + extend_with.
export TRITON_PLUGIN_DIRS="${SRC}/third_party/ascend"
export TRITON_EXT_ENABLED=1
export TRITON_BUILD_PROTON=OFF
export TRITON_WHEEL_NAME=triton

"$ENV/bin/pip" install -e "$SRC" --no-build-isolation --no-deps 2>&1 | tee /tmp/triton_ext_host_build.log

echo "[L3 host] Verify extend_with:"
if ! "$ENV/bin/python" - <<'PY'
import triton._C.libtriton as L
from pathlib import Path
ok = hasattr(L.passes.plugin, "extend_with")
print("extend_with:", ok)
print("libtriton.ascend:", hasattr(L, "ascend"))
inc = Path(__import__("triton").__file__).parent / "include"
print("include:", inc.exists())
raise SystemExit(0 if ok else 1)
PY
then
  echo "[L3 host] WARN: extend_with missing — stale CMake cache? Try:" >&2
  echo "  rm -rf $SRC/build && TRITON_EXT_ENABLED=1 bash $0" >&2
  echo "  Or use triton-ext nightly in plugin env (extend_with=True verified)." >&2
  exit 1
fi
