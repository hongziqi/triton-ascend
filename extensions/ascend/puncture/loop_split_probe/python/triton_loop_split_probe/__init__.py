"""LoopSplit probe: validates extend_with on the host triton."""

from pathlib import Path

import triton._C.libtriton as _libtriton

PLUGIN_DIR = Path(__file__).resolve().parent
PLUGIN_LIBRARY = PLUGIN_DIR / "libloop_split_probe.so"

if not hasattr(_libtriton.passes.plugin, "extend_with"):
    raise RuntimeError(
        "Host triton lacks passes.plugin.extend_with — build/install a TRITON_EXT "
        "enabled triton host first (see scripts/build_triton_ext_host.sh)."
    )

_libtriton.passes.plugin.extend_with(str(PLUGIN_LIBRARY))
