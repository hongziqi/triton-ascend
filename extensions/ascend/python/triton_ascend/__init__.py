"""Triton Ascend out-of-tree extension package.

Step 1 (Python puncture): registers the ascend backend via entry_points.
Step 3+: loads native plugin .so via libtriton.passes.plugin.extend_with().
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["register_native_plugin", "PLUGIN_LIBRARY"]


def _plugin_library_path() -> Path | None:
    here = Path(__file__).resolve().parent
    for name in ("libtriton_ascend.so", "libascend.so"):
        candidate = here / name
        if candidate.is_file():
            return candidate
    return None


PLUGIN_LIBRARY = _plugin_library_path()


def register_native_plugin() -> bool:
    """Load the native Ascend plugin into libtriton if the .so is present."""
    if PLUGIN_LIBRARY is None:
        return False
    import triton._C.libtriton as _libtriton

    if not hasattr(_libtriton.passes.plugin, "extend_with"):
        raise RuntimeError(
            "Host triton lacks passes.plugin.extend_with; install triton>=3.7 "
            "built with TRITON_EXT_ENABLED=1"
        )
    _libtriton.passes.plugin.extend_with(str(PLUGIN_LIBRARY))
    if hasattr(_libtriton.ir, "extend_dialects_with"):
        _libtriton.ir.extend_dialects_with(str(PLUGIN_LIBRARY))
    return True


# Best-effort native registration at import time (no-op until step 3 builds the .so).
try:
    register_native_plugin()
except Exception:
    pass
