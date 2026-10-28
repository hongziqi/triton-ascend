# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Load the Ascend plugin into the host Triton context.

``libtriton_ascend.so`` exports ``tritonGetPluginInfo``. It does not own a
private ``MLIRContext``. ``libtriton`` must already be imported, and the host
wheel must be built with ``TRITON_EXT_ENABLED=1`` so MLIR / shared nanobind
symbols resolve when the plugin is dlopen'd.

API v3 hosts expose ``ir.extend_module_with``, which runs the plugin's
``HostExtendCallback`` and registers the full ``ascend_ir.cc`` surface onto
``triton._C.libtriton.ir._ascend`` (re-exported here as ``triton_ascend.ir``).
"""

from __future__ import annotations

import importlib
import os
import sys
import warnings
from pathlib import Path

_ASCEND_PLUGIN_LOADED = False
ir = None  # set by register_ascend_native()


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def plugin_library() -> Path | None:
    env = os.environ.get("TRITON_ASCEND_EXTENSION") or os.environ.get(
        "TRITON_ASCEND_PASS_PLUGIN"
    )
    if env:
        path = Path(env)
        return path if path.is_file() else None
    c_dir = _package_dir() / "_C"
    if not c_dir.is_dir():
        return None
    # Prefer the largest matching .so so a stale package_data copy of
    # libtriton_ascend.so cannot shadow the setuptools-tagged build output.
    matches = [p for p in c_dir.glob("libtriton_ascend*.so") if p.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_size)


def register_ascend_native() -> None:
    """Load the single Ascend plugin ``.so`` into the host (idempotent).

    Community plugin entry points used here:

    * ``passes.plugin.extend_with(so)`` — PassInfo → ``passes.plugin.add_*``
    * ``ir.extend_module_with(so)`` — HostExtendCallback → ``ir._ascend``
      (full ``ascend_ir.cc`` surface, including ``AscendNPUIROpBuilder``)

    Not used: ``ir.builder.extend_with`` (stock OpInfo). Ascend leaves the ops
    table empty and registers a *subclass* of ``TritonOpBuilder`` on
    ``ir._ascend`` via HostExtend (no OpBuildArgs / OpInfo layout change).
    """
    global _ASCEND_PLUGIN_LOADED, ir
    if _ASCEND_PLUGIN_LOADED:
        return
    libtriton = importlib.import_module("triton._C.libtriton")
    host_ir = libtriton.ir
    # pytest imports this tree's __init__.py as a parent package in addition to
    # the installed triton_ascend wheel. A second dlopen of another copy of
    # libtriton_ascend.so interposes preemptible globals (shmemIntp) and
    # double-frees them at process exit.
    if getattr(libtriton, "_ascend_plugin_loaded", False):
        ascend_mod = getattr(host_ir, "_ascend", None)
        if ascend_mod is not None:
            ir = ascend_mod
            sys.modules[__name__ + ".ir"] = ascend_mod
        _ASCEND_PLUGIN_LOADED = True
        return
    so = plugin_library()
    if so is None:
        raise FileNotFoundError(
            "libtriton_ascend.so not found under triton_ascend/_C/. "
            "Build standalone/ against a TRITON_EXT host."
        )
    so_s = str(so)

    libtriton.passes.plugin.extend_with(so_s)

    if not hasattr(host_ir, "extend_module_with"):
        raise RuntimeError(
            "host missing ir.extend_module_with; apply Track A patch 0003 "
            "(HostExtendCallback + shared libnanobind)"
        )
    host_ir.extend_module_with(so_s)

    ascend_mod = getattr(host_ir, "_ascend", None)
    if ascend_mod is None:
        raise RuntimeError(
            "plugin loaded but ir._ascend is missing; plugin must export "
            "PluginInfo.extend → ascendExtendHost → init_ascend_ir"
        )
    ir = ascend_mod
    sys.modules[__name__ + ".ir"] = ascend_mod
    libtriton._ascend_plugin_loaded = True
    _ASCEND_PLUGIN_LOADED = True


def _ensure_cann_language() -> None:
    """Expose ``triton.language.extra.cann`` without copying it into the wheel."""
    if "triton.language.extra.cann" in sys.modules:
        return
    src = _package_dir() / "language" / "cann"
    if not src.is_dir():
        return
    try:
        extra = importlib.import_module("triton.language.extra")
        spec = importlib.util.spec_from_file_location(
            "triton.language.extra.cann",
            src / "__init__.py",
            submodule_search_locations=[str(src)],
        )
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        sys.modules["triton.language.extra.cann"] = mod
        extra.cann = mod  # type: ignore[attr-defined]
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop("triton.language.extra.cann", None)


try:
    _ensure_cann_language()
    register_ascend_native()
except Exception as _ascend_exc:
    warnings.warn(
        f"triton_ascend plugin registration failed ({_ascend_exc!r}); "
        "build standalone/ against a TRITON_EXT_ENABLED host.",
        ImportWarning,
        stacklevel=1,
    )


__all__ = ["register_ascend_native", "plugin_library", "ir"]
