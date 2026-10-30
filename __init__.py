# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Load the Ascend plugin into the host Triton context (community API).

``libtriton_ascend.so`` exports ``tritonGetPluginInfo``. It does not own a
private ``MLIRContext``. ``libtriton`` must already be imported, and the host
wheel must be built with ``TRITON_EXT_ENABLED=1`` so MLIR symbols resolve when
the plugin is dlopen'd.

Community load path (no HostExtend / ``extend_module_with``)::

    passes.plugin.extend_with(so)   # PassInfo → passes.plugin.add_*
    ir.extend_dialects_with(so)     # DialectInfo queued for load_dialects
    ir.builder.extend_with(so)      # OpInfo → builder.create_<name>
"""

from __future__ import annotations

import importlib
import os
import sys
import warnings
from pathlib import Path

_ASCEND_PLUGIN_LOADED = False
ir = None  # community hosts use stock ir.builder; kept for compat aliases


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
    # Prefer the untagged build product. Do NOT pick by size: a stale
    # libtriton_ascend.cpython-*.so can be larger and still be the wrong ABI.
    plain = c_dir / "libtriton_ascend.so"
    if plain.is_file():
        return plain
    matches = [p for p in c_dir.glob("libtriton_ascend*.so") if p.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def register_ascend_native() -> None:
    """Load the Ascend plugin ``.so`` into the host (idempotent).

    Community plugin entry points:

    * ``passes.plugin.extend_with(so)`` — PassInfo → ``passes.plugin.add_*``
    * ``ir.extend_dialects_with(so)`` — DialectInfo for later ``load_dialects``
    * ``ir.builder.extend_with(so)`` — OpInfo → ``builder.create_<name>``

    Deliberately does **not** call ``ir.extend_module_with`` (HostExtend).
    """
    global _ASCEND_PLUGIN_LOADED, ir
    if _ASCEND_PLUGIN_LOADED:
        return
    libtriton = importlib.import_module("triton._C.libtriton")
    host_ir = libtriton.ir
    # Avoid double-dlopen of another copy of libtriton_ascend.so (pytest may
    # import this tree as a parent package in addition to the installed wheel).
    if getattr(libtriton, "_ascend_plugin_loaded", False):
        ir = host_ir
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

    if hasattr(host_ir, "extend_dialects_with"):
        host_ir.extend_dialects_with(so_s)

    if not hasattr(host_ir, "builder") or not hasattr(host_ir.builder, "extend_with"):
        raise RuntimeError(
            "host missing ir.builder.extend_with; need a Triton wheel built "
            "with TRITON_EXT_ENABLED=1 (community OpInfo path)"
        )
    host_ir.builder.extend_with(so_s)

    # Point triton_ascend.ir at stock host ir for any legacy import sites that
    # only need shared types / parse_attr / load_dialects.
    ir = host_ir
    sys.modules[__name__ + ".ir"] = host_ir
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

# Dual-package: restore former invasive Triton patches via monkey-patch.
# Safe to call before/without NPUOptions; install() is idempotent.
try:
    from .backend.triton_hotpatch import install as _install_triton_hotpatch

    _install_triton_hotpatch()
except Exception as _hotpatch_exc:
    warnings.warn(
        f"triton_ascend hotpatch install failed ({_hotpatch_exc!r})",
        ImportWarning,
        stacklevel=1,
    )


__all__ = ["register_ascend_native", "plugin_library", "ir"]
