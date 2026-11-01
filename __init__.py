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


def _drop_none_module_sentinels(prefix: str) -> None:
    """Clear failed-import ``None`` placeholders under ``prefix``."""
    for key in list(sys.modules):
        if key == prefix or key.startswith(prefix + "."):
            if sys.modules.get(key) is None:
                sys.modules.pop(key, None)


def _seed_backends_ascend_alias() -> None:
    """Publish ``triton.backends.ascend`` before cann.libdevice imports utils.

    Backend discovery imports ``triton_ascend.backend.*`` after this package
    ``__init__`` returns; libdevice still needs the legacy alias during init.
    Importing ``.backend`` here is safe: this module is already in sys.modules.
    """
    if "triton.backends.ascend" in sys.modules:
        return
    backend = importlib.import_module(__name__ + ".backend")
    sys.modules.setdefault("triton.backends.ascend", backend)


def _ensure_cann_language() -> None:
    """Expose ``triton.language.extra.cann`` without copying it into the wheel.

    ``libdevice`` and ``extension`` are loaded independently inside cann's
    ``__init__``; a missing HostExtend affine API must not remove cann itself.
    """
    _drop_none_module_sentinels("triton.language.extra.cann")

    existing = sys.modules.get("triton.language.extra.cann")
    if existing is not None:
        try:
            extra = importlib.import_module("triton.language.extra")
            if getattr(extra, "cann", None) is None:
                extra.cann = existing  # type: ignore[attr-defined]
        except Exception:
            pass
        return

    src = _package_dir() / "language" / "cann"
    if not src.is_dir():
        return
    try:
        _seed_backends_ascend_alias()
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
    except Exception as exc:
        sys.modules.pop("triton.language.extra.cann", None)
        _drop_none_module_sentinels("triton.language.extra.cann")
        try:
            extra = importlib.import_module("triton.language.extra")
            if getattr(extra, "cann", None) is not None:
                delattr(extra, "cann")
        except Exception:
            pass
        warnings.warn(
            f"triton.language.extra.cann injection failed ({exc!r})",
            ImportWarning,
            stacklevel=2,
        )


def _triton_jit_ready() -> bool:
    """False while ``import triton`` (or ``runtime.jit``) is still initializing."""
    triton_mod = sys.modules.get("triton")
    # ``must_use_result`` is assigned near the end of community triton/__init__.py.
    if triton_mod is None or not hasattr(triton_mod, "must_use_result"):
        return False
    jit = sys.modules.get("triton.runtime.jit")
    return jit is not None and hasattr(jit, "JITCallable")


def _install_hotpatch_safe() -> None:
    """Install hotpatch only when community ``triton.runtime.jit`` is complete."""
    if not _triton_jit_ready():
        return
    try:
        from .backend.triton_hotpatch import install as _install_triton_hotpatch

        _install_triton_hotpatch()
    except Exception as exc:
        warnings.warn(
            f"triton_ascend hotpatch install failed ({exc!r})",
            ImportWarning,
            stacklevel=2,
        )


class _DeferredAscendSetupFinder:
    """Finish Ascend setup after ``import triton`` completes.

    Backend discovery imports this package *during* ``import triton``. Eagerly
    executing cann/libdevice or hotpatch then circular-imports
    ``triton.runtime.jit``. Defer until jit is importable.
    """

    _hotpatch_attempted = False

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):  # noqa: ANN001
        cann_request = fullname == "triton.language.extra.cann" or fullname.startswith(
            "triton.language.extra.cann."
        )
        if not _triton_jit_ready():
            return None
        if not cls._hotpatch_attempted:
            cls._hotpatch_attempted = True
            _install_hotpatch_safe()
        if not cann_request:
            return None
        _ensure_cann_language()
        if fullname == "triton.language.extra.cann":
            mod = sys.modules.get(fullname)
            return getattr(mod, "__spec__", None) if mod is not None else None
        # Submodules (libdevice, extension, …) resolve via package __path__.
        return None


def _install_cann_import_hook() -> None:
    if getattr(_install_cann_import_hook, "_installed", False):
        return
    sys.meta_path.insert(0, _DeferredAscendSetupFinder)
    _install_cann_import_hook._installed = True  # type: ignore[attr-defined]


try:
    # Register plugin so create_* / triton_ascend.ir exist before any later
    # cann.extension import (extension must not pull ir mid-init via package).
    register_ascend_native()
except Exception as _ascend_exc:
    warnings.warn(
        f"triton_ascend plugin registration failed ({_ascend_exc!r}); "
        "build standalone/ against a TRITON_EXT_ENABLED host.",
        ImportWarning,
        stacklevel=1,
    )

_install_cann_import_hook()

# If triton is already fully imported (e.g. ``import triton_ascend`` after
# ``import triton``), inject immediately; otherwise the meta_path hook waits.
if _triton_jit_ready():
    try:
        _ensure_cann_language()
    except Exception as _cann_exc:
        warnings.warn(
            f"triton_ascend cann language setup failed ({_cann_exc!r})",
            ImportWarning,
            stacklevel=1,
        )
    _install_hotpatch_safe()


__all__ = [
    "register_ascend_native",
    "plugin_library",
    "ir",
    "_ensure_cann_language",
]
