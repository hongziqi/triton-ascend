# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Load the Ascend plugin into the host Triton context (Triton 3.7 ABI).

``libtriton_ascend.so`` exports ``tritonGetPluginInfo``. Triton 3.7 has no
``extend_with``; plugins are loaded from ``TRITON_PLUGIN_PATHS`` when
``libtriton`` is first imported. Ensure the env var is set *before*
``import triton`` (the installed ``.pth`` / ``_autoload_plugin`` does this).

Verify path after import::

    assert hasattr(passes.plugin, "add_triton_to_linalg")
    assert hasattr(ir.builder, "create_extract_scalar")
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
    plain = c_dir / "libtriton_ascend.so"
    if plain.is_file():
        return plain
    matches = [p for p in c_dir.glob("libtriton_ascend*.so") if p.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _plugin_paths_contain(so: Path) -> bool:
    env = os.environ.get("TRITON_PLUGIN_PATHS", "")
    want = str(so.resolve())
    return any(Path(p).resolve() == Path(want) for p in env.split(":") if p)


def register_ascend_native() -> None:
    """Verify the Ascend plugin was loaded via ``TRITON_PLUGIN_PATHS`` (idempotent).

    Triton 3.7 binds PassInfo / OpInfo / DialectInfo at ``libtriton`` import
    time. There is no ``extend_with``; calling this after import can only
    check that the env-based load succeeded.
    """
    global _ASCEND_PLUGIN_LOADED, ir
    if _ASCEND_PLUGIN_LOADED:
        return
    libtriton = importlib.import_module("triton._C.libtriton")
    host_ir = libtriton.ir
    if getattr(libtriton, "_ascend_plugin_loaded", False):
        ir = host_ir
        _ASCEND_PLUGIN_LOADED = True
        return

    so = plugin_library()
    passes = libtriton.passes
    plugin_mod = getattr(passes, "plugin", None)
    has_pass = plugin_mod is not None and hasattr(plugin_mod, "add_triton_to_linalg")
    has_op = hasattr(host_ir, "builder") and hasattr(
        host_ir.builder, "create_extract_scalar"
    )

    if not has_pass or not has_op:
        so_hint = str(so) if so is not None else "<missing libtriton_ascend.so>"
        in_env = _plugin_paths_contain(so) if so is not None else False
        raise RuntimeError(
            "Ascend plugin was not loaded by host Triton 3.7.\n"
            f"  so={so_hint}\n"
            f"  TRITON_PLUGIN_PATHS contains so: {in_env}\n"
            f"  has add_triton_to_linalg: {has_pass}\n"
            f"  has create_extract_scalar: {has_op}\n"
            "Set TRITON_PLUGIN_PATHS to the .so *before* importing triton "
            "(or reinstall so the triton_ascend .pth runs at startup). "
            "Host must be built with TRITON_EXT_ENABLED=1."
        )

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
    """Publish ``triton.backends.ascend`` before cann.libdevice imports utils."""
    if "triton.backends.ascend" in sys.modules:
        return
    backend = importlib.import_module(__name__ + ".backend")
    sys.modules.setdefault("triton.backends.ascend", backend)


def _ensure_cann_language() -> None:
    """Expose ``triton.language.extra.cann`` without copying it into the wheel."""
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
    """Finish Ascend setup after ``import triton`` completes."""

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
        return None


def _install_cann_import_hook() -> None:
    if getattr(_install_cann_import_hook, "_installed", False):
        return
    sys.meta_path.insert(0, _DeferredAscendSetupFinder)
    _install_cann_import_hook._installed = True  # type: ignore[attr-defined]


# Best-effort: if autoload did not run, still publish the path for the *next*
# process. Too late for the current libtriton if already imported.
try:
    from ._autoload_plugin import ensure_plugin_path as _ensure_plugin_path

    _ensure_plugin_path()
except Exception:
    pass

try:
    register_ascend_native()
except Exception as _ascend_exc:
    warnings.warn(
        f"triton_ascend plugin registration failed ({_ascend_exc!r}); "
        "export TRITON_PLUGIN_PATHS=<libtriton_ascend.so> before importing triton.",
        ImportWarning,
        stacklevel=1,
    )

_install_cann_import_hook()

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
