# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Triton Ascend package entry.

``import triton_ascend`` loads ``libtriton_ascend.so`` (owned TTIR→Linalg
in a private MLIRContext).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

_ASCEND_PLUGIN_LOADED = False
_native = None


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_native():
    global _native
    if _native is not None:
        return _native
    import types

    # Prefer cmake output ``libtriton_ascend.so`` over a stale setuptools-tagged
    # ``libtriton_ascend.cpython-*.so`` that importlib would otherwise pick first.
    c_dir = _package_dir() / "_C"
    plain = c_dir / "libtriton_ascend.so"
    tagged = sorted(c_dir.glob("libtriton_ascend.cpython-*.so"))
    load_path: Path | None = None
    if plain.is_file():
        load_path = plain
    elif tagged:
        load_path = tagged[0]
    else:
        env = os.environ.get("TRITON_ASCEND_EXTENSION")
        if env and Path(env).is_file():
            load_path = Path(env)

    if load_path is not None:
        spec = importlib.util.spec_from_file_location(
            "triton_ascend._C.libtriton_ascend", load_path
        )
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            if "triton_ascend._C" not in sys.modules:
                pkg = types.ModuleType("triton_ascend._C")
                pkg.__path__ = [str(c_dir)]
                sys.modules["triton_ascend._C"] = pkg
            spec.loader.exec_module(mod)
            _native = mod
            return _native

    try:
        _native = importlib.import_module("triton_ascend._C.libtriton_ascend")
        return _native
    except ImportError as exc:
        raise ImportError(
            "libtriton_ascend.so not found under triton_ascend/_C/. "
            "Build with python3 setup.py install / bdist_wheel (LLVM_SYSPATH required)."
        ) from exc


def _ensure_cann_language() -> None:
    """Make ``triton.language.extra.cann`` importable without dropping it into extra/.

    Nightly ``triton_key()`` walks ``language.extra`` packages on disk; a full
    cann tree there is imported during compile and needs the owned IR builder.
    """
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


def register_ascend_native() -> None:
    """dlopen ``libtriton_ascend.so`` (private MLIRContext)."""
    global _ASCEND_PLUGIN_LOADED
    if _ASCEND_PLUGIN_LOADED:
        return
    _load_native()
    _ASCEND_PLUGIN_LOADED = True


try:
    _ensure_cann_language()
    register_ascend_native()
    _nat = _load_native()
    ir = getattr(_nat, "ir", None)
    passes = getattr(_nat, "passes", None)
except Exception as _ascend_exc:  # soft-fail: do not poison bare `import triton`
    import warnings

    warnings.warn(
        f"triton_ascend native registration failed ({_ascend_exc!r}); "
        "rebuild with python3 setup.py install / bdist_wheel (LLVM_SYSPATH required).",
        ImportWarning,
        stacklevel=1,
    )
    ir = None  # type: ignore
    passes = None  # type: ignore


__all__ = ["ir", "passes", "register_ascend_native", "pipeline", "_load_native"]


# Lazy submodule for Step 12 text→text API
def __getattr__(name: str):
    if name == "pipeline":
        # Must not ``from . import pipeline`` — that re-enters __getattr__.
        return importlib.import_module(".pipeline", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
