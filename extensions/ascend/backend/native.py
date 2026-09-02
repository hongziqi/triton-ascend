# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Lazy access to Ascend native (C++) bindings for out-of-tree plugin builds."""

from __future__ import annotations

from typing import Any


class AscendNativeMissingError(ImportError):
    """Raised when Ascend C++ bindings are not available in the host process."""


_ascend_ir: Any | None = None
_libtriton_ascend: Any | None = None
_buffer_ir: Any | None = None
_distributed: Any | None = None
_checked: bool = False


def _try_load():
    global _ascend_ir, _libtriton_ascend, _buffer_ir, _distributed, _checked
    if _checked:
        return
    _checked = True

    # Out-of-tree plugin may register a native .so via triton_ascend package first.
    try:
        import triton_ascend  # noqa: F401
    except Exception:
        pass

    import triton._C.libtriton as libtriton

    _buffer_ir = getattr(libtriton, "buffer_ir", None)

    if hasattr(libtriton, "ascend"):
        _libtriton_ascend = libtriton.ascend
        _ascend_ir = libtriton.ascend.ir
        return

    # Future: standalone pybind module shipped inside triton-ascend wheel.
    for mod_name in ("triton_ascend_ir", "triton_ascend_native"):
        try:
            mod = __import__(mod_name)
        except ImportError:
            continue
        if hasattr(mod, "ir"):
            _ascend_ir = mod.ir
            _libtriton_ascend = mod
            return

    try:
        from triton._C.libtriton import distributed as _distributed  # type: ignore
    except ImportError:
        _distributed = None


def require_ascend_ir():
    _try_load()
    if _ascend_ir is None:
        raise AscendNativeMissingError(
            "Ascend C++ bindings are missing. On a bare community triton host, "
            "install triton-ascend with its native plugin .so (Step 3+) or use a "
            "monolithic triton-ascend build."
        )
    return _ascend_ir


def get_ascend_ir():
    _try_load()
    return _ascend_ir


def get_libtriton_ascend():
    _try_load()
    return _libtriton_ascend


def get_buffer_ir():
    _try_load()
    return _buffer_ir


def get_distributed():
    _try_load()
    return _distributed


def native_available() -> bool:
    _try_load()
    return _ascend_ir is not None
