# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Backend-discovery bootstrap for the Ascend Triton 3.7.1 plugin.

Sets ``TRITON_PLUGIN_PATHS`` before ``libtriton`` is imported by the Ascend
compiler module. No site ``.pth`` — initialization happens when Triton discovers
the ``triton.backends`` entry point ``ascend = triton_ascend.backend``.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

_REQUIRED_TRITON = "3.7.1"
_BOOTSTRAPPED = False


def _package_dir() -> Path:
    # backend/ -> package root (triton_ascend)
    return Path(__file__).resolve().parent.parent


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


def ensure_plugin_path() -> Path | None:
    """Prepend ``libtriton_ascend.so`` to ``TRITON_PLUGIN_PATHS`` if present."""
    so = plugin_library()
    if so is None:
        return None
    path = str(so.resolve())
    existing = os.environ.get("TRITON_PLUGIN_PATHS", "")
    parts = [p for p in existing.split(":") if p]
    if path not in parts:
        os.environ["TRITON_PLUGIN_PATHS"] = (
            path if not parts else path + ":" + ":".join(parts)
        )
    return so


def check_triton_version() -> str | None:
    """Warn if the installed host Triton is not the supported 3.7.1 release."""
    ver: str | None = None
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            ver = version("triton")
        except PackageNotFoundError:
            ver = None
    except Exception:
        ver = None

    if ver is None:
        try:
            import triton  # noqa: WPS433 — best-effort fallback

            ver = getattr(triton, "__version__", None)
        except Exception:
            ver = None

    if ver is None:
        warnings.warn(
            "[triton-ascend] could not determine the installed triton version; "
            f"this plugin requires triton=={_REQUIRED_TRITON}. "
            f"Install with: pip install triton=={_REQUIRED_TRITON}",
            UserWarning,
            stacklevel=2,
        )
        return None

    # Accept 3.7.1 and local builds that still report 3.7.1*.
    if not str(ver).startswith(_REQUIRED_TRITON):
        warnings.warn(
            f"[triton-ascend] unsupported triton version {ver!r}; "
            f"expected triton=={_REQUIRED_TRITON}. "
            f"Install with: pip install triton=={_REQUIRED_TRITON}",
            UserWarning,
            stacklevel=2,
        )
    return ver


def _libtriton_already_imported() -> bool:
    return "triton._C.libtriton" in sys.modules


def bootstrap_ascend_plugin() -> Path | None:
    """Version check + set ``TRITON_PLUGIN_PATHS`` (idempotent).

    Preferred call sites (earliest wins):

    1. ``export TRITON_PLUGIN_PATHS=...`` before starting Python (see README)
    2. ``import triton_ascend`` before ``import triton``
    3. Backend discovery (``triton.backends`` entry point)

    Note: stock Triton 3.7.1 imports ``libtriton`` from ``runtime.autotuner``
    *before* backend discovery, so (3) alone is often too late unless (1)/(2)
    already set the env.
    """
    global _BOOTSTRAPPED
    if not _BOOTSTRAPPED:
        check_triton_version()
        _BOOTSTRAPPED = True

    already = _libtriton_already_imported()
    so = ensure_plugin_path()
    if already and so is not None:
        # Path set after libtriton init cannot load the plugin (pluginsLoaded).
        try:
            libtriton = sys.modules.get("triton._C.libtriton")
            plugin_mod = getattr(getattr(libtriton, "passes", None), "plugin", None)
            loaded = plugin_mod is not None and hasattr(
                plugin_mod, "add_triton_to_linalg"
            )
        except Exception:
            loaded = False
        if not loaded:
            warnings.warn(
                "[triton-ascend] libtriton was imported before TRITON_PLUGIN_PATHS "
                "was set, so the Ascend plugin was not loaded. "
                "Fix: `import triton_ascend` before `import triton`, or "
                f"export TRITON_PLUGIN_PATHS={so}",
                UserWarning,
                stacklevel=2,
            )
    return so
