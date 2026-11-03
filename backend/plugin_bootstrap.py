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
    """Return the Ascend plugin ``.so`` next to this install (editable or wheel).

    Editable: ``<repo>/_C/libtriton_ascend.so``
    Wheel: ``site-packages/triton_ascend/_C/libtriton_ascend.so``

    Does **not** read ``TRITON_PLUGIN_PATHS`` (that env may be stale after
    reinstall). Override only via ``TRITON_ASCEND_EXTENSION`` /
    ``TRITON_ASCEND_PASS_PLUGIN``.
    """
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
    """Publish ``plugin_library()`` into ``TRITON_PLUGIN_PATHS``.

    Drops env entries whose files no longer exist. Host Triton 3.7 aborts if
    *any* path in ``TRITON_PLUGIN_PATHS`` fails to ``dlopen`` — a leftover
    site-packages path after ``pip install -e`` is a common footgun.
    """
    so = plugin_library()
    existing = os.environ.get("TRITON_PLUGIN_PATHS", "")
    kept: list[str] = []
    seen: set[str] = set()
    for raw in existing.split(":"):
        if not raw:
            continue
        try:
            resolved = str(Path(raw).resolve())
        except OSError:
            continue
        if resolved in seen or not Path(resolved).is_file():
            continue
        seen.add(resolved)
        kept.append(resolved)

    if so is not None:
        path = str(so.resolve())
        if path not in seen:
            kept.insert(0, path)

    if kept:
        os.environ["TRITON_PLUGIN_PATHS"] = ":".join(kept)
    elif "TRITON_PLUGIN_PATHS" in os.environ:
        del os.environ["TRITON_PLUGIN_PATHS"]

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
    """Scrub/set plugin path, then version-check (idempotent).

    ``ensure_plugin_path`` must run before any ``import triton``: stock 3.7.1
    loads plugins from ``TRITON_PLUGIN_PATHS`` at first ``libtriton`` import.
    """
    global _BOOTSTRAPPED
    already = _libtriton_already_imported()
    so = ensure_plugin_path()
    if not _BOOTSTRAPPED:
        check_triton_version()
        _BOOTSTRAPPED = True

    if already and so is not None:
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
