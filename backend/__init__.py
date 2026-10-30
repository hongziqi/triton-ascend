# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Ascend backend package (entry point: triton.backends → triton_ascend.backend).
# Keep import side-effects minimal so community ``import triton`` is not poisoned
# during backend discovery.

__version__ = "3.8.0"

import logging
import sys

# Make ``triton.backends.ascend`` resolve to this package for legacy imports
# during entry-point discovery (package name is ``triton_ascend.backend``).
sys.modules.setdefault("triton.backends.ascend", sys.modules[__name__])


def _apply_ascend_patch():
    """Apply dual-package hot-patches + ensure Ascend dialects are registered.

    Former monorepo invasive edits to community ``triton`` (see
    ``backend/triton_hotpatch.py``) are restored here instead of patching
    host sources. Also re-registers plugin dialects for unit-test call sites.
    """
    try:
        from .triton_hotpatch import install as _install_hotpatch

        _install_hotpatch()
    except Exception as e:
        logging.warning(f"[Ascend] triton hotpatch skipped: {e}")

    try:
        from .compiler import register_ascend_plugin_dialects

        register_ascend_plugin_dialects()
    except Exception as e:
        logging.warning(f"[Ascend] plugin registration skipped: {e}")


try:
    from .testing import do_bench_npu
except Exception:
    do_bench_npu = None  # type: ignore

__all__ = ["do_bench_npu", "_apply_ascend_patch"]
