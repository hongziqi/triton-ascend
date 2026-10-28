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
    """Idempotent ensure: ``import triton_ascend`` already registers the plugin.

    Kept for unit tests / legacy call sites that still invoke this name.
    """
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
