# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Ascend backend package (entry point: triton.backends → triton_ascend.backend).
# Keep import side-effects minimal so community ``import triton`` is not poisoned
# during backend discovery (Step 12 standalone / nightly host).

__version__ = "3.8.0"

import logging
import sys

# Make ``triton.backends.ascend`` resolve to this package for legacy imports
# during entry-point discovery (package name is ``triton_ascend.backend``).
sys.modules.setdefault("triton.backends.ascend", sys.modules[__name__])

ascend_ir = None
try:
    from triton_ascend import ir as ascend_ir
except Exception:
    pass


def _apply_ascend_patch():
    """Optional host monkey-patches — call after backends are fully discovered."""
    try:
        from .compiler import register_ascend_native

        register_ascend_native()
    except Exception as e:
        logging.warning(f"[Ascend] native registration skipped: {e}")

    try:
        from triton.compiler.code_generator import CodeGenerator
    except Exception as e:
        logging.warning(f"[Ascend Patch] CodeGenerator unavailable: {e}")
        return

    if not getattr(CodeGenerator, "_ascend_patch_applied", False):
        _original_cg_init = CodeGenerator.__init__

        def _patched_cg_init(self, *args, **kwargs):
            _original_cg_init(self, *args, **kwargs)
            options = self.builder.options
            context = self.context
            if ascend_ir is None or not hasattr(options, "arch") or not options.arch:
                return
            # Owned builder wraps triton_ascend.ir.context, not host libtriton MLIRContext.
            # compiler.py sets hacc.target after parsing TTIR into the owned context.
            owned_ctx_cls = getattr(ascend_ir, "context", None)
            if owned_ctx_cls is None or not isinstance(context, owned_ctx_cls):
                return
            try:
                builder = ascend_ir.ascendnpu_ir_builder(context, options.arch)
                target_attr_str = f'#hacc.target<"{options.arch}">'
                self.module.set_attr("hacc.target", builder.parse_attr(target_attr_str))
            except Exception as e:
                logging.warning(f"[Ascend Patch] Failed to set hacc.target: {e}")

        CodeGenerator.__init__ = _patched_cg_init
        CodeGenerator._ascend_patch_applied = True


try:
    from .testing import do_bench_npu
except Exception:
    do_bench_npu = None  # type: ignore

__all__ = ["do_bench_npu", "_apply_ascend_patch"]
