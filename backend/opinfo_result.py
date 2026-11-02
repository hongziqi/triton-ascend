# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Recover OpInfo results without patching community ``ir.cc``.

Stock Triton 3.7.1 binds plugin OpInfo as ``create_*(self, list[Value]) -> None``
with **no** result-slot prepend and **no** ``return args[0]``. Ascend packing
still uses ``ops[0]`` as the result slot (``ops[1+]`` = args), matching main.

This module:
1. Stashes the current ``ModuleOp`` on the builder (``_ascend_module``).
2. Wraps Ascend ``create_*`` so each call prepends a placeholder and returns the
   produced Value (reuse ``args[0]`` as placeholder when possible — no extra IR;
   poison i1 only for nullary ops).
3. Recovers that Value via ``module.walk`` + attrs from ``AscendOpInfo.cpp``
   ``setOpResult`` (``ascend.opinfo_seq`` / ``ascend.last_opinfo_seq``).
"""

from __future__ import annotations

import logging

_LOG = logging.getLogger("triton_ascend.hotpatch.opinfo")
_INSTALLED = False

# OpInfo names registered by AscendOpInfo.cpp (include create_ prefix).
_ASCEND_OPINFO_CREATE = (
    "create_extract_scalar",
    "create_extract_slice",
    "create_insert_slice",
    "create_custom_op_for_inter_core_sync",
    "create_index_select_simd",
    "create_index_put",
    "create_gather_out_to_ub",
    "create_scatter_ub_to_out",
    "create_conv1d",
    "create_dot",
    "create_sort",
    "create_flip",
    "create_tanh",
    "create_annotation",
    "create_fixpipe",
    "create_annotation_mark",
    "create_bind_buffer",
    "create_debug_barrier",
    "create_custom_op",
    "create_custom_macro_op",
    "create_scope_op",
    "create_scope_return",
    "create_sync_block_set",
    "create_sync_block_wait",
    "create_sync_block_all",
    "create_get_sub_vec_id",
    "create_copy_buffer",
    "create_copy_tensor",
    "create_convert_layout",
    "create_auto_overflow_assert",
    "create_buffer_ty_carrier",
    "create_alloc",
    "create_to_buffer",
    "create_to_tensor",
    "create_subview",
)


def stash_module(builder, module) -> None:
    """Record the ModuleOp used for OpInfo result recovery."""
    builder._ascend_module = module


def placeholder(builder, args):
    """Result-slot placeholder. Prefer reusing an existing arg (no new IR)."""
    args = list(args)
    if args:
        return args[0]
    return builder.create_poison(builder.get_int1_ty())


def recover_result(builder):
    """Return the Value tagged by the latest Ascend OpInfo ``setOpResult``."""
    mod = getattr(builder, "_ascend_module", None)
    if mod is None:
        return None
    seq = mod.get_int_attr("ascend.last_opinfo_seq")
    if seq is None:
        return None
    found = []

    def _visit(op):
        op_seq = op.get_int_attr("ascend.opinfo_seq")
        if op_seq is None or op_seq != seq:
            return
        if op.get_num_results() > 0:
            found.append(op.get_result(0))

    mod.walk(_visit)
    return found[0] if found else None


def install_opinfo_result_recovery() -> None:
    """Idempotent: stash module on ``create_module``; wrap Ascend OpInfo creates.

    After this, ``builder.create_<op>([args...])`` prepends a result slot and
    returns the produced Value (or ``None`` for void ops), matching main's
    patched ``ir.cc`` behaviour — without touching community Triton sources.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from triton._C.libtriton import ir

    if getattr(ir.builder, "_ascend_opinfo_result_recovery", False):
        _INSTALLED = True
        return

    try:
        import triton_ascend

        triton_ascend.register_ascend_native()
    except Exception as e:  # pragma: no cover
        _LOG.warning("register_ascend_native before opinfo recovery failed: %s", e)

    _orig_create_module = ir.builder.create_module

    def create_module(self):
        mod = _orig_create_module(self)
        stash_module(self, mod)
        return mod

    ir.builder.create_module = create_module

    wrapped = 0
    for name in _ASCEND_OPINFO_CREATE:
        orig = getattr(ir.builder, name, None)
        if orig is None or getattr(orig, "_ascend_opinfo_recovered", False):
            continue

        def _make(orig_fn):
            def wrapped_create(self, args):
                mod = getattr(self, "_ascend_module", None)
                before = (
                    mod.get_int_attr("ascend.last_opinfo_seq")
                    if mod is not None
                    else None
                )
                packed = list(args)
                ph = placeholder(self, packed)
                orig_fn(self, [ph, *packed])
                after = (
                    mod.get_int_attr("ascend.last_opinfo_seq")
                    if mod is not None
                    else None
                )
                # Void / early-return OpInfo callbacks do not bump the seq.
                if after is None or after == before:
                    return None
                return recover_result(self)

            wrapped_create._ascend_opinfo_recovered = True  # type: ignore[attr-defined]
            return wrapped_create

        setattr(ir.builder, name, _make(orig))
        wrapped += 1

    ir.builder._ascend_opinfo_result_recovery = True
    _INSTALLED = True
    _LOG.info("Ascend OpInfo result recovery installed (%d create_* wrapped)", wrapped)
