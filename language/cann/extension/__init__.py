import sys

from triton.backends.ascend.utils import is_compile_on_910_95

# Do not ``from triton_ascend import ir`` here: this package is often imported
# while ``triton_ascend`` is still initializing (backend discovery → __init__).
# Prefer an already-published alias, else the host libtriton ir module.
_ascend_ir = sys.modules.get("triton_ascend.ir")
if _ascend_ir is None:
    from triton._C.libtriton import ir as _ascend_ir

# MLIR affine bindings (HostExtend-only on some builds). Soft-bind so OpInfo
# hosts can still import sibling extension submodules that do not need them.
affine_expr = getattr(_ascend_ir, "affine_expr", None)
affine_constant_expr = getattr(_ascend_ir, "affine_constant_expr", None)
affine_dim_expr = getattr(_ascend_ir, "affine_dim_expr", None)
affine_symbol_expr = getattr(_ascend_ir, "affine_symbol_expr", None)
affine_binary_op_expr = getattr(_ascend_ir, "affine_binary_op_expr", None)
affine_map = getattr(_ascend_ir, "affine_map", None)

AffineExpr = affine_expr
AffineConstantExpr = affine_constant_expr
AffineDimExpr = affine_dim_expr
AffineSymbolExpr = affine_symbol_expr
AffineBinaryOpExpr = affine_binary_op_expr
AffineMap = affine_map

from .core import (
    ascend_address_space,
    builtin,
    CORE,
    copy_from_ub_to_l1,
    copy,
    debug_barrier,
    fixpipe,
    FixpipeDMAMode,
    FixpipeDualDstMode,
    FixpipePreQuantMode,
    FixpipePreReluMode,
    int64,
    is_builtin,
    MODE,
    PIPE,
    SYNC_HINT,
    EVENT_ID,
    IteratorType,
    sub_vec_id,
    sub_vec_num,
    sync_block_all,
    sync_block_set,
    sync_block_wait,
    SYNC_IN_VF,
    conv1d,
    dot,
)

from .scope import scope

from .custom_op import (
    custom,
    custom_semantic,
    register_custom_op,
    SyncEventSlot,
)

from . import builtin_custom_ops

from .math_ops import (atan2, isfinited, finitef)

from .aux_ops import (
    parallel,
    compile_hint,
    multibuffer,
)

from .vec_ops import (
    insert_slice,
    extract_slice,
    get_element,
    sort,
    flip,
    cast,
)

from .mem_ops import (
    index_put,
    gather_out_to_ub,
    scatter_ub_to_out,
    index_select_simd,
)

__all__ = [
    # core
    "builtin",
    "copy_from_ub_to_l1",
    "copy",
    "CORE",
    "debug_barrier",
    "fixpipe",
    "FixpipeDMAMode",
    "FixpipeDualDstMode",
    "FixpipePreQuantMode",
    "FixpipePreReluMode",
    "int64",
    "is_builtin",
    "MODE",
    "PIPE",
    "SYNC_HINT",
    "EVENT_ID",
    "IteratorType",
    "sub_vec_id",
    "sub_vec_num",
    "sync_block_all",
    "SYNC_IN_VF",
    "conv1d",

    # address space
    "ascend_address_space",

    # ascend IR affine (MLIR)
    "affine_expr",
    "affine_constant_expr",
    "affine_dim_expr",
    "affine_symbol_expr",
    "affine_binary_op_expr",
    "affine_map",
    "AffineExpr",
    "AffineConstantExpr",
    "AffineDimExpr",
    "AffineSymbolExpr",
    "AffineBinaryOpExpr",
    "AffineMap",

    # scope
    "scope",

    # custom op
    "custom",
    "custom_semantic",
    "register_custom_op",
    "SyncEventSlot",

    # math ops
    "atan2",
    "isfinited",
    "finitef",

    # aux ops
    "sync_block_set",
    "sync_block_wait",
    "parallel",
    "compile_hint",
    "multibuffer",

    # vec ops
    "insert_slice",
    "extract_slice",
    "get_element",
    "sort",
    "flip",
    "cast",

    # mem ops
    "index_put",
    "gather_out_to_ub",
    "scatter_ub_to_out",
    "index_select_simd",

    # utils
    "is_compile_on_910_95",

    # dot ops
    "dot",
]
