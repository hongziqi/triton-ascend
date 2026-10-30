# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Hot-patch community Triton with former Ascend invasive (in-tree) edits.

Dual-package rule: do **not** patch stock ``triton`` sources / ``ir.cc``.
Restore behaviour at import / ``NPUOptions`` time by monkey-patching Python
and by relying on plugin OpInfo for C++-only APIs.

Inventory vs ``triton-ascend-*.patch`` (Python / recoverable bits)
=================================================================

| Former invasive site                         | Hot-patched here? | Notes |
|----------------------------------------------|-------------------|-------|
| ``ir.cc`` ``create_auto_overflow_assert``    | Yes (OpInfo + wrap) | Plugin OpInfo ``auto_overflow_assert`` + monorepo-shaped wrapper |
| ``semantic.device_assert(auto_overflow=)``   | Yes | |
| ``semantic.binary_op_sanitize_overflow_impl``| Yes | passes ``auto_overflow=True`` |
| ``runtime.interpreter.create_auto_overflow`` | Yes | |
| ``_utils.validate_block_shape`` power-of-two | Yes | Ascend allows non-PoT shapes |
| ``semantic.arange`` power-of-two check       | Yes | |
| ``compiler.parse`` Ascend artifacts          | Yes | ``ttadapter`` / ``npubin`` / ``bcmlir`` |
| ``semantic.dot`` HF32 / imprecise acc        | Yes | |
| ``CodeGenerator`` ``hacc.target``            | Yes | stock ``parse_attr`` (no HostExtend) |
| ``dtype_guard``                              | Best-effort | if ``dtype_guard.py`` is packaged |
| ``ir.cc`` HF32 enum / Traits.cpp PoT         | **No** | needs host rebuild / LLVM; not recoverable in Python |
| ``code_generator`` cann dispatch imports     | Partial | language injected via ``triton_ascend`` / ``setup_unified_builder`` |
| ``math.py`` / ``standard.py`` / ``jit`` bits | Deferred | add here when a failing case needs them |

Call ``install()`` from ``_apply_ascend_patch`` (and prefer once at plugin import).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

_LOG = logging.getLogger("triton_ascend.hotpatch")
_INSTALLED = False

# Exact frontend message used by stock sanitize_overflow (must stay in sync
# with triton.language.semantic.binary_op_sanitize_overflow_impl).
_OVERFLOW_ASSERT_MSG_RE = re.compile(
    r"^int\d+ overflow detected for operation \w+$"
)


def _pack_string(builder, s: str):
    """OpInfo string packing: i32 length + i32 char codes."""
    out = [builder.get_int32(len(s))]
    out.extend(builder.get_int32(ord(c)) for c in s)
    return out


def _install_auto_overflow_assert_builder() -> None:
    """Expose monorepo API ``builder.create_auto_overflow_assert(cond, msg)``.

    Community hosts only have ``create_assert``. Ascend registers OpInfo
    ``create_auto_overflow_assert(list[Value])`` via ``extend_with``; wrap it
    to the (condition, message) signature semantic.py expects.
    """
    from triton._C.libtriton import ir

    if getattr(ir.builder, "_ascend_auto_overflow_assert_wrapped", False):
        return

    # Ensure Dialect/OpInfo plugins are loaded so create_auto_overflow_assert exists.
    try:
        import triton_ascend

        triton_ascend.register_ascend_native()
    except Exception as e:  # pragma: no cover
        _LOG.warning("register_ascend_native before overflow hotpatch failed: %s", e)

    opinfo_fn = getattr(ir.builder, "create_auto_overflow_assert", None)
    if opinfo_fn is None:
        _LOG.warning(
            "create_auto_overflow_assert OpInfo missing; "
            "overflow asserts will not get tt.auto_overflow_assert "
            "(rebuild libtriton_ascend.so / pip install -e .)"
        )
        return

    def create_auto_overflow_assert(self, condition, message: str) -> None:
        args = [condition] + _pack_string(self, message)
        # OpInfo binding: create_<name>(self, list[Value])
        return opinfo_fn(self, args)

    ir.builder.create_auto_overflow_assert = create_auto_overflow_assert
    ir.builder._ascend_auto_overflow_assert_wrapped = True


def _install_semantic_overflow_patches() -> None:
    from triton.language import semantic as semantic_mod

    TritonSemantic = semantic_mod.TritonSemantic
    if getattr(TritonSemantic, "_ascend_overflow_hotpatch", False):
        return

    _orig_device_assert = TritonSemantic.device_assert
    _orig_sanitize = TritonSemantic.binary_op_sanitize_overflow_impl

    def device_assert(self, cond, msg, mask=None, auto_overflow: bool = False):
        import triton.language as tl

        if not self.builder.options.debug:
            return
        if mask is not None:
            cond = self.or_(cond, self.not_(mask))
        if auto_overflow and hasattr(self.builder, "create_auto_overflow_assert"):
            self.builder.create_auto_overflow_assert(cond.handle, msg)
            return self.make_tensor(None, tl.void)
        return _orig_device_assert(self, cond, msg, mask)

    def binary_op_sanitize_overflow_impl(self, lhs, rhs, binary_op):
        # Same body as stock Triton, but pass auto_overflow=True into device_assert.
        if lhs.type.scalar.int_bitwidth >= 64 or not self.builder.options.sanitize_overflow:
            return
        import triton.language as tl

        lhs_sca_ty = lhs.type.scalar
        rhs_sca_ty = rhs.type.scalar
        assert lhs_sca_ty == rhs_sca_ty, (
            f"expected matching operand types, got {lhs_sca_ty} and {rhs_sca_ty}"
        )
        assert lhs_sca_ty.is_int(), f"expected integer type, got {lhs_sca_ty}"
        lhs64 = self.cast(lhs, tl.int64)
        rhs64 = self.cast(rhs, tl.int64)
        ret = binary_op(lhs64, rhs64, False)
        max_value = self.scalar_constant(lhs_sca_ty.get_int_max_value(), tl.int64)
        min_value = self.scalar_constant(lhs_sca_ty.get_int_min_value(), tl.int64)
        cond = self.and_(
            self.less_equal(ret, max_value), self.greater_equal(ret, min_value)
        )
        msg = (
            f"int{lhs_sca_ty.int_bitwidth} overflow detected for operation "
            f"{binary_op.__name__}"
        )
        self.device_assert(cond, msg, None, auto_overflow=True)

    TritonSemantic.device_assert = device_assert
    TritonSemantic.binary_op_sanitize_overflow_impl = binary_op_sanitize_overflow_impl
    TritonSemantic._ascend_overflow_hotpatch = True
    TritonSemantic._ascend_orig_device_assert = _orig_device_assert
    TritonSemantic._ascend_orig_sanitize = _orig_sanitize


def _install_interpreter_overflow() -> None:
    try:
        from triton.runtime import interpreter as interp
    except Exception:
        return
    BuilderProxy = getattr(interp, "InterpreterBuilder", None) or getattr(
        interp, "builder", None
    )
    # Interpreter builder class lives as InterpretedFunction's builder type.
    # Patch the method on the class used by create_assert if present.
    for cls_name in ("InterpreterBuilder", "BuilderProxy"):
        cls = getattr(interp, cls_name, None)
        if cls is None:
            continue
        if getattr(cls, "_ascend_auto_overflow_assert", False):
            return
        if not hasattr(cls, "create_assert"):
            continue

        def create_auto_overflow_assert(self, condition, message):
            # Interpreter has no MLIR attr; reuse create_assert.
            return self.create_assert(condition, message)

        cls.create_auto_overflow_assert = create_auto_overflow_assert
        cls._ascend_auto_overflow_assert = True
        return

    # Fallback: attach on module for tests that look up by name.
    if not hasattr(interp, "create_auto_overflow_assert_note"):
        interp.create_auto_overflow_assert_note = (
            "Ascend hotpatch: InterpreterBuilder.create_auto_overflow_assert installed when available"
        )


def _install_block_shape_relax() -> None:
    """Drop power-of-two shape enforcement (former Traits.cpp / _utils patch)."""
    try:
        import triton._utils as utils
    except Exception:
        return
    if getattr(utils, "_ascend_block_shape_hotpatch", False):
        return
    orig = utils.validate_block_shape

    def validate_block_shape(shape):
        # Re-implement without PoT check; keep numel limit from original when possible.
        from triton._utils import TRITON_MAX_TENSOR_NUMEL

        numel = 1
        for i, d in enumerate(shape):
            if not isinstance(d, int):
                raise TypeError(
                    f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d)}]`"
                )
            numel *= d
        if numel > TRITON_MAX_TENSOR_NUMEL:
            raise ValueError(
                f"numel ({numel}) exceeds Triton's maximum tensor numel ({TRITON_MAX_TENSOR_NUMEL})"
            )
        return numel

    utils.validate_block_shape = validate_block_shape
    utils._ascend_block_shape_hotpatch = True
    utils._ascend_orig_validate_block_shape = orig


def _install_arange_relax() -> None:
    from triton.language.semantic import TritonSemantic

    if getattr(TritonSemantic, "_ascend_arange_hotpatch", False):
        return

    def arange(self, start: int, end: int, *, ret_ty=None):
        # Stock body minus power-of-two range check (Ascend invasive patch).
        import triton.language as tl

        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("arange's arguments must be of type tl.constexpr")
        if bool(start >> 32) or bool(end >> 32):
            raise ValueError("arange must fit in int32")
        if end <= start:
            raise ValueError(
                "arange's end argument must be greater than the start argument"
            )
        range_ = end - start
        shape = [range_]
        if ret_ty is None:
            ret_ty = tl.block_type(tl.int32, shape)
        ret_ty_ir = ret_ty.to_ir(self.builder)
        return self.make_tensor(
            self.builder.create_make_range(ret_ty_ir, start, end), ret_ty
        )

    TritonSemantic.arange = arange
    TritonSemantic._ascend_arange_hotpatch = True


def _install_compiler_parse() -> None:
    from pathlib import Path

    from triton.compiler import compiler as compiler_mod
    from triton._C.libtriton import ir as _ir

    if getattr(compiler_mod, "_ascend_parse_patch_applied", False):
        return

    def _patched_parse(full_name, ext, context):
        if ext in ("ttir", "ttgir"):
            module = _ir.parse_mlir_module(full_name, context)
            module.context = context
            return module
        if ext in ("llir", "ptx", "amdgcn", "ttadapter", "bcmlir"):
            return Path(full_name).read_text()
        if ext in ("cubin", "hsaco", "mlirbc", "npubin"):
            return Path(full_name).read_bytes()
        raise ValueError(f"unknown file extension {ext}")

    compiler_mod.parse = _patched_parse
    compiler_mod._ascend_parse_patch_applied = True


def _install_dot_hf32_guard() -> None:
    from triton.language.semantic import TritonSemantic

    if getattr(TritonSemantic, "_ascend_dot_patch_applied", False):
        return
    _original_dot = TritonSemantic.dot

    def _patched_dot(
        self, lhs, rhs, acc, input_precision, max_num_imprecise_acc, out_dtype
    ):
        if input_precision == "tf32":
            input_precision = "hf32"
        if max_num_imprecise_acc is not None:
            print(
                "max_num_imprecise_acc in tl.dot is not supported on Ascend yet. "
                "Thus it is ignored."
            )
            max_num_imprecise_acc = None
        return _original_dot(
            self, lhs, rhs, acc, input_precision, max_num_imprecise_acc, out_dtype
        )

    TritonSemantic.dot = _patched_dot
    TritonSemantic._ascend_dot_patch_applied = True


def _install_hacc_target_on_codegen() -> None:
    """Inject ``hacc.target`` after CodeGenerator init (community parse_attr)."""
    from triton.compiler.code_generator import CodeGenerator

    if getattr(CodeGenerator, "_ascend_patch_applied", False):
        return
    _original_cg_init = CodeGenerator.__init__

    def _patched_cg_init(self, *args, **kwargs):
        _original_cg_init(self, *args, **kwargs)
        options = getattr(self, "builder", None)
        options = getattr(options, "options", None) if options is not None else None
        arch = getattr(options, "arch", None) if options is not None else None
        if not arch:
            return
        try:
            from triton._C.libtriton import ir

            # Prefer module.context; fall back to builder context.
            ctx = getattr(self, "context", None) or self.builder.context
            b = ir.builder(ctx)
            if not hasattr(b, "parse_attr"):
                return
            self.module.set_attr(
                "hacc.target", b.parse_attr(f'#hacc.target<"{arch}">')
            )
        except Exception as e:
            _LOG.warning("Failed to set hacc.target: %s", e)

    CodeGenerator.__init__ = _patched_cg_init
    CodeGenerator._ascend_patch_applied = True


def _install_dtype_guard() -> None:
    from triton.compiler.code_generator import CodeGenerator

    if getattr(CodeGenerator, "_ascend_dtype_guard_applied", False):
        return
    try:
        from triton.backends.ascend.dtype_guard import install_dtype_guard

        install_dtype_guard()
        CodeGenerator._ascend_dtype_guard_applied = True
    except Exception as e:
        _LOG.debug("dtype_guard not installed: %s", e)


def is_overflow_sanitize_message(message: Optional[str]) -> bool:
    """Shared helper for Pass-side fallback (optional)."""
    if not message:
        return False
    return bool(_OVERFLOW_ASSERT_MSG_RE.match(message.strip()))


def install() -> None:
    """Idempotent: apply all dual-package Triton hot-patches."""
    global _INSTALLED
    if _INSTALLED:
        return
    _install_auto_overflow_assert_builder()
    _install_semantic_overflow_patches()
    _install_interpreter_overflow()
    _install_block_shape_relax()
    _install_arange_relax()
    _install_compiler_parse()
    _install_dot_hf32_guard()
    _install_hacc_target_on_codegen()
    _install_dtype_guard()
    _INSTALLED = True
    _LOG.info("Ascend Triton hot-patches installed")
