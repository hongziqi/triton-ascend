# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Map former BufferOpBuilder / AscendNPUIROpBuilder APIs onto OpInfo create_*.

Community ``ir.builder`` only gets ``create_<name>(list[Value])`` from the
plugin OpInfo table. Buffer language (``triton.extension.buffer``) and cann
extension still call monorepo-shaped helpers (``alloc``, ``get_buffer_ty``,
``to_buffer``, …). Install thin Python wrappers that pack arguments the way
``AscendOpInfo.cpp`` expects.
"""

from __future__ import annotations

import logging

_LOG = logging.getLogger("triton_ascend.hotpatch.buffer")
_INSTALLED = False


def _pack_i32(builder, value):
    return builder.get_int32(int(value))


def _pack_i64(builder, value):
    return builder.get_int64(int(value))


def _pack_string(builder, text: str):
    text = "" if text is None else str(text)
    out = [_pack_i32(builder, len(text))]
    out.extend(_pack_i32(builder, ord(ch)) for ch in text)
    return out


def _pack_int_vector(builder, values, width=64):
    pack = _pack_i32 if width == 32 else _pack_i64
    out = [_pack_i32(builder, len(values))]
    out.extend(pack(builder, v) for v in values)
    return out


def _pack_value_vector(builder, values):
    values = list(values)
    return [_pack_i32(builder, len(values)), *values]


def _call(builder, name, args):
    """OpInfo create_*; result recovered by ``opinfo_result`` wrap when installed."""
    fn = getattr(builder, f"create_{name}")
    return fn(list(args))


def _space_to_i32(space) -> int:
    """Encode address-space Attribute / enum / int as OpInfo i32 (-1 = none)."""
    if space is None:
        return -1
    if isinstance(space, bool):
        return -1
    if isinstance(space, int):
        return int(space)
    # enum.Enum (AddressSpace) or ascend_address_space_base.real_address_space
    if hasattr(space, "value") and not callable(space.value):
        try:
            return int(space.value)
        except (TypeError, ValueError):
            pass
    if hasattr(space, "real_address_space"):
        return _space_to_i32(space.real_address_space)
    # Unknown Attribute object — treat as present but unspecified numeric tag.
    return 0


def install_buffer_builder_methods() -> None:
    """Idempotent: attach monorepo buffer helpers onto ``ir.builder``."""
    global _INSTALLED
    if _INSTALLED:
        return

    from triton._C.libtriton import ir

    if getattr(ir.builder, "_ascend_buffer_builder_wrapped", False):
        _INSTALLED = True
        return

    try:
        import triton_ascend

        triton_ascend.register_ascend_native()
    except Exception as e:  # pragma: no cover
        _LOG.warning("register_ascend_native before buffer hotpatch failed: %s", e)

    if not hasattr(ir.builder, "create_buffer_ty_carrier"):
        _LOG.warning(
            "create_buffer_ty_carrier OpInfo missing; rebuild libtriton_ascend.so"
        )
        return

    def get_null_attr(self):
        return None

    def get_str_array_attr(self, array):
        # Pass-through: create_annotation_mark wrapper understands list[str].
        return list(array)

    def get_target_attribute(self, address_space):
        # OpInfo path cannot return MLIR Attributes; return HIVM enum i32.
        return _space_to_i32(address_space)

    def get_buffer_ty(self, shape, element_type, memory_space=None):
        elem_carrier = self.create_poison(element_type)
        args = []
        args += _pack_int_vector(self, list(shape))
        args.append(elem_carrier)
        args.append(_pack_i32(self, _space_to_i32(memory_space)))
        args += _pack_int_vector(self, [])
        carrier = _call(self, "buffer_ty_carrier", args)
        return carrier.get_type()

    def get_buffer_ty_with_strides(self, shape, element_type, strides,
                                   memory_space=None):
        elem_carrier = self.create_poison(element_type)
        args = []
        args += _pack_int_vector(self, list(shape))
        args.append(elem_carrier)
        args.append(_pack_i32(self, _space_to_i32(memory_space)))
        args += _pack_int_vector(self, list(strides))
        carrier = _call(self, "buffer_ty_carrier", args)
        return carrier.get_type()

    def alloc(self, memref_type):
        carrier = self.create_poison(memref_type)
        return _call(self, "alloc", [carrier])

    def to_buffer(self, src, address_space=None):
        return _call(
            self,
            "to_buffer",
            [src, _pack_i32(self, _space_to_i32(address_space))],
        )

    def to_tensor(self, src, writable=True):
        return _call(self, "to_tensor", [src, _pack_i32(self, bool(writable))])

    def subview(self, source, offsets, sizes, strides):
        args = [source]
        args += _pack_value_vector(self, list(offsets))
        args += _pack_int_vector(self, list(sizes))
        args += _pack_int_vector(self, list(strides))
        return _call(self, "subview", args)

    _orig_annotation_mark = getattr(ir.builder, "create_annotation_mark", None)

    def create_annotation_mark(self, ptr, key, attr=None):
        """Monorepo (ptr, key, attr) → OpInfo packed form."""
        args = [ptr] + _pack_string(self, key)
        if attr is None or attr is True:
            args.append(_pack_i32(self, 0))  # unit
        elif isinstance(attr, (list, tuple)) and (
            not attr or isinstance(attr[0], str)
        ):
            args.append(_pack_i32(self, 2))  # str array
            args.append(_pack_i32(self, len(attr)))
            for s in attr:
                args += _pack_string(self, s)
        elif isinstance(attr, int):
            args.append(_pack_i32(self, 1))
            args.append(_pack_i64(self, attr))
        else:
            # UnitAttr / unknown Attribute objects → unit
            args.append(_pack_i32(self, 0))
        if _orig_annotation_mark is None:
            return _call(self, "annotation_mark", args)
        # OpInfo binding is create_annotation_mark(self, list)
        return _orig_annotation_mark(self, args)

    _orig_bind_buffer = getattr(ir.builder, "create_bind_buffer", None)

    def create_bind_buffer(self, src, alloc_buf):
        if _orig_bind_buffer is None:
            return _call(self, "bind_buffer", [src, alloc_buf])
        return _orig_bind_buffer(self, [src, alloc_buf])

    ir.builder.get_null_attr = get_null_attr
    ir.builder.get_str_array_attr = get_str_array_attr
    ir.builder.get_target_attribute = get_target_attribute
    ir.builder.get_buffer_ty = get_buffer_ty
    ir.builder.get_buffer_ty_with_strides = get_buffer_ty_with_strides
    ir.builder.alloc = alloc
    ir.builder.to_buffer = to_buffer
    ir.builder.to_tensor = to_tensor
    ir.builder.subview = subview
    ir.builder.create_annotation_mark = create_annotation_mark
    ir.builder.create_bind_buffer = create_bind_buffer
    ir.builder._ascend_buffer_builder_wrapped = True
    _INSTALLED = True
    _LOG.info("Ascend buffer builder hot-patches installed")
