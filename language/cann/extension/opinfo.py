# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Pack Python arguments for community OpInfo ``builder.create_*(list[Value])``.

Triton 3.7 binds each ``OpInfo.name`` onto ``ir.builder`` as
``create_<name>(self, Sequence[Value]) -> None``. The callback writes the
result into ``operands[0]``; the host binder does not return that slot.
Strings are ``i32`` length plus ``i32`` char codes (``AscendOpInfo::decodeString``).
"""

from __future__ import annotations

try:
    import triton.extension.buffer.language as bl
except ModuleNotFoundError:
    class _AddressSpace:
        def to_ir(self, builder):
            raise NotImplementedError("address_space.to_ir")

    class _Buffer:
        pass

    class _BufferLanguage:
        address_space = _AddressSpace
        buffer = _Buffer

    bl = _BufferLanguage()


def pack_i32(builder, value):
    return builder.get_int32(int(value))


def pack_i64(builder, value):
    return builder.get_int64(int(value))


def pack_bool(builder, value):
    return builder.get_int1(bool(value))


def pack_string(builder, text: str):
    text = "" if text is None else str(text)
    out = [pack_i32(builder, len(text))]
    out.extend(pack_i32(builder, ord(ch)) for ch in text)
    return out


def pack_value_vector(builder, values):
    values = list(values)
    return [pack_i32(builder, len(values)), *values]


def call(builder, name, args):
    """Invoke ``builder.create_<name>`` with the OpInfo value vector."""
    fn = getattr(builder, f"create_{name}")
    return fn(list(args))
