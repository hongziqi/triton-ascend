# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Pack Python arguments for community OpInfo ``builder.create_*(list[Value])``.

Triton 3.7 binds each ``OpInfo.name`` onto ``ir.builder`` as
``create_<name>(self, Sequence[Value]) -> None``. Ascend packing uses
``ops[0]`` as the result slot; stock hosts neither prepend that slot nor return
``args[0]``. ``triton_ascend.backend.opinfo_result`` wraps Ascend ``create_*``
to prepend + recover the Value via module attrs (no community ``ir.cc`` edit).

Strings are ``i32`` length plus ``i32`` char codes (``AscendOpInfo::decodeString``).
"""

from __future__ import annotations

# Injected by triton_ascend._ensure_buffer_language (hotpatch); do not
# invade site-packages/triton/extension/.
import triton.extension.buffer.language as bl


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
    """Invoke ``builder.create_<name>``; return recovered Value when present.

    Relies on ``install_opinfo_result_recovery`` (prepend + module.walk).
    """
    fn = getattr(builder, f"create_{name}")
    return fn(list(args))
