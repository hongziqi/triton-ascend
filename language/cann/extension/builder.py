# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
Ascend builder helpers for the community OpInfo path.

``ir.builder.extend_with`` already installs ``create_<name>(list[Value])``
factories from AscendOpInfo. ``setup_unified_builder`` is therefore a no-op for
op attachment (no AscendNPUIROpBuilder / HostExtend).

Thin Python wrappers below pack non-list / kwargs call sites into the OpInfo
Value-vector convention (see PLUGINIZATION_1_3.md).
"""

__all__ = [
    "create_builder_method_wrapper",
    "attach_builder_methods",
    "setup_unified_builder",
    "pack_i32",
    "pack_i64",
    "pack_bool",
    "pack_string",
    "pack_int_vector",
    "pack_value_vector",
    "attach_opinfo_wrappers",
]


def create_builder_method_wrapper(main_builder, delegate_builder, method_name):
    """Delegate a method while syncing insertion point / location (legacy)."""
    delegate_method = getattr(delegate_builder, method_name)

    def wrapper(*args, **kwargs):
        saved_ip = main_builder.get_insertion_point()
        saved_loc = main_builder.get_loc()
        delegate_builder.restore_insertion_point(saved_ip)
        if saved_loc:
            delegate_builder.set_loc(saved_loc)
        result = delegate_method(*args, **kwargs)
        main_builder.restore_insertion_point(saved_ip)
        if saved_loc:
            main_builder.set_loc(saved_loc)
        return result

    wrapper.__name__ = method_name
    wrapper.__doc__ = getattr(delegate_method, '__doc__', None)
    return wrapper


def attach_builder_methods(main_builder, delegate_builder, method_names):
    for method_name in method_names:
        wrapper = create_builder_method_wrapper(main_builder, delegate_builder, method_name)
        setattr(main_builder, method_name, wrapper)


def pack_i32(builder, value):
    return builder.get_int32(int(value))


def pack_i64(builder, value):
    return builder.get_int64(int(value))


def pack_bool(builder, value):
    return builder.get_int1(bool(value))


def pack_string(builder, s: str):
    """i32 length + i32 char codes (AscendOpInfo decodeString)."""
    out = [pack_i32(builder, len(s))]
    out.extend(pack_i32(builder, ord(c)) for c in s)
    return out


def pack_int_vector(builder, values, width=32):
    pack = pack_i32 if width == 32 else pack_i64
    out = [pack_i32(builder, len(values))]
    out.extend(pack(builder, v) for v in values)
    return out


def pack_value_vector(builder, values):
    out = [pack_i32(builder, len(values))]
    out.extend(values)
    return out


def _type_carrier(builder, ty):
    """Scalar zero constant whose MLIR type is the carrier."""
    # Prefer int/float zeros; fall back to undef-like via get_null_value if present.
    if hasattr(builder, "get_null_value"):
        return builder.get_null_value(ty)
    if hasattr(builder, "create_bitcast"):
        # last resort: use an i32 zero and rely on caller for real carriers
        return builder.get_int32(0)
    return builder.get_int32(0)


def attach_opinfo_wrappers(main_builder):
    """Attach kwargs-friendly wrappers that call OpInfo create_*(list)."""
    b = main_builder

    def _call(name, args):
        fn = getattr(b, f"create_{name}")
        return fn(list(args))

    # Alias sync_block_* (legacy names) → create_sync_block_*
    if hasattr(b, "create_sync_block_set"):
        def sync_block_set(sender, receiver, event_id, sender_pipe, receiver_pipe):
            args = []
            args += pack_string(b, sender)
            args += pack_string(b, receiver)
            args.append(event_id)
            sp = sender_pipe.value if hasattr(sender_pipe, "value") else int(sender_pipe)
            rp = receiver_pipe.value if hasattr(receiver_pipe, "value") else int(receiver_pipe)
            args.append(pack_i32(b, sp))
            args.append(pack_i32(b, rp))
            return _call("sync_block_set", args)

        def sync_block_wait(sender, receiver, event_id, sender_pipe, receiver_pipe):
            args = []
            args += pack_string(b, sender)
            args += pack_string(b, receiver)
            args.append(event_id)
            sp = sender_pipe.value if hasattr(sender_pipe, "value") else int(sender_pipe)
            rp = receiver_pipe.value if hasattr(receiver_pipe, "value") else int(receiver_pipe)
            args.append(pack_i32(b, sp))
            args.append(pack_i32(b, rp))
            return _call("sync_block_wait", args)

        def sync_block_all(mode, event_id):
            args = pack_string(b, mode) + [pack_i32(b, event_id)]
            return _call("sync_block_all", args)

        b.sync_block_set = sync_block_set
        b.sync_block_wait = sync_block_wait
        b.sync_block_all = sync_block_all

    if hasattr(b, "create_extract_scalar"):
        def create_extract_scalar(src, indices):
            return _call("extract_scalar", [src] + list(indices))

        def create_extract_slice(ful, offs, sizes, strides):
            args = [ful]
            args += pack_value_vector(b, offs)
            args += pack_int_vector(b, sizes)
            args += pack_int_vector(b, strides)
            return _call("extract_slice", args)

        def create_insert_slice(ful, sub, offs, sizes, strides):
            args = [ful, sub]
            args += pack_value_vector(b, offs)
            args += pack_int_vector(b, sizes)
            args += pack_int_vector(b, strides)
            return _call("insert_slice", args)

        def create_flip(src, dim):
            return _call("flip", [src, pack_i64(b, dim)])

        def create_sort(src, dim, descending):
            return _call("sort", [src, pack_i64(b, dim), pack_i32(b, bool(descending))])

        def create_tanh(val):
            return _call("tanh", [val])

        def create_get_sub_vec_id():
            return _call("get_sub_vec_id", [])

        def create_copy_buffer(src, dst):
            return _call("copy_buffer", [src, dst])

        def create_copy_tensor(src, dst):
            return _call("copy_tensor", [src, dst])

        def scope_return(operands):
            return _call("scope_return", list(operands))

        def create_auto_overflow_assert(condition, message: str):
            """Monorepo-shaped wrapper over OpInfo ``create_auto_overflow_assert``."""
            return _call("auto_overflow_assert", [condition] + pack_string(b, message))

        b.create_extract_scalar = create_extract_scalar
        b.create_extract_slice = create_extract_slice
        b.create_insert_slice = create_insert_slice
        b.create_flip = create_flip
        b.create_sort = create_sort
        b.create_tanh = create_tanh
        b.create_get_sub_vec_id = create_get_sub_vec_id
        b.create_copy_buffer = create_copy_buffer
        b.create_copy_tensor = create_copy_tensor
        b.scope_return = scope_return
        # OpInfo list form → monorepo (cond, msg) signature.
        b.create_auto_overflow_assert = create_auto_overflow_assert


def setup_unified_builder(main_builder, ascend_builder=None):
    """No-op for HostExtend; optionally attach OpInfo packing wrappers.

    ``create_*`` factories already live on the community builder via
    ``ir.builder.extend_with``. When ``ascend_builder`` is passed (legacy), it
    is stored only as ``_ascend_builder`` for call sites that still look it up.
    """
    if ascend_builder is not None:
        main_builder._ascend_builder = ascend_builder
    attach_opinfo_wrappers(main_builder)
    return main_builder
