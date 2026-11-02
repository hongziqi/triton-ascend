# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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

__all__ = [
    "fixpipe",
    "create_address_space",
]

import enum
from typing import (TypeVar, List, Union)

from triton._C.libtriton import ir
import triton.language.core as tl
import triton.language.extra.cann.extension as al
from triton.backends.ascend.utils import is_register_based_arch

from .opinfo import bl, call, pack_bool, pack_i32, pack_string

T = TypeVar('T')


def create_address_space(address_space, builder: ir.builder) -> ir.attribute:
    getter = getattr(builder, "get_target_attribute", None)
    if getter is None:
        raise AttributeError(
            "ir.builder has no get_target_attribute; "
            "HIVM address_space attrs are not OpInfo create_* ops"
        )
    return getter(address_space)


class PIPE(enum.Enum):
    PIPE_S = 0
    PIPE_V = 1
    PIPE_M = 2
    PIPE_MTE1 = 3
    PIPE_MTE2 = 4
    PIPE_MTE3 = 5
    PIPE_ALL = 6
    PIPE_FIX = 10


class SYNC_HINT(enum.Enum):
    WAIT = 0
    SET = 1
    INTERNAL = 2


class EVENT_ID(enum.Enum):
    EVENT_ID0 = 0
    EVENT_ID1 = 1
    EVENT_ID2 = 2
    EVENT_ID3 = 3
    EVENT_ID4 = 4
    EVENT_ID5 = 5
    EVENT_ID6 = 6
    EVENT_ID7 = 7


def _event_id_value(event_id, _semantic):
    if isinstance(event_id, int):
        return _semantic.to_tensor(tl.constexpr(event_id)).handle
    if isinstance(event_id, tl.constexpr):
        return _semantic.to_tensor(event_id).handle
    return event_id.handle


def _sync_block(op_name, sender, receiver, event_id, sender_pipe, receiver_pipe, _semantic):
    builder = _semantic.builder
    args = []
    args += pack_string(builder, sender)
    args += pack_string(builder, receiver)
    args.append(_event_id_value(event_id, _semantic))
    args.append(pack_i32(builder, sender_pipe.value))
    args.append(pack_i32(builder, receiver_pipe.value))
    return call(builder, op_name, args)


def create_sync_block_set(sender, receiver, event_id, sender_pipe: PIPE, receiver_pipe: PIPE, _semantic=None):
    return _sync_block("sync_block_set", sender, receiver, event_id, sender_pipe, receiver_pipe, _semantic)


def create_sync_block_wait(sender, receiver, event_id, sender_pipe: PIPE, receiver_pipe: PIPE, _semantic=None):
    return _sync_block("sync_block_wait", sender, receiver, event_id, sender_pipe, receiver_pipe, _semantic)


def sub_vec_id(_semantic=None) -> tl.tensor:
    return tl.tensor(call(_semantic.builder, "get_sub_vec_id", []), tl.int64)


def copy_from_ub_to_l1(src: Union[tl.tensor, bl.buffer], dst: Union[tl.tensor, bl.buffer], _semantic=None):
    if not is_register_based_arch(_semantic.builder.options.arch):
        raise RuntimeError("this feature is only supported on Ascend910_95 and Ascend960")
    if isinstance(src, tl.tensor) or isinstance(dst, tl.tensor):
        raise TypeError("tensor not support yet")
    if src.shape != dst.shape:
        raise TypeError("src and dst must have same shape")
    if src.dtype != dst.dtype:
        raise TypeError("src and dst need to have the same type")
    if isinstance(src, bl.buffer) and isinstance(dst, bl.buffer):
        if src.space != al.ascend_address_space.UB:
            raise TypeError("src's AddressSpace must be UB")
        if dst.space != al.ascend_address_space.L1:
            raise TypeError("dst's AddressSpace must be L1")
        call(_semantic.builder, "copy_buffer", [src.handle, dst.handle])
    else:
        raise TypeError("src and dst must be tl.tensor or bl.buffer")


def copy(src: Union[tl.tensor, bl.buffer], dst: Union[tl.tensor, bl.buffer], _semantic=None):
    if not is_register_based_arch(_semantic.builder.options.arch):
        raise RuntimeError("this feature is only supported on Ascend910_95 and Ascend960")
    if isinstance(src, tl.tensor) or isinstance(dst, tl.tensor):
        raise TypeError("tensor not support yet")
    if src.shape != dst.shape:
        raise TypeError("src and dst must have same shape")
    if src.dtype != dst.dtype:
        raise TypeError("src and dst need to have the same type")
    if isinstance(src, bl.buffer) and isinstance(dst, bl.buffer):
        if src.space != al.ascend_address_space.UB:
            raise TypeError("src's AddressSpace must be UB")
        if dst.space not in (al.ascend_address_space.L1, al.ascend_address_space.UB):
            raise TypeError("dst's AddressSpace must be UB or L1")
        call(_semantic.builder, "copy_buffer", [src.handle, dst.handle])
    else:
        raise TypeError("src and dst must be tl.tensor or bl.buffer")


def fixpipe(
    src: tl.tensor,
    dst,
    dma_mode,
    dual_dst_mode,
    pre_quant_mode,
    pre_relu_mode,
    _semantic=None,
):
    builder = _semantic.builder
    dst_value = ir.value() if dst is None else dst.handle
    result = call(builder, "fixpipe", [
        src.handle,
        dst_value,
        pack_i32(builder, dma_mode.value),
        pack_i32(builder, dual_dst_mode.value),
        pack_i32(builder, pre_quant_mode.value),
        pack_i32(builder, pre_relu_mode.value),
    ])
    if dst is None:
        if dual_dst_mode == al.FixpipeDualDstMode.ROW_SPLIT:
            new_shape = list(src.type.shape)
            if len(new_shape) >= 1 and new_shape[0] > 0:
                new_shape[0] = new_shape[0] // 2
            new_type = tl.block_type(src.type.element_ty, new_shape)
            return tl.tensor(result, new_type)
        elif dual_dst_mode == al.FixpipeDualDstMode.COLUMN_SPLIT:
            new_shape = list(src.type.shape)
            if len(new_shape) >= 2 and new_shape[1] > 0:
                new_shape[1] = new_shape[1] // 2
            new_type = tl.block_type(src.type.element_ty, new_shape)
            return tl.tensor(result, new_type)
        else:
            return tl.tensor(result, src.type)
    return None


def debug_barrier(sync_mode: str, _semantic=None) -> None:
    builder = _semantic.builder
    target = tl.tensor(builder.get_int64(0), tl.int64)
    # OpInfo stores an optional i64 on the mark. The mode name is the attr key.
    call(builder, "debug_barrier", [target.handle] + pack_string(builder, sync_mode))


def conv1d(input_tensor: tl.tensor, weight_tensor: tl.tensor, bias: Union[tl.tensor, None], stride: int,
           padding: Union[int, tuple], dilation: int, groups: int, output_shape, _semantic=None) -> tl.tensor:
    builder = _semantic.builder
    bias_handle = ir.value() if bias is None else bias.handle
    pad = padding[0] if isinstance(padding, tuple) else padding

    element_ty = input_tensor.type.element_ty
    output_ty = tl.block_type(element_ty, output_shape)
    # Type carrier: OpInfo reads ops[idx].getType() as the result type.
    out_ty = output_ty.to_ir(builder)
    carrier = builder.get_null_value(out_ty) if hasattr(builder, "get_null_value") else builder.get_int32(0)
    out = call(builder, "conv1d", [
        input_tensor.handle,
        weight_tensor.handle,
        bias_handle,
        pack_i32(builder, stride),
        pack_i32(builder, pad),
        pack_i32(builder, dilation),
        pack_i32(builder, groups),
        carrier,
    ])
    return tl.tensor(out, output_ty)


def dot(a: tl.tensor, b: tl.tensor, fractal_a: bool, fractal_b: bool, fractal_c: bool, output_shape,
        _semantic=None) -> tl.tensor:
    # Pass a Value list so pybind selects the OpInfo overload
    # create_dot(Sequence[Value]) -> None, not community tt.dot.
    builder = _semantic.builder
    out = call(builder, "dot", [
        a.handle,
        b.handle,
        pack_bool(builder, fractal_a),
        pack_bool(builder, fractal_b),
        pack_bool(builder, fractal_c),
    ])
    # Result carries the cube accumulator dtype: f32 for float inputs, i32 for int8.
    in_ty = a.type.element_ty
    acc_ty = tl.float32 if in_ty.is_floating() else tl.int32
    output_ty = tl.block_type(acc_ty, output_shape)
    return tl.tensor(out, output_ty)


def conv2d(input_tensor: tl.tensor, weight_tensor: tl.tensor, bias: Union[tl.tensor, None], stride: Union[int, tuple],
           padding: Union[int, tuple], dilation: Union[int,
                                                       tuple], groups: int, output_shape, _semantic=None) -> tl.tensor:
    raise RuntimeError(
        "create_conv2d is not in getAscendOpInfoTable; "
        "this tree's TritonAscendOps.td has no Conv2dOp"
    )
