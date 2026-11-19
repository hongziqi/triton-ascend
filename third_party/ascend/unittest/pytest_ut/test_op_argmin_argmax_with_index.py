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

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _maximum_with_index(a_value, a_index, b_value, b_index):
    """Prefer larger value; on tie prefer smaller index."""
    mask = a_value > b_value
    mask = mask | ((a_value == b_value) & (a_index < b_index))
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def _minimum_with_index(a_value, a_index, b_value, b_index):
    """Prefer smaller value; on tie prefer smaller index."""
    mask = a_value < b_value
    mask = mask | ((a_value == b_value) & (a_index < b_index))
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def _reduce_with_index_kernel(in_v, in_i, out_idx, N: tl.constexpr, IS_MIN: tl.constexpr):
    """Core op under test: reduce (value, index) along axis 0."""
    v = tl.load(in_v + tl.arange(0, N)[:, None])
    i = tl.load(in_i + tl.arange(0, N)[:, None])
    if IS_MIN:
        _, idx = tl.reduce((v, i), 0, _minimum_with_index)
    else:
        _, idx = tl.reduce((v, i), 0, _maximum_with_index)
    tl.store(out_idx + tl.arange(0, 1), idx.to(tl.int32))

@triton.jit
def _two_tile_argmax_kernel(in_ptr, out_idx, n, BLOCK: tl.constexpr):
    """Inductor-style 2-tile merge then final reduce; exposes global-index loss.

    After merge, accumulator holds correct global indices; final reduce must
    return payload (e.g. 6), not lane within the [BLOCK,1] column (e.g. 2).
    """
    offs = tl.arange(0, BLOCK)[:, None]
    acc_v = tl.full([BLOCK, 1], float("-inf"), tl.float32)
    acc_i = tl.full([BLOCK, 1], 2147483647, tl.int32)
    for start in tl.range(0, n, BLOCK):
        idx = start + offs
        mask = idx < n
        val = tl.load(in_ptr + tl.reshape(idx, [BLOCK]), tl.reshape(mask, [BLOCK]),
                      other=0.0).to(tl.float32)
        val = tl.reshape(val, [BLOCK, 1])
        gidx = idx.to(tl.int32)
        nxt_v, nxt_i = _maximum_with_index(acc_v, acc_i, val, gidx)
        acc_v = tl.where(mask, nxt_v, acc_v)
        acc_i = tl.where(mask, nxt_i, acc_i)
    _, out = tl.reduce((acc_v, acc_i), 0, _maximum_with_index)
    tl.store(out_idx + tl.arange(0, 1), out.to(tl.int32))


def _run_reduce(v_1d, i_1d, is_min: bool) -> int:
    n = v_1d.numel()
    v = v_1d.view(n, 1).contiguous()
    i = i_1d.view(n, 1).contiguous()
    out = torch.empty((1, ), device="npu", dtype=torch.int32)
    _reduce_with_index_kernel[(1, )](v, i, out, N=n, IS_MIN=is_min)
    return int(out.item())


def _ref_index_payload(v_1d, i_1d, is_min: bool) -> int:
    """Reference: index label at torch.arg{min,max}(value)."""
    lane = int(torch.argmin(v_1d).item() if is_min else torch.argmax(v_1d).item())
    return int(i_1d[lane].item())


def test_argmax_index_payload_not_lane():
    """Smoking gun: index payload != lane id.

    values [1,9,3,7], indices [10,20,30,40]
    ref = i[torch.argmax(v)]; got==lane means returned lane id.
    """
    v = torch.tensor([1.0, 9.0, 3.0, 7.0], device="npu", dtype=torch.float32)
    i = torch.tensor([10, 20, 30, 40], device="npu", dtype=torch.int32)
    ref = _ref_index_payload(v, i, is_min=False)
    got = _run_reduce(v, i, is_min=False)
    lane = int(torch.argmax(v).item())
    assert got == ref, f"expect payload={ref}, got={got} (lane={lane})"


def test_argmin_index_payload_not_lane():
    """Same for min: ref = i[torch.argmin(v)]."""
    v = torch.tensor([4.0, 1.0, 3.0, 2.0], device="npu", dtype=torch.float32)
    i = torch.tensor([10, 20, 30, 40], device="npu", dtype=torch.int32)
    ref = _ref_index_payload(v, i, is_min=True)
    got = _run_reduce(v, i, is_min=True)
    lane = int(torch.argmin(v).item())
    assert got == ref, f"expect payload={ref}, got={got} (lane={lane})"


def test_argmax_lane_equals_payload_false_pass():
    """Control: index == arange → lane-return accidentally matches (false pass)."""
    v = torch.tensor([1.0, 9.0, 3.0, 7.0], device="npu", dtype=torch.float32)
    i = torch.arange(4, device="npu", dtype=torch.int32)
    ref = _ref_index_payload(v, i, is_min=False)
    got = _run_reduce(v, i, is_min=False)
    assert got == ref, f"lane==payload control: expect={ref} got={got}"


def test_two_tile_argmax_returns_global_index_not_lane():
    """Data [0,1,2,3 | 4,5,9,7], BLOCK=4; ref=torch.argmax(data), lane-return gives 2."""
    data = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 7.0],
                        device="npu", dtype=torch.float32)
    ref = int(torch.argmax(data).item())
    out = torch.empty((1, ), device="npu", dtype=torch.int32)
    _two_tile_argmax_kernel[(1, )](data, out, data.numel(), BLOCK=4)
    got = int(out.item())
    assert got == ref, f"expect global index={ref}, got={got}"
