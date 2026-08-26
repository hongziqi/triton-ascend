"""Minimal @jit kernels for plugin puncture (must be top-level for source inspect)."""
import triton
import triton.language as tl


@triton.jit
def kernel_baseline(BLOCK_SIZE: tl.constexpr):
    return


@triton.jit
def kernel_hooked(BLOCK_SIZE: tl.constexpr):
    return


@triton.jit
def kernel_plugin(BLOCK_SIZE: tl.constexpr):
    return
