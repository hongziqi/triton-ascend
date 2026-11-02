# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Minimal AffineMap stand-in when HostExtend ``ascend_ir.affine_map`` is absent.

Descriptors are packed into OpInfo value vectors; ``AscendOpInfo.cpp`` rebuilds
real ``mlir::AffineMap`` / ``AffineMapAttr``. Enough for ``get_identity`` and
common constructors used by custom-op ``indexing_map``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


# kind tags consumed by AscendOpInfo decode
KIND_IDENTITY = 0
KIND_EMPTY = 1
KIND_CONSTANT = 2
KIND_PERMUTATION = 3
KIND_MINOR_IDENTITY = 4


@dataclass(frozen=True)
class AffineMapDesc:
    kind: int
    payload: tuple[Any, ...]

    def to_opinfo_ints(self) -> list[int]:
        """Flat i32 payload after kind (caller prefixes kind)."""
        if self.kind == KIND_IDENTITY:
            return [int(self.payload[0])]
        if self.kind == KIND_EMPTY:
            return []
        if self.kind == KIND_CONSTANT:
            return [int(self.payload[0])]
        if self.kind == KIND_PERMUTATION:
            perm = list(self.payload[0])
            return [len(perm), *[int(x) for x in perm]]
        if self.kind == KIND_MINOR_IDENTITY:
            return [int(self.payload[0]), int(self.payload[1])]
        raise ValueError(f"unsupported AffineMapDesc kind {self.kind}")


class _AffineMapAPI:
    """Drop-in for ``ascend_ir.affine_map`` static constructors."""

    @staticmethod
    def get_identity(num_dims: int, context=None) -> AffineMapDesc:
        return AffineMapDesc(KIND_IDENTITY, (int(num_dims),))

    @staticmethod
    def get_empty(context=None) -> AffineMapDesc:
        return AffineMapDesc(KIND_EMPTY, ())

    @staticmethod
    def get_constant(value: int, context=None) -> AffineMapDesc:
        return AffineMapDesc(KIND_CONSTANT, (int(value),))

    @staticmethod
    def get_permutation(permutation: Sequence[int], context=None) -> AffineMapDesc:
        return AffineMapDesc(KIND_PERMUTATION, (tuple(int(x) for x in permutation),))

    @staticmethod
    def get_minor_identity(dims: int, results: int, context=None) -> AffineMapDesc:
        return AffineMapDesc(KIND_MINOR_IDENTITY, (int(dims), int(results)))

    @staticmethod
    def get(num_dims: int, num_symbols: int, results, context=None) -> AffineMapDesc:
        # General maps need full expr trees; not supported without HostExtend.
        raise NotImplementedError(
            "affine_map.get(...) requires HostExtend or a fuller soft binding; "
            "use get_identity / get_permutation / get_constant when possible"
        )


affine_map = _AffineMapAPI()
AffineMap = AffineMapDesc
