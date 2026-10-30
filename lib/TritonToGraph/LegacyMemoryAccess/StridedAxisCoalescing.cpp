/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Former implementation coalesced MakeTensorPtr-based strided axes.
 * Community Triton removed MakeTensorPtr; host
 * add_rewrite_tensor_descriptor_to_pointer runs before Ascend, so this
 * entry point has no block-pointer IR to rewrite.
 */

#include "TritonToGraph/LegacyMemoryAccess/StridedAxisCoalescing.h"

namespace StridedAxisCoalescing {

void rewriteStridedAxisCoalesce(mlir::ModuleOp /*moduleOp*/) {}

} // namespace StridedAxisCoalescing
