/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Former patterns rewrote strided load/store via MakeTensorPtr / Advance /
 * PtrAnalysis. Those ops are gone from community Triton.
 */

#include "TritonToGraph/LegacyMemoryAccess/StridedLoadStoreRewrite.h"

namespace StridedLoadStoreRewrite {

LogicalResult LoadConverter::matchAndRewrite(triton::LoadOp,
                                             PatternRewriter &) const {
  return failure();
}

LogicalResult StoreConverter::matchAndRewrite(triton::StoreOp,
                                              PatternRewriter &) const {
  return failure();
}

} // namespace StridedLoadStoreRewrite
