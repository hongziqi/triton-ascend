// Community Ascend OpInfo table for ir.builder.extend_with.
// Ported from ascend_ir.cc create_* / sync_block_* (no HostExtend).
//
// Convention (AddOpCallback):
//   operands[0] = result slot (host inserts empty Value; callback writes result)
//   ints/bools  = arith.constant Values
//   strings     = i32 length + i32 char codes
//   variadic    = i32 length + elements
//
// No conv2d: TritonAscendOps.td in this tree has no Conv2dOp.

#include "AscendOpInfo.h"

#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include <optional>
#include <string>
#include <vector>

using namespace mlir;

namespace mlir::triton::ascend {
namespace {

// ---------------------------------------------------------------------------
// Packing helpers
// ---------------------------------------------------------------------------

static std::optional<int64_t> extractConstantInt(Value v) {
  if (!v)
    return std::nullopt;
  Operation *def = v.getDefiningOp();
  if (auto constIntOp = dyn_cast_or_null<arith::ConstantIntOp>(def))
    return constIntOp.value();
  if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(def)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

static bool extractConstantBool(Value v, bool def = false) {
  if (auto c = extractConstantInt(v))
    return *c != 0;
  return def;
}

/// Decode string packed as i32 len + i32 char codes. Advances ``idx``.
static bool decodeString(const std::vector<Value> &ops, size_t &idx,
                         std::string &out) {
  if (idx >= ops.size())
    return false;
  auto lenOpt = extractConstantInt(ops[idx++]);
  if (!lenOpt || *lenOpt < 0)
    return false;
  size_t len = static_cast<size_t>(*lenOpt);
  if (idx + len > ops.size())
    return false;
  out.clear();
  out.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    auto c = extractConstantInt(ops[idx++]);
    if (!c)
      return false;
    out.push_back(static_cast<char>(*c));
  }
  return true;
}

/// Decode i32-length-prefixed Value list. Advances ``idx``.
static bool decodeValueList(const std::vector<Value> &ops, size_t &idx,
                            SmallVectorImpl<Value> &out) {
  if (idx >= ops.size())
    return false;
  auto lenOpt = extractConstantInt(ops[idx++]);
  if (!lenOpt || *lenOpt < 0)
    return false;
  size_t len = static_cast<size_t>(*lenOpt);
  if (idx + len > ops.size())
    return false;
  out.clear();
  out.reserve(len);
  for (size_t i = 0; i < len; ++i)
    out.push_back(ops[idx++]);
  return true;
}

/// Decode i32-length-prefixed int list (as constant Values). Advances ``idx``.
static bool decodeIntList(const std::vector<Value> &ops, size_t &idx,
                          SmallVectorImpl<int64_t> &out) {
  if (idx >= ops.size())
    return false;
  auto lenOpt = extractConstantInt(ops[idx++]);
  if (!lenOpt || *lenOpt < 0)
    return false;
  size_t len = static_cast<size_t>(*lenOpt);
  if (idx + len > ops.size())
    return false;
  out.clear();
  out.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    auto v = extractConstantInt(ops[idx++]);
    if (!v)
      return false;
    out.push_back(*v);
  }
  return true;
}

static Value toIndex(TritonOpBuilder &b, Value v) {
  if (!v || v.getType().isIndex())
    return v;
  return b.create<arith::IndexCastOp>(b.getBuilder().getIndexType(), v);
}

// ---------------------------------------------------------------------------
// Sync helpers (from ascend_ir.cc)
// ---------------------------------------------------------------------------

static hivm::TCoreTypeAttr getSyncCore(MLIRContext *ctx, StringRef opName,
                                       StringRef sender) {
  if (sender == "cube") {
    if (opName == "sync_block_set")
      return hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::CUBE);
    return hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::VECTOR);
  }
  // default / "vector"
  if (opName == "sync_block_set")
    return hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::VECTOR);
  return hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::CUBE);
}

static void buildSyncBlockOp(TritonOpBuilder &self, const std::string &opName,
                             const std::string &sender,
                             const std::string & /*receiver*/, Value id,
                             hivm::PIPE senderPipe, hivm::PIPE receiverPipe) {
  auto *ctx = self.getBuilder().getContext();
  auto coreAttr = getSyncCore(ctx, opName, sender);
  auto prodPipe = hivm::PipeAttr::get(ctx, senderPipe);
  auto consPipe = hivm::PipeAttr::get(ctx, receiverPipe);
  Value idI64 = id;
  if (!id.getType().isInteger(64)) {
    auto i64Ty = IntegerType::get(ctx, 64);
    idI64 = mlir::convertScalarToDtype(self.getBuilder(), id.getLoc(), id,
                                       i64Ty, /*isUnsignedCast=*/true);
  }
  if (opName == "sync_block_set")
    self.create<hivm::SyncBlockSetOp>(coreAttr, prodPipe, consPipe, idI64);
  else
    self.create<hivm::SyncBlockWaitOp>(coreAttr, prodPipe, consPipe, idI64);
}

static void getSyncBlockModeAndPipes(MLIRContext *ctx, const std::string &mode,
                                     hivm::SyncBlockModeAttr &modeAttr,
                                     hivm::PipeAttr &cubePipe,
                                     hivm::PipeAttr &vectorPipe) {
  if (mode == "all_cube") {
    modeAttr = hivm::SyncBlockModeAttr::get(ctx, hivm::SyncBlockMode::ALL_CUBE);
    cubePipe = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
    vectorPipe = hivm::PipeAttr{};
  } else if (mode == "all_vector") {
    modeAttr =
        hivm::SyncBlockModeAttr::get(ctx, hivm::SyncBlockMode::ALL_VECTOR);
    cubePipe = hivm::PipeAttr{};
    vectorPipe = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
  } else if (mode == "all") {
    modeAttr = hivm::SyncBlockModeAttr::get(ctx, hivm::SyncBlockMode::ALL);
    cubePipe = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
    vectorPipe = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
  } else if (mode == "all_sub_vector") {
    modeAttr =
        hivm::SyncBlockModeAttr::get(ctx, hivm::SyncBlockMode::ALL_SUB_VECTOR);
    cubePipe = hivm::PipeAttr{};
    vectorPipe = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
  } else {
    modeAttr = hivm::SyncBlockModeAttr::get(ctx, hivm::SyncBlockMode::ALL);
    cubePipe = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
    vectorPipe = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
  }
}

// ---------------------------------------------------------------------------
// Op callbacks
// ---------------------------------------------------------------------------

// create_extract_scalar(src, *indices) → Value
static void addExtractScalar(TritonOpBuilder &b, std::vector<Value> &ops) {
  if (ops.size() < 2)
    return;
  Value src = ops[1];
  SmallVector<Value> indices;
  for (size_t i = 2; i < ops.size(); ++i)
    indices.push_back(toIndex(b, ops[i]));
  ops[0] = b.create<tensor::ExtractOp>(src, indices);
}

// create_extract_slice(ful, offs[], sizes[], strides[])
static void addExtractSlice(TritonOpBuilder &b, std::vector<Value> &ops) {
  size_t idx = 1;
  if (idx >= ops.size())
    return;
  Value ful = ops[idx++];
  SmallVector<Value> offsVals;
  SmallVector<int64_t> sizs, strds;
  if (!decodeValueList(ops, idx, offsVals) || !decodeIntList(ops, idx, sizs) ||
      !decodeIntList(ops, idx, strds))
    return;

  SmallVector<Value> offsets;
  SmallVector<int64_t> staticOffsets;
  for (Value o : offsVals) {
    offsets.push_back(toIndex(b, o));
    staticOffsets.push_back(ShapedType::kDynamic);
  }
  SmallVector<Value> sizes;
  SmallVector<int64_t> staticSizes;
  SmallVector<int64_t> retSizes;
  for (int64_t s : sizs) {
    staticSizes.push_back(s);
    retSizes.push_back(s);
  }
  SmallVector<Value> strides;
  SmallVector<int64_t> staticStrides;
  for (int64_t s : strds) {
    strides.push_back(b.create<arith::ConstantIndexOp>(s));
    staticStrides.push_back(ShapedType::kDynamic);
  }
  auto retTy = RankedTensorType::get(
      retSizes, cast<RankedTensorType>(ful.getType()).getElementType());
  ops[0] = b.create<tensor::ExtractSliceOp>(retTy, ful, offsets, sizes, strides,
                                            staticOffsets, staticSizes,
                                            staticStrides);
}

// create_insert_slice(ful, sub, offs[], sizes[], strides[])
static void addInsertSlice(TritonOpBuilder &b, std::vector<Value> &ops) {
  size_t idx = 1;
  if (ops.size() < 3)
    return;
  Value ful = ops[idx++];
  Value sub = ops[idx++];
  SmallVector<Value> offsVals;
  SmallVector<int64_t> sizs, strds;
  if (!decodeValueList(ops, idx, offsVals) || !decodeIntList(ops, idx, sizs) ||
      !decodeIntList(ops, idx, strds))
    return;

  SmallVector<Value> offsets;
  SmallVector<int64_t> staticOffsets;
  for (Value o : offsVals) {
    offsets.push_back(toIndex(b, o));
    staticOffsets.push_back(ShapedType::kDynamic);
  }
  SmallVector<Value> sizes;
  SmallVector<int64_t> staticSizes;
  for (int64_t s : sizs)
    staticSizes.push_back(s);
  SmallVector<Value> strides;
  SmallVector<int64_t> staticStrides;
  for (int64_t s : strds) {
    strides.push_back(b.create<arith::ConstantIndexOp>(s));
    staticStrides.push_back(ShapedType::kDynamic);
  }
  ops[0] = b.create<tensor::InsertSliceOp>(sub, ful, offsets, sizes, strides,
                                           staticOffsets, staticSizes,
                                           staticStrides);
}

// create_custom_op_for_inter_core_sync(op_name, mode_or_sender, id) → void
static void addCustomOpForInterCoreSync(TritonOpBuilder &b,
                                        std::vector<Value> &ops) {
  size_t idx = 1;
  std::string opName, modeOrSender;
  if (!decodeString(ops, idx, opName) || !decodeString(ops, idx, modeOrSender))
    return;
  if (idx >= ops.size())
    return;
  auto idOpt = extractConstantInt(ops[idx]);
  if (!idOpt)
    return;
  auto args = b.getBuilder().getArrayAttr(
      {b.getBuilder().getStringAttr(modeOrSender),
       b.getBuilder().getI32IntegerAttr(static_cast<int32_t>(*idOpt))});
  b.create<triton::ascend::CustomOp>(opName, args, ValueRange());
}

// create_index_select_simd(src, index, dim, srcShape[], srcOffset[],
//                          readShape[], returnShape[])
static void addIndexSelectSimd(TritonOpBuilder &b, std::vector<Value> &ops) {
  size_t idx = 1;
  if (ops.size() < 4)
    return;
  Value src = ops[idx++];
  Value index = ops[idx++];
  auto dimOpt = extractConstantInt(ops[idx++]);
  if (!dimOpt)
    return;
  SmallVector<Value> srcShape, srcOffset;
  SmallVector<int64_t> readShapeI64, returnShapeI64;
  if (!decodeValueList(ops, idx, srcShape) ||
      !decodeValueList(ops, idx, srcOffset) ||
      !decodeIntList(ops, idx, readShapeI64) ||
      !decodeIntList(ops, idx, returnShapeI64))
    return;

  Type elemType;
  if (auto ptrTy = dyn_cast<triton::PointerType>(src.getType()))
    elemType = ptrTy.getPointeeType();
  else
    return;

  SmallVector<int64_t> retShape(returnShapeI64.begin(), returnShapeI64.end());
  auto retTensorType = RankedTensorType::get(retShape, elemType);

  SmallVector<Value> srcShapeIndex, srcOffsetIndex;
  for (Value v : srcShape)
    srcShapeIndex.push_back(toIndex(b, v));
  for (Value v : srcOffset)
    srcOffsetIndex.push_back(toIndex(b, v));

  SmallVector<int32_t> readShape32;
  for (int64_t s : readShapeI64)
    readShape32.push_back(static_cast<int32_t>(s));

  auto &builder = b.getBuilder();
  auto loc = b.getLastLoc();
  auto dimAttr = builder.getI32IntegerAttr(static_cast<int32_t>(*dimOpt));
  auto readShapeAttr = builder.getDenseI32ArrayAttr(readShape32);
  auto op = builder.create<triton::ascend::IndexSelectSimdOp>(
      loc, retTensorType, src, index, dimAttr, srcShapeIndex, srcOffsetIndex,
      readShapeAttr);
  ops[0] = op.getResult();
}

// create_index_put(ptr, index, value, dim, indexBoundary,
//                  endOffset[], startOffset[], dstStride[]) → void
static void addIndexPut(TritonOpBuilder &b, std::vector<Value> &ops) {
  size_t idx = 1;
  if (ops.size() < 6)
    return;
  Value ptr = ops[idx++];
  Value index = ops[idx++];
  Value value = ops[idx++];
  auto dimOpt = extractConstantInt(ops[idx++]);
  auto boundOpt = extractConstantInt(ops[idx++]);
  if (!dimOpt || !boundOpt)
    return;
  SmallVector<Value> endOffset, startOffset, dstStride;
  if (!decodeValueList(ops, idx, endOffset) ||
      !decodeValueList(ops, idx, startOffset) ||
      !decodeValueList(ops, idx, dstStride))
    return;
  auto dimVal = b.create<arith::ConstantIntOp>(b.getBuilder().getI32Type(),
                                               static_cast<int32_t>(*dimOpt));
  auto boundVal = b.create<arith::ConstantIntOp>(b.getBuilder().getI64Type(),
                                                 *boundOpt);
  b.create<triton::ascend::IndexPutOp>(ptr, index, value, dimVal, boundVal,
                                       endOffset, startOffset, dstStride);
}

// create_gather_out_to_ub(src, index, indexBoundary, dim,
//                         srcStride[], endOffset[], startOffset[], [other])
static void addGatherOutToUb(TritonOpBuilder &b, std::vector<Value> &ops) {
  size_t idx = 1;
  if (ops.size() < 5)
    return;
  Value src = ops[idx++];
  Value index = ops[idx++];
  auto boundOpt = extractConstantInt(ops[idx++]);
  auto dimOpt = extractConstantInt(ops[idx++]);
  if (!boundOpt || !dimOpt)
    return;
  SmallVector<Value> srcStride, endOffset, startOffset;
  if (!decodeValueList(ops, idx, srcStride) ||
      !decodeValueList(ops, idx, endOffset) ||
      !decodeValueList(ops, idx, startOffset))
    return;
  Value other;
  if (idx < ops.size())
    other = ops[idx++];

  auto elemTy = cast<triton::PointerType>(src.getType()).getPointeeType();
  auto idxTy = cast<RankedTensorType>(index.getType());
  auto idxShape = idxTy.getShape();
  SmallVector<int64_t> retShape(idxShape.begin(), idxShape.end());
  auto resType = RankedTensorType::get(retShape, elemTy);
  auto boundVal = b.create<arith::ConstantIntOp>(b.getBuilder().getI64Type(),
                                                 *boundOpt);
  auto dimVal = b.create<arith::ConstantIntOp>(b.getBuilder().getI32Type(),
                                               static_cast<int32_t>(*dimOpt));
  ops[0] = b.create<triton::ascend::GatherOutToUbOp>(
      resType, src, index, boundVal, dimVal, srcStride, endOffset, startOffset,
      other);
}

// create_scatter_ub_to_out(...) → void
static void addScatterUbToOut(TritonOpBuilder &b, std::vector<Value> &ops) {
  size_t idx = 1;
  if (ops.size() < 6)
    return;
  Value ptr = ops[idx++];
  Value value = ops[idx++];
  Value index = ops[idx++];
  auto boundOpt = extractConstantInt(ops[idx++]);
  auto dimOpt = extractConstantInt(ops[idx++]);
  if (!boundOpt || !dimOpt)
    return;
  SmallVector<Value> dstStride, endOffset, startOffset;
  if (!decodeValueList(ops, idx, dstStride) ||
      !decodeValueList(ops, idx, endOffset) ||
      !decodeValueList(ops, idx, startOffset))
    return;
  auto boundVal = b.create<arith::ConstantIntOp>(b.getBuilder().getI64Type(),
                                                 *boundOpt);
  auto dimVal = b.create<arith::ConstantIntOp>(b.getBuilder().getI32Type(),
                                               static_cast<int32_t>(*dimOpt));
  b.create<triton::ascend::ScatterUbToOutOp>(ptr, value, index, boundVal,
                                             dimVal, dstStride, endOffset,
                                             startOffset);
}

// create_conv1d(input, weight, [bias|null], stride, pad, dil, groups,
//               out_type_carrier)
static void addConv1d(TritonOpBuilder &b, std::vector<Value> &ops) {
  size_t idx = 1;
  if (ops.size() < 8)
    return;
  Value input = ops[idx++];
  Value weight = ops[idx++];
  Value bias = ops[idx++]; // may be null/poison; treated as optional
  auto stride = extractConstantInt(ops[idx++]);
  auto padding = extractConstantInt(ops[idx++]);
  auto dilation = extractConstantInt(ops[idx++]);
  auto groups = extractConstantInt(ops[idx++]);
  if (!stride || !padding || !dilation || !groups || idx >= ops.size())
    return;
  Type outputType = ops[idx].getType();
  auto &builder = b.getBuilder();
  // Null bias: use empty Value when defining op is absent / null typed.
  Value biasValue = bias;
  ops[0] = b.create<triton::ascend::Conv1dOp>(
      outputType, input, weight, biasValue, builder.getI64IntegerAttr(*stride),
      builder.getI64IntegerAttr(*padding), builder.getI64IntegerAttr(*dilation),
      builder.getI64IntegerAttr(*groups));
}

// create_dot(a, b, fractal_a, fractal_b, fractal_c)
static void addDot(TritonOpBuilder &b, std::vector<Value> &ops) {
  if (ops.size() < 6)
    return;
  Value a = ops[1];
  Value bb = ops[2];
  bool fa = extractConstantBool(ops[3]);
  bool fb = extractConstantBool(ops[4]);
  bool fc = extractConstantBool(ops[5]);
  auto &builder = b.getBuilder();
  ops[0] = b.create<triton::ascend::DotOp>(
      a, bb, builder.getBoolAttr(fa), builder.getBoolAttr(fb),
      builder.getBoolAttr(fc));
}

// create_sort(src, dim, descending)
static void addSort(TritonOpBuilder &b, std::vector<Value> &ops) {
  if (ops.size() < 4)
    return;
  Value src = ops[1];
  auto dim = extractConstantInt(ops[2]);
  if (!dim)
    return;
  bool descending = extractConstantBool(ops[3]);
  auto &builder = b.getBuilder();
  auto loc = b.getLastLoc();
  auto op = builder.create<triton::ascend::SortOp>(
      loc, src, builder.getI64IntegerAttr(*dim),
      builder.getBoolAttr(descending));
  ops[0] = op->getResult(0);
}

// create_flip(src, dim)
static void addFlip(TritonOpBuilder &b, std::vector<Value> &ops) {
  if (ops.size() < 3)
    return;
  Value src = ops[1];
  auto dim = extractConstantInt(ops[2]);
  if (!dim)
    return;
  auto &builder = b.getBuilder();
  auto loc = b.getLastLoc();
  auto op = builder.create<triton::ascend::FlipOp>(
      loc, src, builder.getI64IntegerAttr(*dim));
  ops[0] = op->getResult(0);
}

// create_tanh(val)
static void addTanh(TritonOpBuilder &b, std::vector<Value> &ops) {
  if (ops.size() < 2)
    return;
  ops[0] = b.create<math::TanhOp>(ops[1]);
}

// create_annotation(ptr, key_string, i64_attr_value) — Attribute simplified
static void addAnnotation(TritonOpBuilder &b, std::vector<Value> &ops) {
  size_t idx = 1;
  if (idx >= ops.size())
    return;
  Value ptr = ops[idx++];
  std::string key;
  if (!decodeString(ops, idx, key))
    return;
  auto annotationOp = b.create<triton::ascend::AnnotationOp>(ptr);
  if (idx < ops.size()) {
    if (auto v = extractConstantInt(ops[idx])) {
      annotationOp->setAttr(b.getBuilder().getStringAttr(key),
                            b.getBuilder().getI64IntegerAttr(*v));
    }
  }
}

// create_fixpipe(src, [dst|null], dma, dual, pre_quant, pre_relu)
static void addFixpipe(TritonOpBuilder &b, std::vector<Value> &ops) {
  size_t idx = 1;
  if (ops.size() < 7)
    return;
  Value src = ops[idx++];
  Value dstObj = ops[idx++];
  auto dma = extractConstantInt(ops[idx++]);
  auto dual = extractConstantInt(ops[idx++]);
  auto preQ = extractConstantInt(ops[idx++]);
  auto preR = extractConstantInt(ops[idx++]);
  if (!dma || !dual || !preQ || !preR)
    return;
  if (!dyn_cast<RankedTensorType>(src.getType()))
    return;

  auto *ctx = b.getBuilder().getContext();
  bool needCreateDst = !dstObj;
  Value dstValue;
  if (needCreateDst) {
    auto srcType = cast<RankedTensorType>(src.getType());
    SmallVector<int64_t> dstShape(srcType.getShape().begin(),
                                  srcType.getShape().end());
    auto dualMode = static_cast<hivm::FixpipeDualDstMode>(*dual);
    if (dualMode == hivm::FixpipeDualDstMode::ROW_SPLIT && !dstShape.empty() &&
        dstShape[0] > 0)
      dstShape[0] /= 2;
    else if (dualMode == hivm::FixpipeDualDstMode::COLUMN_SPLIT &&
             dstShape.size() >= 2 && dstShape[1] > 0)
      dstShape[1] /= 2;
    dstValue = b.create<tensor::EmptyOp>(dstShape, srcType.getElementType());
  } else {
    dstValue = dstObj;
  }

  auto dmaAttr = hivm::FixpipeDMAModeAttr::get(
      ctx, static_cast<hivm::FixpipeDMAMode>(*dma));
  auto dualAttr = hivm::FixpipeDualDstModeAttr::get(
      ctx, static_cast<hivm::FixpipeDualDstMode>(*dual));
  auto preQAttr = hivm::FixpipePreQuantModeAttr::get(
      ctx, static_cast<hivm::FixpipePreQuantMode>(*preQ));
  auto preRAttr = hivm::FixpipePreReluModeAttr::get(
      ctx, static_cast<hivm::FixpipePreReluMode>(*preR));
  auto channelSplit = BoolAttr::get(ctx, false);
  if (needCreateDst) {
    ops[0] = b.create<hivm::FixpipeOp>(TypeRange{dstValue.getType()}, src,
                                       dstValue, dmaAttr, dualAttr, nullptr,
                                       preQAttr, preRAttr, channelSplit)
                 .getResult(0);
  } else {
    b.create<hivm::FixpipeOp>(TypeRange{}, src, dstValue, dmaAttr, dualAttr,
                              nullptr, preQAttr, preRAttr, channelSplit);
  }
}

// create_annotation_mark / create_debug_barrier(ptr, key, i64)
static void addAnnotationMarkLike(TritonOpBuilder &b, std::vector<Value> &ops) {
  size_t idx = 1;
  if (idx >= ops.size())
    return;
  Value ptr = ops[idx++];
  std::string key;
  if (!decodeString(ops, idx, key))
    return;
  auto annotationOp = b.create<annotation::MarkOp>(ptr);
  if (idx < ops.size()) {
    if (auto v = extractConstantInt(ops[idx])) {
      annotationOp->setAttr(b.getBuilder().getStringAttr(key),
                            b.getBuilder().getI64IntegerAttr(*v));
    }
  }
}

static void addAnnotationMark(TritonOpBuilder &b, std::vector<Value> &ops) {
  addAnnotationMarkLike(b, ops);
}

static void addDebugBarrier(TritonOpBuilder &b, std::vector<Value> &ops) {
  addAnnotationMarkLike(b, ops);
}

// create_bind_buffer(src, alloc) → void
static void addBindBuffer(TritonOpBuilder &b, std::vector<Value> &ops) {
  if (ops.size() < 3)
    return;
  auto *ctx = b.getBuilder().getContext();
  auto bind = StringAttr::get(ctx, "bind_buffer");
  b.create<annotation::MarkOp>(ops[1], ValueRange{ops[2]},
                               ArrayAttr::get(ctx, bind));
}

// create_custom_op / create_custom_macro_op
// Packing: name_str, ins[], outs[]  (attrs omitted → empty)
static void addCustomHivmOp(TritonOpBuilder &b, std::vector<Value> &ops,
                            bool macro) {
  size_t idx = 1;
  std::string name;
  if (!decodeString(ops, idx, name))
    return;
  SmallVector<Value> ins, outs;
  if (!decodeValueList(ops, idx, ins) || !decodeValueList(ops, idx, outs))
    return;
  TypeRange resTypes(outs);
  if (macro) {
    auto op = b.create<hivm::CustomMacroOp>(resTypes, name, ValueRange(ins),
                                            ValueRange(outs), ValueRange{},
                                            ValueRange{});
    if (!op->getResults().empty())
      ops[0] = op->getResult(0);
  } else {
    auto op = b.create<hivm::CustomOp>(resTypes, name, ValueRange(ins),
                                       ValueRange(outs), ValueRange{});
    if (!op->getResults().empty())
      ops[0] = op->getResult(0);
  }
}

static void addCustomOp(TritonOpBuilder &b, std::vector<Value> &ops) {
  addCustomHivmOp(b, ops, /*macro=*/false);
}

static void addCustomMacroOp(TritonOpBuilder &b, std::vector<Value> &ops) {
  addCustomHivmOp(b, ops, /*macro=*/true);
}

// create_scope_op: type carriers for result types (attrs omitted)
static void addScopeOp(TritonOpBuilder &b, std::vector<Value> &ops) {
  size_t idx = 1;
  SmallVector<Type> resultTypes;
  // Remaining operands are type carriers.
  for (; idx < ops.size(); ++idx)
    resultTypes.push_back(ops[idx].getType());
  auto scopeOp = b.create<scope::ScopeOp>(TypeRange(resultTypes));
  // Return the op's first result if any; else leave result slot empty.
  if (!scopeOp->getResults().empty())
    ops[0] = scopeOp->getResult(0);
}

// scope_return(operands...)
static void addScopeReturn(TritonOpBuilder &b, std::vector<Value> &ops) {
  SmallVector<Value> operands;
  for (size_t i = 1; i < ops.size(); ++i)
    operands.push_back(ops[i]);
  b.create<scope::ReturnOp>(ValueRange(operands));
}

// sync_block_set / wait: sender_str, receiver_str, id, sender_pipe_i32,
//                        receiver_pipe_i32
static void addSyncBlockSetOrWait(TritonOpBuilder &b, std::vector<Value> &ops,
                                  const char *opName) {
  size_t idx = 1;
  std::string sender, receiver;
  if (!decodeString(ops, idx, sender) || !decodeString(ops, idx, receiver))
    return;
  if (idx + 2 >= ops.size())
    return;
  Value id = ops[idx++];
  auto sp = extractConstantInt(ops[idx++]);
  auto rp = extractConstantInt(ops[idx++]);
  if (!sp || !rp)
    return;
  buildSyncBlockOp(b, opName, sender, receiver, id,
                   static_cast<hivm::PIPE>(*sp), static_cast<hivm::PIPE>(*rp));
}

static void addSyncBlockSet(TritonOpBuilder &b, std::vector<Value> &ops) {
  addSyncBlockSetOrWait(b, ops, "sync_block_set");
}

static void addSyncBlockWait(TritonOpBuilder &b, std::vector<Value> &ops) {
  addSyncBlockSetOrWait(b, ops, "sync_block_wait");
}

// sync_block_all(mode_str, event_id_i32)
static void addSyncBlockAll(TritonOpBuilder &b, std::vector<Value> &ops) {
  size_t idx = 1;
  std::string mode;
  if (!decodeString(ops, idx, mode) || idx >= ops.size())
    return;
  auto idOpt = extractConstantInt(ops[idx]);
  if (!idOpt)
    return;
  auto *ctx = b.getBuilder().getContext();
  hivm::SyncBlockModeAttr modeAttr;
  hivm::PipeAttr cubePipe, vectorPipe;
  getSyncBlockModeAndPipes(ctx, mode, modeAttr, cubePipe, vectorPipe);
  auto indexAttribute =
      IntegerAttr::get(IndexType::get(ctx), static_cast<int64_t>(*idOpt));
  b.create<hivm::SyncBlockOp>(modeAttr, indexAttribute, Value{}, cubePipe,
                              vectorPipe);
}

// create_get_sub_vec_id()
static void addGetSubVecId(TritonOpBuilder &b, std::vector<Value> &ops) {
  auto subBlockIdxOp = b.create<hivm::GetSubBlockIdxOp>();
  if (auto moduleOp = subBlockIdxOp->getParentOfType<ModuleOp>()) {
    moduleOp->setAttr("hivm.disable_auto_tile_and_bind_subblock",
                      UnitAttr::get(b.getBuilder().getContext()));
  }
  ops[0] = subBlockIdxOp;
}

// create_copy_buffer(src, dst) → void
static void addCopyBuffer(TritonOpBuilder &b, std::vector<Value> &ops) {
  if (ops.size() < 3)
    return;
  b.create<hivm::CopyOp>(TypeRange{}, ops[1], ops[2]);
}

// create_copy_tensor(src, dst) → result
static void addCopyTensor(TritonOpBuilder &b, std::vector<Value> &ops) {
  if (ops.size() < 3)
    return;
  ops[0] = b.create<hivm::CopyOp>(TypeRange{ops[2].getType()}, ops[1], ops[2])
               .getResult(0);
}

// create_convert_layout(src, type_carrier)
static void addConvertLayout(TritonOpBuilder &b, std::vector<Value> &ops) {
  if (ops.size() < 3)
    return;
  Value src = ops[1];
  Type memrefType = ops[2].getType();
  auto *ctx = b.getBuilder().getContext();
  ops[0] = b.create<hivm::ConvertLayoutOp>(
                 memrefType, src,
                 hivm::DataLayoutAttr::get(ctx, hivm::DataLayout::ND),
                 hivm::DataLayoutAttr::get(ctx, hivm::DataLayout::ND))
               .getResult();
}

// create_auto_overflow_assert(condition, message_str)
// Dual-package stand-in for the invasive host ir.cc binding that monorepo
// Ascend used to ship. Marks AssertOp with tt.auto_overflow_assert so
// DeviceAssertConverter can erase sanitize_overflow asserts.
static void addAutoOverflowAssert(TritonOpBuilder &b, std::vector<Value> &ops) {
  if (ops.size() < 2)
    return;
  Value condition = ops[1];
  size_t idx = 2;
  std::string message;
  if (!decodeString(ops, idx, message))
    message = "overflow detected";
  auto *ctx = b.getBuilder().getContext();
  auto messageAttr = StringAttr::get(ctx, message);
  auto assertOp = b.create<triton::AssertOp>(condition, messageAttr);
  assertOp->setAttr("tt.auto_overflow_assert", UnitAttr::get(ctx));
}

} // namespace

mlir::triton::plugin::OpInfo *getAscendOpInfoTable(size_t &numOps) {
  static mlir::triton::plugin::OpInfo kOps[] = {
      {"extract_scalar", addExtractScalar},
      {"extract_slice", addExtractSlice},
      {"insert_slice", addInsertSlice},
      {"custom_op_for_inter_core_sync", addCustomOpForInterCoreSync},
      {"index_select_simd", addIndexSelectSimd},
      {"index_put", addIndexPut},
      {"gather_out_to_ub", addGatherOutToUb},
      {"scatter_ub_to_out", addScatterUbToOut},
      {"conv1d", addConv1d},
      {"dot", addDot},
      {"sort", addSort},
      {"flip", addFlip},
      {"tanh", addTanh},
      {"annotation", addAnnotation},
      {"fixpipe", addFixpipe},
      {"annotation_mark", addAnnotationMark},
      {"bind_buffer", addBindBuffer},
      {"debug_barrier", addDebugBarrier},
      {"custom_op", addCustomOp},
      {"custom_macro_op", addCustomMacroOp},
      {"scope_op", addScopeOp},
      {"scope_return", addScopeReturn},
      {"sync_block_set", addSyncBlockSet},
      {"sync_block_wait", addSyncBlockWait},
      {"sync_block_all", addSyncBlockAll},
      {"get_sub_vec_id", addGetSubVecId},
      {"copy_buffer", addCopyBuffer},
      {"copy_tensor", addCopyTensor},
      {"convert_layout", addConvertLayout},
      {"auto_overflow_assert", addAutoOverflowAssert},
  };
  numOps = sizeof(kOps) / sizeof(kOps[0]);
  return kOps;
}

} // namespace mlir::triton::ascend
