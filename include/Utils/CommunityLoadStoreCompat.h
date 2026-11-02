/*
 * Compatibility helpers for Ascend passes against community Triton 3.7 Load/Store.
 *
 * Triton 3.7 keeps separate CacheModifier / EvictionPolicy attrs (not the
 * later CachePolicyAttr fold). Overload surface mirrors the 3.9 compat header
 * so call sites shared with the plugin-1-3 tree keep compiling.
 */

#ifndef TRITONNPU_UTILS_COMMUNITY_LOAD_STORE_COMPAT_H
#define TRITONNPU_UTILS_COMMUNITY_LOAD_STORE_COMPAT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include <optional>

namespace mlir {
namespace triton {

inline bool isTensorPointerType(Type type) {
  if (auto ptrTy = dyn_cast<triton::PointerType>(type))
    return isa<RankedTensorType>(ptrTy.getPointeeType());
  return false;
}

namespace ascend {
namespace lscompat {

template <typename LoadOrStoreOp>
inline CacheModifier getCache(LoadOrStoreOp op) {
  if (auto attr = op.getCacheAttr())
    return attr.getValue();
  return CacheModifier::NONE;
}

template <typename LoadOrStoreOp>
inline EvictionPolicy getEvict(LoadOrStoreOp op) {
  if (auto attr = op.getEvictAttr())
    return attr.getValue();
  return EvictionPolicy::NORMAL;
}

inline bool getIsVolatile(LoadOp op) { return op.getIsVolatile(); }

inline ArrayRef<int32_t> getBoundaryCheck(Operation *) { return {}; }

template <typename LoadOrStoreOp>
inline ArrayRef<int32_t> getBoundaryCheck(LoadOrStoreOp op) {
  return op.getBoundaryCheck();
}

inline std::optional<PaddingOption> getPadding(LoadOp op) {
  if (auto attr = op.getPaddingAttr())
    return attr.getValue();
  return std::nullopt;
}

template <typename OpBuilderT>
inline LoadOp createLoad(OpBuilderT &b, Location loc, Value ptr, Value mask,
                         Value other, ArrayRef<int32_t> boundaryCheck,
                         std::optional<PaddingOption> padding,
                         CacheModifier cache, EvictionPolicy evict,
                         bool isVolatile) {
  return b.template create<LoadOp>(loc, ptr, mask, other, boundaryCheck,
                                   padding, cache, evict, isVolatile);
}

template <typename OpBuilderT>
inline LoadOp createLoad(OpBuilderT &b, Location loc, Value ptr, Value mask,
                         Value other, CacheModifier cache, EvictionPolicy evict,
                         bool isVolatile) {
  return createLoad(b, loc, ptr, mask, other, /*boundaryCheck=*/ArrayRef<int32_t>{},
                    /*padding=*/std::nullopt, cache, evict, isVolatile);
}

template <typename OpBuilderT>
inline LoadOp createLoad(OpBuilderT &b, Location loc, Value ptr, Value mask,
                         CacheModifier cache, EvictionPolicy evict,
                         bool isVolatile) {
  return createLoad(b, loc, ptr, mask, /*other=*/Value(), cache, evict,
                    isVolatile);
}

template <typename OpBuilderT>
inline LoadOp createLoad(OpBuilderT &b, Location loc, Type /*resultTy*/,
                         Value ptr, Value mask, Value other,
                         ArrayRef<int32_t> boundaryCheck,
                         std::optional<PaddingOption> padding,
                         CacheModifier cache, EvictionPolicy evict,
                         bool isVolatile) {
  return createLoad(b, loc, ptr, mask, other, boundaryCheck, padding, cache,
                    evict, isVolatile);
}

template <typename OpBuilderT>
inline LoadOp createLoad(OpBuilderT &b, Location loc, Type resultTy, Value ptr,
                         Value mask, Value other,
                         ArrayRef<int32_t> boundaryCheck, Attribute,
                         CacheModifier cache, EvictionPolicy evict,
                         bool isVolatile) {
  return createLoad(b, loc, resultTy, ptr, mask, other, boundaryCheck,
                    std::nullopt, cache, evict, isVolatile);
}

template <typename OpBuilderT>
inline LoadOp createLoad(OpBuilderT &b, Location loc, Value ptr,
                         ArrayRef<int32_t> boundaryCheck,
                         std::optional<PaddingOption> padding,
                         CacheModifier cache, EvictionPolicy evict,
                         bool isVolatile) {
  return createLoad(b, loc, ptr, /*mask=*/Value(), /*other=*/Value(),
                    boundaryCheck, padding, cache, evict, isVolatile);
}

template <typename OpBuilderT>
inline LoadOp createLoad(OpBuilderT &b, Location loc, Value ptr,
                         CacheModifier cache, EvictionPolicy evict,
                         bool isVolatile) {
  return createLoad(b, loc, ptr, /*mask=*/Value(), /*other=*/Value(), cache,
                    evict, isVolatile);
}

template <typename OpBuilderT>
inline LoadOp createLoad(OpBuilderT &b, Location loc, Value ptr,
                         std::nullptr_t, std::nullptr_t,
                         ArrayRef<int32_t> boundaryCheck, std::nullptr_t) {
  return createLoad(b, loc, ptr, /*mask=*/Value(), /*other=*/Value(),
                    boundaryCheck, /*padding=*/std::nullopt,
                    CacheModifier::NONE, EvictionPolicy::NORMAL,
                    /*isVolatile=*/false);
}

template <typename OpBuilderT>
inline StoreOp createStore(OpBuilderT &b, Location loc, Value ptr, Value value,
                           Value mask, ArrayRef<int32_t> boundaryCheck,
                           CacheModifier cache, EvictionPolicy evict) {
  // 3.7 has no dedicated builder with mask+boundaryCheck together; prefer
  // mask builder when a mask is present, else boundaryCheck builder.
  if (mask)
    return b.template create<StoreOp>(loc, ptr, value, mask, cache, evict);
  if (!boundaryCheck.empty())
    return b.template create<StoreOp>(loc, ptr, value, boundaryCheck, cache,
                                      evict);
  return b.template create<StoreOp>(loc, ptr, value, cache, evict);
}

template <typename OpBuilderT>
inline StoreOp createStore(OpBuilderT &b, Location loc, Value ptr, Value value,
                           Value mask, CacheModifier cache,
                           EvictionPolicy evict) {
  return createStore(b, loc, ptr, value, mask, /*boundaryCheck=*/ArrayRef<int32_t>{},
                     cache, evict);
}

template <typename OpBuilderT>
inline StoreOp createStore(OpBuilderT &b, Location loc, Value ptr, Value value,
                           Value mask) {
  return createStore(b, loc, ptr, value, mask, CacheModifier::NONE,
                     EvictionPolicy::NORMAL);
}

template <typename OpBuilderT>
inline StoreOp createStore(OpBuilderT &b, Location loc, Value ptr, Value value,
                           CacheModifier cache, EvictionPolicy evict) {
  return createStore(b, loc, ptr, value, /*mask=*/Value(), cache, evict);
}

template <typename OpBuilderT>
inline StoreOp createStore(OpBuilderT &b, Location loc, Value ptr, Value value,
                           ArrayRef<int32_t> boundaryCheck, CacheModifier cache,
                           EvictionPolicy evict) {
  return createStore(b, loc, ptr, value, /*mask=*/Value(), boundaryCheck, cache,
                     evict);
}

} // namespace lscompat
} // namespace ascend
} // namespace triton
} // namespace mlir

#endif // TRITONNPU_UTILS_COMMUNITY_LOAD_STORE_COMPAT_H
