/*
 * Compatibility helpers for Ascend passes against community Triton Load/Store.
 *
 * Community folded cache/evict into CachePolicyAttr and removed MakeTensorPtr
 * (and with it Load/Store boundaryCheck + Load padding).
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

/// Community Triton removed tensor-pointer types (MakeTensorPtr).
inline bool isTensorPointerType(Type) { return false; }

namespace ascend {
namespace lscompat {

inline CachePolicyAttr makeCachePolicy(MLIRContext *ctx, CacheModifier cache,
                                       EvictionPolicy evict) {
  return CachePolicyAttr::get(ctx, cache, evict);
}

template <typename LoadOrStoreOp>
inline CacheModifier getCache(LoadOrStoreOp op) {
  if (auto policy =
          dyn_cast_or_null<CachePolicyAttr>(op.getCachePolicyAttr()))
    return policy.getCacheModifier();
  return CacheModifier::NONE;
}

template <typename LoadOrStoreOp>
inline EvictionPolicy getEvict(LoadOrStoreOp op) {
  if (auto policy =
          dyn_cast_or_null<CachePolicyAttr>(op.getCachePolicyAttr()))
    return policy.getEvictionPolicy();
  return EvictionPolicy::NORMAL;
}

inline bool getIsVolatile(LoadOp op) { return op.getIsVolatile(); }

inline ArrayRef<int32_t> getBoundaryCheck(Operation *) { return {}; }

template <typename LoadOrStoreOp>
inline ArrayRef<int32_t> getBoundaryCheck(LoadOrStoreOp) {
  return {};
}

inline std::optional<PaddingOption> getPadding(LoadOp) { return std::nullopt; }

inline Attribute cachePolicyAttrOrNull(MLIRContext *ctx, CacheModifier cache,
                                       EvictionPolicy evict) {
  if (cache == CacheModifier::NONE && evict == EvictionPolicy::NORMAL)
    return Attribute();
  return makeCachePolicy(ctx, cache, evict);
}

template <typename OpBuilderT>
inline LoadOp createLoad(OpBuilderT &b, Location loc, Value ptr, Value mask,
                         Value other, CacheModifier cache, EvictionPolicy evict,
                         bool isVolatile) {
  Attribute policy = cachePolicyAttrOrNull(b.getContext(), cache, evict);
  return b.template create<LoadOp>(loc, ptr, mask, other, policy, isVolatile);
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
                         ArrayRef<int32_t> /*boundaryCheck*/,
                         std::optional<PaddingOption> /*padding*/,
                         CacheModifier cache, EvictionPolicy evict,
                         bool isVolatile) {
  return createLoad(b, loc, ptr, mask, other, cache, evict, isVolatile);
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
inline LoadOp createLoad(OpBuilderT &b, Location loc, Value ptr, Value mask,
                         Value other, ArrayRef<int32_t> /*boundaryCheck*/,
                         std::optional<PaddingOption> /*padding*/,
                         CacheModifier cache, EvictionPolicy evict,
                         bool isVolatile) {
  return createLoad(b, loc, ptr, mask, other, cache, evict, isVolatile);
}

template <typename OpBuilderT>
inline LoadOp createLoad(OpBuilderT &b, Location loc, Value ptr,
                         ArrayRef<int32_t> /*boundaryCheck*/,
                         std::optional<PaddingOption> /*padding*/,
                         CacheModifier cache, EvictionPolicy evict,
                         bool isVolatile) {
  return createLoad(b, loc, ptr, /*mask=*/Value(), /*other=*/Value(), cache,
                    evict, isVolatile);
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
                         ArrayRef<int32_t> /*boundaryCheck*/, std::nullptr_t) {
  return createLoad(b, loc, ptr, /*mask=*/Value(), /*other=*/Value(),
                    CacheModifier::NONE, EvictionPolicy::NORMAL,
                    /*isVolatile=*/false);
}

template <typename OpBuilderT>
inline StoreOp createStore(OpBuilderT &b, Location loc, Value ptr, Value value,
                           Value mask, CacheModifier cache,
                           EvictionPolicy evict) {
  Attribute policy = cachePolicyAttrOrNull(b.getContext(), cache, evict);
  return b.template create<StoreOp>(loc, ptr, value, mask, policy,
                                    /*ignore_cta=*/false);
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
                           Value mask, ArrayRef<int32_t> /*boundaryCheck*/,
                           CacheModifier cache, EvictionPolicy evict) {
  return createStore(b, loc, ptr, value, mask, cache, evict);
}

template <typename OpBuilderT>
inline StoreOp createStore(OpBuilderT &b, Location loc, Value ptr, Value value,
                           ArrayRef<int32_t> /*boundaryCheck*/,
                           CacheModifier cache, EvictionPolicy evict) {
  return createStore(b, loc, ptr, value, /*mask=*/Value(), cache, evict);
}

} // namespace lscompat
} // namespace ascend
} // namespace triton
} // namespace mlir

#endif // TRITONNPU_UTILS_COMMUNITY_LOAD_STORE_COMPAT_H
