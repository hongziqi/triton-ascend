// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
//
// Private MLIRContext for libtriton_ascend.so.
//
// This header must not include MLIR. bindings.cpp is compiled WITH RTTI
// (nanobind). LLVM 3.7 is LLVM_ENABLE_RTTI=OFF, so all MLIR types live in
// ir.cpp (-fno-rtti). Python still sees:
//   triton_ascend.ir.{context,module,pass_manager,ascendnpu_ir_builder}
//
// ascendnpu_ir_builder here is only parse_attr / set_attr (hacc.target).
// It is not a full CANN / TritonOpBuilder.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace triton_ascend::ir {

class Context {
public:
  Context();
  ~Context();
  Context(Context &&) noexcept;
  Context &operator=(Context &&) noexcept;
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  struct Impl;
  Impl *impl() { return impl_.get(); }

private:
  std::unique_ptr<Impl> impl_;
};

class Attribute {
public:
  Attribute();
  ~Attribute();
  Attribute(Attribute &&) noexcept;
  Attribute &operator=(Attribute &&) noexcept;
  Attribute(const Attribute &) = delete;
  Attribute &operator=(const Attribute &) = delete;
  struct Impl;
  Impl *impl() { return impl_.get(); }

private:
  std::unique_ptr<Impl> impl_;
};

class Module {
public:
  Module();
  ~Module();
  Module(Module &&) noexcept;
  Module &operator=(Module &&) noexcept;
  Module(const Module &) = delete;
  Module &operator=(const Module &) = delete;
  struct Impl;
  Impl *impl() { return impl_.get(); }

private:
  std::unique_ptr<Impl> impl_;
};

// Thin attribute parser, not a full op builder.
class Builder {
public:
  Builder(Context &ctx, std::string target, std::string compile_mode);
  ~Builder();
  Builder(Builder &&) noexcept;
  Builder &operator=(Builder &&) noexcept;
  Builder(const Builder &) = delete;
  Builder &operator=(const Builder &) = delete;
  struct Impl;
  Impl *impl() { return impl_.get(); }

private:
  std::unique_ptr<Impl> impl_;
};

class PassManager {
public:
  explicit PassManager(Context &ctx);
  ~PassManager();
  PassManager(PassManager &&) noexcept;
  PassManager &operator=(PassManager &&) noexcept;
  PassManager(const PassManager &) = delete;
  PassManager &operator=(const PassManager &) = delete;
  struct Impl;
  Impl *impl() { return impl_.get(); }

private:
  std::unique_ptr<Impl> impl_;
};

Module parse(const std::string &mlir_text, Context &ctx);
std::string module_str(Module &mod);
void module_set_attr(Module &mod, const std::string &name, Attribute &attr);
std::optional<int64_t> get_int_attr(Module &mod, const std::string &name);
void remove_attr(Module &mod, const std::string &name);
Attribute parse_attr(Builder &builder, const std::string &value);
void pass_manager_run(PassManager &pm, Module &mod);

void add_triton_control_flow_opt(PassManager &pm);
void add_triton_to_structure(PassManager &pm, bool enableMaskFallbackConversion,
                             bool optimizeDynamicOffset);
void add_discrete_mask_access_conversion(PassManager &pm, bool compileOn91095,
                                         const std::string &compileMode);
void add_triton_to_annotation(PassManager &pm);
void add_triton_to_unstructure(PassManager &pm, bool compileOn91095,
                               const std::string &compileMode);
void add_triton_to_hivm(PassManager &pm);
void add_triton_to_hfusion(PassManager &pm, bool compileOn91095);
void add_triton_to_llvm(PassManager &pm);
void add_bubble_up_operation(PassManager &pm);
void add_triton_to_linalg(PassManager &pm, bool globalKernel, bool namedOps,
                          bool enableNd2nzOnVector, bool enableSelectAnalysis,
                          bool compileOn91095, const std::string &compileMode);
void add_merge_concat_load_buffer(PassManager &pm);
void add_dynamic_cv_pipeline(PassManager &pm, bool compileOn91095);
void add_normalize_debug_line_locations(PassManager &pm);
void add_graph_optimize(PassManager &pm, std::uint64_t ruleMask,
                        std::uint64_t maxRewritesPerFunction,
                        std::uint64_t ubCapacityBytes,
                        const std::string &compileMode);
void set_enable_cube_block_merge(bool enable);
void set_enable_buffer_insert_optimization(Module &mod);
void set_buffer_count(Module &mod, const std::string &type, int count);

} // namespace triton_ascend::ir
