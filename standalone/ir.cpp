// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
//
// MLIR implementation for the private context. Compiled -fno-rtti to match
// LLVM 3.7 (LLVM_ENABLE_RTTI=OFF). Do not include this file from bindings.cpp.

#include "ir.h"

#include "DiscreteMaskAccessConversion/Passes.h"
#include "DynamicCVPipeline/Common/BufferCountManager.h"
#include "DynamicCVPipeline/Common/Utils.h"
#include "DynamicCVPipeline/Passes.h"
#include "TritonControlFlowOpt/Passes.h"
#include "TritonToAnnotation/Passes.h"
#include "TritonToGraph/GraphOptimization.h"
#include "TritonToHFusion/Passes.h"
#include "TritonToHIVM/Passes.h"
#include "TritonToLLVM/Passes.h"
#include "TritonToLinalg/Passes.h"
#include "TritonToStructured/Passes.h"
#include "TritonToUnstructure/Passes.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace mlir::triton::cfg {
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createMergeConcatLoadBufferPass();
}

namespace {

void loadDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::triton::TritonDialect>();
  registry.insert<mlir::annotation::AnnotationDialect>();
  registry.insert<mlir::hivm::HIVMDialect>();
  registry.insert<mlir::scope::ScopeDialect>();
  registry.insert<mlir::hacc::HACCDialect>();
  registry.insert<mlir::DLTIDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

} // namespace

namespace triton_ascend::ir {

struct Context::Impl {
  mlir::MLIRContext context;
  Impl() { loadDialects(context); }
};

struct Attribute::Impl {
  mlir::Attribute attr;
};

struct Module::Impl {
  mlir::OwningOpRef<mlir::ModuleOp> op;
};

struct Builder::Impl {
  mlir::OpBuilder builder;
  std::string target;
  std::string compile_mode;
  Impl(mlir::MLIRContext *ctx, std::string target, std::string compile_mode)
      : builder(ctx), target(std::move(target)),
        compile_mode(std::move(compile_mode)) {}
};

struct PassManager::Impl {
  mlir::PassManager pm;
  explicit Impl(mlir::MLIRContext *ctx) : pm(ctx) {}
};

Context::Context() : impl_(std::make_unique<Impl>()) {}
Context::~Context() = default;
Context::Context(Context &&) noexcept = default;
Context &Context::operator=(Context &&) noexcept = default;

Attribute::Attribute() : impl_(std::make_unique<Impl>()) {}
Attribute::~Attribute() = default;
Attribute::Attribute(Attribute &&) noexcept = default;
Attribute &Attribute::operator=(Attribute &&) noexcept = default;

Module::Module() : impl_(std::make_unique<Impl>()) {}
Module::~Module() = default;
Module::Module(Module &&) noexcept = default;
Module &Module::operator=(Module &&) noexcept = default;

Builder::Builder(Context &ctx, std::string target, std::string compile_mode)
    : impl_(std::make_unique<Impl>(&ctx.impl()->context, std::move(target),
                                   std::move(compile_mode))) {}
Builder::~Builder() = default;
Builder::Builder(Builder &&) noexcept = default;
Builder &Builder::operator=(Builder &&) noexcept = default;

PassManager::PassManager(Context &ctx)
    : impl_(std::make_unique<Impl>(&ctx.impl()->context)) {}
PassManager::~PassManager() = default;
PassManager::PassManager(PassManager &&) noexcept = default;
PassManager &PassManager::operator=(PassManager &&) noexcept = default;

Module parse(const std::string &mlir_text, Context &ctx) {
  Module mod;
  mod.impl()->op =
      mlir::parseSourceString<mlir::ModuleOp>(mlir_text, &ctx.impl()->context);
  if (!mod.impl()->op)
    throw std::runtime_error(
        "triton_ascend.ir.parse: failed to parse MLIR text");
  return mod;
}

std::string module_str(Module &mod) {
  if (!mod.impl()->op)
    return {};
  std::string out;
  llvm::raw_string_ostream os(out);
  mod.impl()->op->print(os);
  return out;
}

void module_set_attr(Module &mod, const std::string &name, Attribute &attr) {
  if (!mod.impl()->op)
    throw std::runtime_error("triton_ascend.ir.module.set_attr: empty module");
  (*mod.impl()->op)->setAttr(name, attr.impl()->attr);
}

std::optional<int64_t> get_int_attr(Module &mod, const std::string &name) {
  if (!mod.impl()->op)
    return std::nullopt;
  auto ret = (*mod.impl()->op)->getAttrOfType<mlir::IntegerAttr>(name);
  if (!ret)
    return std::nullopt;
  return ret.getInt();
}

void remove_attr(Module &mod, const std::string &name) {
  if (!mod.impl()->op)
    return;
  (*mod.impl()->op)->removeAttr(name);
}

Attribute parse_attr(Builder &builder, const std::string &value) {
  auto *ctx = builder.impl()->builder.getContext();
  ctx->allowUnregisteredDialects();
  mlir::Attribute attr = mlir::parseAttribute(value, ctx);
  if (!attr)
    throw std::runtime_error("triton_ascend.ir.parse_attr failed: " + value);
  Attribute out;
  out.impl()->attr = attr;
  return out;
}

void pass_manager_run(PassManager &pm, Module &mod) {
  if (!mod.impl()->op)
    throw std::runtime_error(
        "triton_ascend.ir.pass_manager.run: empty module");
  if (mlir::failed(pm.impl()->pm.run(*mod.impl()->op)))
    throw std::runtime_error("triton_ascend.passes: PassManager::run failed");
}

void add_triton_control_flow_opt(PassManager &pm) {
  pm.impl()->pm.addPass(mlir::triton::createTritonControlFlowOptPass());
}

void add_triton_to_structure(PassManager &pm, bool enableMaskFallbackConversion,
                             bool optimizeDynamicOffset) {
  pm.impl()->pm.addPass(mlir::triton::createTritonToStructuredPass(
      enableMaskFallbackConversion, optimizeDynamicOffset));
}

void add_discrete_mask_access_conversion(PassManager &pm, bool compileOn91095,
                                         const std::string &compileMode) {
  DiscreteMaskAccessConversionOptions opts;
  opts.compileOn91095 = compileOn91095;
  opts.compileMode = compileMode;
  pm.impl()->pm.addPass(
      mlir::triton::createDiscreteMaskAccessConversionPass(opts));
}

void add_triton_to_annotation(PassManager &pm) {
  pm.impl()->pm.addPass(mlir::triton::createTritonToAnnotationPass());
}

void add_triton_to_unstructure(PassManager &pm, bool compileOn91095,
                               const std::string &compileMode) {
  TritonToUnstructureOptions opts;
  opts.compileOn91095 = compileOn91095;
  opts.compileMode = compileMode;
  pm.impl()->pm.addPass(mlir::triton::createTritonToUnstructurePass(opts));
}

void add_triton_to_hivm(PassManager &pm) {
  pm.impl()->pm.addPass(mlir::triton::createTritonToHIVMPass());
}

void add_triton_to_hfusion(PassManager &pm, bool compileOn91095) {
  pm.impl()->pm.addPass(mlir::triton::createTritonToHFusionPass(compileOn91095));
}

void add_triton_to_llvm(PassManager &pm) {
  pm.impl()->pm.addPass(mlir::triton::createTritonToLLVMPass());
}

void add_bubble_up_operation(PassManager &pm) {
  pm.impl()->pm.addPass(mlir::triton::createBubbleUpOperationPass());
}

void add_triton_to_linalg(PassManager &pm, bool globalKernel, bool namedOps,
                          bool enableNd2nzOnVector, bool enableSelectAnalysis,
                          bool compileOn91095, const std::string &compileMode) {
  pm.impl()->pm.addPass(mlir::triton::createTritonToLinalgPass(
      globalKernel, namedOps, enableNd2nzOnVector, enableSelectAnalysis,
      compileOn91095, compileMode));
}

void add_merge_concat_load_buffer(PassManager &pm) {
  pm.impl()->pm.addPass(mlir::triton::cfg::createMergeConcatLoadBufferPass());
}

void add_dynamic_cv_pipeline(PassManager &pm, bool compileOn91095) {
  AddDynamicCVPipelineOptions opts;
  opts.compileOn91095 = compileOn91095;
  pm.impl()->pm.addPass(mlir::triton::createAddDynamicCVPipelinePass(opts));
}

void add_normalize_debug_line_locations(PassManager &pm) {
  pm.impl()->pm.addPass(mlir::triton::createNormalizeDebugLineLocationsPass());
}

void add_graph_optimize(PassManager &pm, std::uint64_t ruleMask,
                        std::uint64_t maxRewritesPerFunction,
                        std::uint64_t ubCapacityBytes,
                        const std::string &compileMode) {
  if (ruleMask > std::numeric_limits<std::uint16_t>::max())
    throw std::runtime_error("rule_mask must fit in uint16_t");
  mlir::triton::cfg::GraphOptimizationOptions options;
  options.enabledRuleMask = static_cast<std::uint16_t>(ruleMask);
  options.maxRewritesPerFunction =
      static_cast<unsigned>(maxRewritesPerFunction);
  options.ubCapacityBytes = static_cast<unsigned>(ubCapacityBytes);
  options.compileMode = compileMode;
  pm.impl()->pm.addPass(mlir::triton::cfg::createGraphOptimizePass(options));
}

void set_enable_cube_block_merge(bool enable) {
  mlir::CVPipeline::setEnableCubeBlockMerge(enable);
}

void set_enable_buffer_insert_optimization(Module &mod) {
  if (!mod.impl()->op)
    return;
  mlir::OpBuilder builder((*mod.impl()->op)->getContext());
  (*mod.impl()->op)
      ->setAttr(mlir::CVPipeline::kInsertionOptimization, builder.getUnitAttr());
}

void set_buffer_count(Module &mod, const std::string &type, int count) {
  if (!mod.impl()->op)
    return;
  mlir::triton::BufferCountManager mgr(*mod.impl()->op);
  if (type == "INTRA")
    mgr.setBufferCount(mlir::triton::BufferCountManager::DepType::IntraCore,
                       count);
  else if (type == "INTER")
    mgr.setBufferCount(mlir::triton::BufferCountManager::DepType::InterCore,
                       count);
  else if (type == "LOAD")
    mgr.setBufferCount(mlir::triton::BufferCountManager::DepType::LoadStore,
                       count);
}

} // namespace triton_ascend::ir
