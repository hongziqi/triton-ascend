// Ascend MLIR plugin entry (tritonGetPluginInfo).
// Registers dialects, OpInfo, and Ascend lowering PassInfo so
// TRITON_PLUGIN_PATHS binds PassInfo.name as passes.plugin.<name> (include add_ prefix on 3.7) (e.g. add_graph_optimize).
// Triton 3.7 Plugin ABI v2 via TRITON_PLUGIN_PATHS (no extend_with / HostExtend).

#include "ArgParser.h"
#include "AscendOpInfo.h"

#include "triton/Tools/PluginUtils.h"

#include "DiscreteMaskAccessConversion/DiscreteMaskAccessConversionPass.h"
#include "DynamicCVPipeline/AddDynamicCVPipeline.h"
#include "TritonControlFlowOpt/TritonControlFlowOptPass.h"
#include "TritonToAnnotation/Passes.h"
#include "TritonToGraph/GraphOptimization.h"
#include "TritonToGraph/Passes.h"
#include "TritonToHFusion/Passes.h"
#include "TritonToHIVM/Passes.h"
#include "TritonToLinalg/TritonToLinalgPass.h"
#include "TritonToLLVM/Passes.h"
#include "TritonToStructured/TritonToStructuredPass.h"
#include "TritonToUnstructure/BubbleUpOperation.h"
#include "TritonToUnstructure/UnstructureConversionPass.h"

#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "DynamicCVPipeline/Common/BufferCountManager.h"
#include "DynamicCVPipeline/Common/Utils.h"

#include "triton/Version.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include <memory>

using namespace mlir;
using namespace ascend_pass_plugin;

namespace {

static constexpr const char *kPluginVersion = "0.1.0";

static void registerTritonAscendDialect(DialectRegistry *registry) {
  registry->insert<triton::ascend::TritonAscendDialect>();
}

static void registerAscendNpuIrDialects(DialectRegistry *registry) {
  registry->insert<annotation::AnnotationDialect, hacc::HACCDialect,
                   hivm::HIVMDialect, scope::ScopeDialect>();
}

static void addGraphOptimize(PassManager *pm,
                             const std::vector<std::string> &args) {
  triton::cfg::GraphOptimizationOptions options;
  options.enabledRuleMask = static_cast<triton::cfg::GraphOptimizationRuleMask>(
      parseUInt64(args, "rule_mask", options.enabledRuleMask));
  options.maxRewritesPerFunction = static_cast<unsigned>(
      parseUInt64(args, "max_rewrites_per_function",
                  options.maxRewritesPerFunction));
  options.ubCapacityBytes = static_cast<unsigned>(
      parseUInt64(args, "ub_capacity_bytes", options.ubCapacityBytes));
  options.mappingUBCapacityBytes = static_cast<unsigned>(
      parseUInt64(args, "mapping_ub_capacity_bytes",
                  options.mappingUBCapacityBytes));
  options.storeCoalescingUBBudgetBytes = static_cast<unsigned>(
      parseUInt64(args, "store_coalescing_ub_budget_bytes",
                  options.storeCoalescingUBBudgetBytes));
  options.ubSafetyPercent = static_cast<unsigned>(
      parseUInt64(args, "ub_safety_percent", options.ubSafetyPercent));
  options.reservedUBBytes = static_cast<unsigned>(
      parseUInt64(args, "reserved_ub_bytes", options.reservedUBBytes));
  options.compileMode = parseString(args, "compile_mode", options.compileMode);
  pm->addPass(triton::cfg::createGraphOptimizePass(options));
}

static void addTritonControlFlowOpt(PassManager *pm,
                                    const std::vector<std::string> &) {
  pm->addPass(triton::createTritonControlFlowOptPass());
}

static void addTritonToStructure(PassManager *pm,
                                 const std::vector<std::string> &args) {
  bool enableMaskFallback =
      parseBool(args, "enable_mask_fallback_conversion", false);
  bool optimizeDynamicOffset =
      parseBool(args, "optimize_dynamic_offset", false);
  // TritonToStructuredPass only accepts the two options above; compile_on_910_95
  // is forwarded by backend/compiler.py for other passes and ignored here.
  (void)parseBool(args, "compile_on_910_95", false);
  pm->addPass(triton::createTritonToStructuredPass(
      enableMaskFallback, optimizeDynamicOffset));
}

static void
addDiscreteMaskAccessConversion(PassManager *pm,
                                const std::vector<std::string> &args) {
  DiscreteMaskAccessConversionOptions opts;
  opts.compileOn91095 = parseBool(args, "compile_on_910_95", false);
  opts.compileMode = parseString(args, "compile_mode", opts.compileMode);
  pm->addPass(triton::createDiscreteMaskAccessConversionPass(opts));
}

static void addTritonToAnnotation(PassManager *pm,
                                  const std::vector<std::string> &) {
  pm->addPass(triton::createTritonToAnnotationPass());
}

static void addTritonToUnstructure(PassManager *pm,
                                   const std::vector<std::string> &args) {
  TritonToUnstructureOptions opts;
  opts.compileOn91095 = parseBool(args, "compile_on_910_95", false);
  opts.compileMode = parseString(args, "compile_mode", opts.compileMode);
  pm->addPass(triton::createTritonToUnstructurePass(opts));
}

static void addTritonToHivm(PassManager *pm,
                            const std::vector<std::string> &) {
  pm->addPass(triton::createTritonToHIVMPass());
}

static void addTritonToHfusion(PassManager *pm,
                               const std::vector<std::string> &args) {
  bool compileOn91095 = parseBool(args, "compile_on_910_95", false);
  pm->addPass(triton::createTritonToHFusionPass(compileOn91095));
}

static void addTritonToLlvm(PassManager *pm,
                            const std::vector<std::string> &) {
  pm->addPass(triton::createTritonToLLVMPass());
}

static void addBubbleUpOperation(PassManager *pm,
                                 const std::vector<std::string> &args) {
  BubbleUpOperationOptions opts;
  opts.compileOn91095 = parseBool(args, "compile_on_910_95", false);
  pm->addPass(triton::createBubbleUpOperationPass(opts));
}

// Module attribute setters used to live on the in-tree pybind
// (set_buffer_count / set_enable_buffer_insert_optimization). They run as
// the first passes in the pipeline so later passes see the same attributes.
struct SetModuleAttrPass
    : public PassWrapper<SetModuleAttrPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetModuleAttrPass)

  StringRef getArgument() const final { return "ascend-set-module-attr"; }
  StringRef getDescription() const final {
    return "Set ssbuffer module attributes for DynamicCVPipeline";
  }

  std::string kind;
  int count = 0;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    if (kind == "insert") {
      module->setAttr(CVPipeline::kInsertionOptimization, builder.getUnitAttr());
      return;
    }
    BufferCountManager mgr(module);
    if (kind == "INTRA")
      mgr.setBufferCount(BufferCountManager::DepType::IntraCore, count);
    else if (kind == "INTER")
      mgr.setBufferCount(BufferCountManager::DepType::InterCore, count);
    else if (kind == "LOAD")
      mgr.setBufferCount(BufferCountManager::DepType::LoadStore, count);
  }
};

static void addSetEnableBufferInsert(PassManager *pm,
                                     const std::vector<std::string> &) {
  auto pass = std::make_unique<SetModuleAttrPass>();
  pass->kind = "insert";
  pm->addPass(std::move(pass));
}

static void addSetBufferCount(PassManager *pm,
                              const std::vector<std::string> &args) {
  auto pass = std::make_unique<SetModuleAttrPass>();
  pass->kind = parseString(args, "type", "INTRA");
  pass->count = static_cast<int>(parseUInt64(args, "count", 0));
  pm->addPass(std::move(pass));
}

static void addTritonToLinalg(PassManager *pm,
                              const std::vector<std::string> &args) {
  bool globalKernel = parseBool(args, "global_kernel", false);
  bool namedOps = parseBool(args, "named_ops", false);
  bool enableNd2nz = parseBool(args, "enable_nd2nz_on_vector", false);
  bool enableSelect = parseBool(args, "enable_select_analysis", true);
  bool compileOn91095 = parseBool(args, "compile_on_910_95", false);
  std::string compileMode =
      parseString(args, "compile_mode", "simd_simt_template");
  pm->addPass(triton::createTritonToLinalgPass(
      globalKernel, namedOps, enableNd2nz, enableSelect, compileOn91095,
      compileMode));
}

static void addMergeConcatLoadBuffer(PassManager *pm,
                                     const std::vector<std::string> &) {
  pm->addPass(triton::cfg::createMergeConcatLoadBufferPass());
}

static void addDynamicCvPipeline(PassManager *pm,
                                 const std::vector<std::string> &args) {
  AddDynamicCVPipelineOptions opts;
  opts.compileOn91095 = parseBool(args, "compile_on_910_95", false);
  pm->addPass(triton::createAddDynamicCVPipelinePass(opts));
}

static void addNormalizeDebugLineLocations(PassManager *pm,
                                           const std::vector<std::string> &) {
  pm->addPass(triton::createNormalizeDebugLineLocationsPass());
}

static void registerNoOp() {}

} // namespace

TRITON_PLUGIN_API mlir::triton::plugin::PluginInfo *tritonGetPluginInfo() {
  static mlir::triton::plugin::PassInfo passes[] = {
      {"add_graph_optimize", kPluginVersion, addGraphOptimize, registerNoOp},
      {"add_triton_control_flow_opt", kPluginVersion, addTritonControlFlowOpt,
       registerNoOp},
      {"add_triton_to_structure", kPluginVersion, addTritonToStructure,
       registerNoOp},
      {"add_discrete_mask_access_conversion", kPluginVersion,
       addDiscreteMaskAccessConversion, registerNoOp},
      {"add_triton_to_annotation", kPluginVersion, addTritonToAnnotation,
       registerNoOp},
      {"add_triton_to_unstructure", kPluginVersion, addTritonToUnstructure,
       registerNoOp},
      {"add_triton_to_hivm", kPluginVersion, addTritonToHivm, registerNoOp},
      {"add_triton_to_hfusion", kPluginVersion, addTritonToHfusion, registerNoOp},
      {"add_triton_to_llvm", kPluginVersion, addTritonToLlvm, registerNoOp},
      {"add_bubble_up_operation", kPluginVersion, addBubbleUpOperation,
       registerNoOp},
      {"add_set_enable_buffer_insert_optimization", kPluginVersion,
       addSetEnableBufferInsert, registerNoOp},
      {"add_set_buffer_count", kPluginVersion, addSetBufferCount, registerNoOp},
      {"add_triton_to_linalg", kPluginVersion, addTritonToLinalg, registerNoOp},
      {"add_merge_concat_load_buffer", kPluginVersion, addMergeConcatLoadBuffer,
       registerNoOp},
      {"add_dynamic_cv_pipeline", kPluginVersion, addDynamicCvPipeline,
       registerNoOp},
      {"add_normalize_debug_line_locations", kPluginVersion,
       addNormalizeDebugLineLocations, registerNoOp},
  };

  static mlir::triton::plugin::DialectInfo dialects[] = {
      {"triton_ascend", kPluginVersion, registerTritonAscendDialect},
      {"ascend_npu_ir", kPluginVersion, registerAscendNpuIrDialects},
  };

  size_t numOps = 0;
  auto *ops = mlir::triton::ascend::getAscendOpInfoTable(numOps);

  static mlir::triton::plugin::PluginInfo info = {
      TRITON_PLUGIN_API_VERSION,
      "TritonAscendPassPlugin",
      kPluginVersion,
      passes,
      sizeof(passes) / sizeof(passes[0]),
      dialects,
      sizeof(dialects) / sizeof(dialects[0]),
      ops,
      numOps,
      TRITON_VERSION,
  };
  return &info;
}
