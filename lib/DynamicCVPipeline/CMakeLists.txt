file(GLOB STANDARDIZE_OP_PASS_SRC CONFIGURE_DEPENDS
  "StandardizeOp/*.cpp"
)
file(GLOB PLAN_COMPUTE_BLOCK_PASS_SRC CONFIGURE_DEPENDS
  "PlanComputeBlock/*.cpp"
)
file(GLOB COMMON_SRC CONFIGURE_DEPENDS
  "Common/*.cpp"
)
file(GLOB SPLIT_DATAFLOW_SRC_FILES CONFIGURE_DEPENDS
    "SplitDataflow/*.cpp"
)
file(GLOB SEPARATE_MEMORY_FROM_COMPUTE_SRC_FILES CONFIGURE_DEPENDS
  "SeparateMemoryFromCompute/*.cpp"
)

file(GLOB ALLOC_MULTI_CACHE_SRCS_FILES CONFIGURE_DEPENDS
  "AllocMultiCache/*.cpp"
)

file(GLOB ADD_CONTROLFLOW_CONDITION_SRC_FILES CONFIGURE_DEPENDS
    "AddControlFlowCondition/*.cpp"
)

file(GLOB Compute_BLOCK_OPT_SRC_FILES CONFIGURE_DEPENDS
    "ComputeBlockOpt/*.cpp"
    "ComputeBlockOpt/SplitIfByBlockId/*.cpp"
)

file(GLOB ANALYZE_DATAFLOW_SRC_FILES CONFIGURE_DEPENDS
    "AnalyzeDataFlow/*.cpp"
)

file(GLOB PRE_CHECK_AVAILABLE_SRC_FILES CONFIGURE_DEPENDS
    "PreCheckAvailable/*.cpp"
)

set(_dynamic_cv_pass_link_libs
  BiShengIRHIVMDialect
  BiShengIRScopeDialect
  BiShengIRAnnotationDialect
  MLIRLinalgTransformOps
  MLIRLinalgTransforms
  MLIRSCFTransformOps
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRTransforms
)

set(_dynamic_cv_pass_src
  AddDynamicCVPipeline.cpp
  PreCheckAvailable.cpp
  StandardizeOp.cpp
  ${STANDARDIZE_OP_PASS_SRC}
  PlanComputeBlock.cpp
  ComputeBlockOptPass.cpp
  AnalyzeDataFlow.cpp
  ${PLAN_COMPUTE_BLOCK_PASS_SRC}
  ${Compute_BLOCK_OPT_SRC_FILES}
  ${ANALYZE_DATAFLOW_SRC_FILES}
  ${PRE_CHECK_AVAILABLE_SRC_FILES}
  SplitDataflow.cpp
  SeparateMemoryFromComputePass.cpp
  SeparateMemoryFromCompute/MarkGMLoadPass.cpp
  AllocMultiCache.cpp
  AddControlFlowCondition.cpp
  ${SPLIT_DATAFLOW_SRC_FILES}
  ${SEPARATE_MEMORY_FROM_COMPUTE_SRC_FILES}
  ${ALLOC_MULTI_CACHE_SRCS_FILES}
  ${ADD_CONTROLFLOW_CONDITION_SRC_FILES}
  RemoveAttributes.cpp
)

if(TRITON_ASCEND_PASSES_IN_PLUGIN)
  # Module-attribute helpers for ascend.passes.ttir (set_buffer_count, etc.).
  # Keep out of TRITON_LIBS so host libtriton stays Ascend-free under EXTENSION.
  add_ascend_pass_library(DynamicCVPipelineCommon
    Common/BufferCountManager.cpp
    Common/HostHelpers.cpp
    LINK_LIBS
    BiShengIRScopeDialect
    MLIRIR
    MLIRSupport
  )

  set(_dynamic_cv_common_for_plugin ${COMMON_SRC})
  list(FILTER _dynamic_cv_common_for_plugin EXCLUDE REGEX ".*/(BufferCountManager|HostHelpers)\\.cpp$")

  add_ascend_pass_library(DynamicCVPipeline
    ${_dynamic_cv_pass_src}
    ${_dynamic_cv_common_for_plugin}
    DEPENDS
    AddDynamicCVPipelinePassIncGen
    LINK_LIBS
    DynamicCVPipelineCommon
    ${_dynamic_cv_pass_link_libs}
  )
else()
  add_triton_library(DynamicCVPipeline
    ${_dynamic_cv_pass_src}
    ${COMMON_SRC}
    DEPENDS
    AddDynamicCVPipelinePassIncGen
    LINK_LIBS
    ${_dynamic_cv_pass_link_libs}
  )
endif()
