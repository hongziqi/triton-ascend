# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Included by CMakeLists.txt (not a CMakeLists.txt of its own).
#
# Why this is a .cmake and not CMakeLists.txt:
#   cmake -S standalone needs one project() / CMakeLists.txt.
#   Triton dialect, BiShengIR, and Ascend pass sources live *outside*
#   this directory (third_party/triton, AscendNPU-IR, lib/, include/).
#   include() pulls those add_subdirectory() calls into the same project
#   without inventing a fake source tree. A nested CMakeLists.txt would
#   be for add_subdirectory(standalone/foo) with sources under foo/.
#
# Compiles into OBJECT libraries that CMakeLists.txt links into
# libtriton_ascend.so. Submodule third_party/triton is used as-is.

# Triton IR sources: community git submodule third_party/triton (v3.7.1).
# Compiled as-is into libtriton_ascend.so (private MLIRContext). No overlay,
# no apply of candy-dev's triton-ascend-3.7.0.patch onto community dialect.
set(_TRITON_SUBMODULE "${ASCEND_ROOT}/third_party/triton")
set(_TRITON_VENDOR_FALLBACK "${ASCEND_ROOT}/standalone/vendor_triton")
if(DEFINED ENV{TRITON_SRC_DIR} AND EXISTS "$ENV{TRITON_SRC_DIR}/include/triton/Dialect/Triton")
  set(TRITON_VENDOR_DIR "$ENV{TRITON_SRC_DIR}")
elseif(TRITON_ASCEND_USE_SUBMODULE_TRITON AND
       EXISTS "${_TRITON_SUBMODULE}/include/triton/Dialect/Triton")
  set(TRITON_VENDOR_DIR "${_TRITON_SUBMODULE}")
  message(STATUS "TritonIR from submodule as-is")
elseif(EXISTS "${_TRITON_VENDOR_FALLBACK}/include/triton/Dialect/Triton")
  set(TRITON_VENDOR_DIR "${_TRITON_VENDOR_FALLBACK}")
else()
  message(FATAL_ERROR "No Triton IR sources (submodule third_party/triton or standalone/vendor_triton)")
endif()
message(STATUS "triton_ascend TritonIR src=${TRITON_VENDOR_DIR}")
set(BISHENGIR_SRC_DIR "${ASCEND_ROOT}/AscendNPU-IR/bishengir")
set(BISHENGIR_MAIN_INCLUDE_DIR "${BISHENGIR_SRC_DIR}/include")
set(BISHENGIR_BINARY_DIR "${CMAKE_BINARY_DIR}/bishengir")
set(BISHENGIR_INCLUDE_DIR "${BISHENGIR_BINARY_DIR}/include")

set(BISHENGIR_BUILD_STANDALONE_IR_ONLY ON)
set(BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES ON)
set(BISHENGIR_ENABLE_TRITON_COMPILE 0)
set(BISHENGIR_ENABLE_TORCH_CONVERSIONS 0)
set(BISHENGIR_ENABLE_PM_CL_OPTIONS 0)
set(BISHENGIR_PUBLISH OFF)
set(MLIR_ENABLE_EXECUTION_ENGINE 0)
set(MLIR_INCLUDE_TESTS OFF)
set(MLIR_ENABLE_BINDINGS_PYTHON OFF)
add_definitions(-DBISHENGIR_BUILD_STANDALONE_IR_ONLY)

# Match BiShengIR tablegen `if(LLVM_MAJOR_VERSION_*_COMPATIBLE)` checks.
if(LLVM_VERSION_MAJOR GREATER_EQUAL 21)
  set(LLVM_MAJOR_VERSION_21_COMPATIBLE ON)
endif()
if(LLVM_VERSION_MAJOR GREATER_EQUAL 22)
  set(LLVM_MAJOR_VERSION_22_COMPATIBLE ON)
endif()
if(LLVM_VERSION_MAJOR GREATER_EQUAL 23)
  set(LLVM_MAJOR_VERSION_23_COMPATIBLE ON)
endif()
if(LLVM_VERSION_MAJOR GREATER_EQUAL 24)
  set(LLVM_MAJOR_VERSION_24_COMPATIBLE ON)
endif()

if(NOT TARGET mlir-headers)
  add_custom_target(mlir-headers)
endif()
if(NOT TARGET mlir-generic-headers)
  add_custom_target(mlir-generic-headers)
endif()
if(NOT TARGET mlir-doc)
  add_custom_target(mlir-doc)
endif()
if(NOT TARGET bishengir-doc)
  add_custom_target(bishengir-doc)
endif()

file(WRITE "${CMAKE_BINARY_DIR}/tablegen_compile_commands.yml" "")

list(INSERT CMAKE_MODULE_PATH 0 "${BISHENGIR_SRC_DIR}/cmake/modules")
include(AddBiShengIR)
include(BiShengIRVersion)

file(MAKE_DIRECTORY "${BISHENGIR_INCLUDE_DIR}/bishengir/Config")
file(MAKE_DIRECTORY "${BISHENGIR_INCLUDE_DIR}/bishengir/Version")
configure_file(
  "${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Config/bishengir-config.h.cmake"
  "${BISHENGIR_INCLUDE_DIR}/bishengir/Config/bishengir-config.h")
set(BISHENGIR_VERSION
    "${BISHENGIR_VERSION_MAJOR}.${BISHENGIR_VERSION_MINOR}.${BISHENGIR_VERSION_PATCHLEVEL}${BISHENGIR_VERSION_SUFFIX}")
string(TIMESTAMP BISHENGIR_BUILD_DATE "%Y-%m-%d")
configure_file(
  "${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Version/Version.inc.in"
  "${BISHENGIR_INCLUDE_DIR}/bishengir/Version/Version.inc")

# Hundreds of sources still #include "ascend/include/..." from the monorepo layout.
set(_ASCEND_SRC_COMPAT "${CMAKE_BINARY_DIR}/include_compat")
set(_ASCEND_GEN_COMPAT "${CMAKE_BINARY_DIR}/gen_compat")
file(MAKE_DIRECTORY "${_ASCEND_SRC_COMPAT}/ascend")
file(MAKE_DIRECTORY "${_ASCEND_GEN_COMPAT}/ascend")
file(CREATE_LINK "${ASCEND_ROOT}/include" "${_ASCEND_SRC_COMPAT}/ascend/include" SYMBOLIC)
file(CREATE_LINK "${CMAKE_BINARY_DIR}/include" "${_ASCEND_GEN_COMPAT}/ascend/include" SYMBOLIC)

include_directories(
  "${_ASCEND_SRC_COMPAT}"
  "${_ASCEND_GEN_COMPAT}"
  "${ASCEND_ROOT}"
  "${ASCEND_ROOT}/include"
  "${CMAKE_BINARY_DIR}/include"
  "${TRITON_VENDOR_DIR}"
  "${TRITON_VENDOR_DIR}/include"
  "${CMAKE_BINARY_DIR}/triton_inc"
  "${BISHENGIR_MAIN_INCLUDE_DIR}"
  "${BISHENGIR_INCLUDE_DIR}"
)

# ---------------------------------------------------------------------------
# BiShengIR tblgen tools + IR-only dialects
# ---------------------------------------------------------------------------
add_subdirectory(
  "${BISHENGIR_SRC_DIR}/tools/bishengir-target-spec-tblgen"
  "${CMAKE_BINARY_DIR}/bishengir/tools/bishengir-target-spec-tblgen")
set(BISHENGIR_TARGET_SPEC_TABLEGEN_EXE "${BISHENGIR_TARGET_SPEC_TABLEGEN_EXE}" CACHE INTERNAL "")
set(BISHENGIR_TARGET_SPEC_TABLEGEN_TARGET "${BISHENGIR_TARGET_SPEC_TABLEGEN_TARGET}" CACHE INTERNAL "")

add_subdirectory(
  "${BISHENGIR_SRC_DIR}/tools/bishengir-hfusion-ods-gen"
  "${CMAKE_BINARY_DIR}/bishengir/tools/bishengir-hfusion-ods-gen")

add_subdirectory(
  "${BISHENGIR_SRC_DIR}/include/bishengir/Interfaces"
  "${BISHENGIR_INCLUDE_DIR}/bishengir/Interfaces")

# Include tablegen first (all dialects), then IR implementations in dep order.
set(_BISHENG_INC_DIALECTS
  Annotation HACC HFusion HIVM MemRef MemRefExt Scope Symbol Tensor MathExt)
foreach(_d ${_BISHENG_INC_DIALECTS})
  add_subdirectory(
    "${BISHENGIR_SRC_DIR}/include/bishengir/Dialect/${_d}"
    "${BISHENGIR_INCLUDE_DIR}/bishengir/Dialect/${_d}")
endforeach()

add_subdirectory(
  "${BISHENGIR_SRC_DIR}/lib/Dialect/Annotation/IR"
  "${CMAKE_BINARY_DIR}/bishengir/lib/Dialect/Annotation/IR")
add_subdirectory(
  "${BISHENGIR_SRC_DIR}/lib/Dialect/Scope/IR"
  "${CMAKE_BINARY_DIR}/bishengir/lib/Dialect/Scope/IR")
add_subdirectory(
  "${BISHENGIR_SRC_DIR}/lib/Dialect/HACC/IR"
  "${CMAKE_BINARY_DIR}/bishengir/lib/Dialect/HACC/IR")
add_subdirectory(
  "${BISHENGIR_SRC_DIR}/lib/Dialect/HACC/Transforms"
  "${CMAKE_BINARY_DIR}/bishengir/lib/Dialect/HACC/Transforms")
add_subdirectory(
  "${BISHENGIR_SRC_DIR}/lib/Dialect/HACC/Utils"
  "${CMAKE_BINARY_DIR}/bishengir/lib/Dialect/HACC/Utils")
add_subdirectory(
  "${BISHENGIR_SRC_DIR}/lib/Dialect/MemRef/IR"
  "${CMAKE_BINARY_DIR}/bishengir/lib/Dialect/MemRef/IR")
add_subdirectory(
  "${BISHENGIR_SRC_DIR}/lib/Dialect/MemRefExt/IR"
  "${CMAKE_BINARY_DIR}/bishengir/lib/Dialect/MemRefExt/IR")
add_subdirectory(
  "${BISHENGIR_SRC_DIR}/lib/Dialect/Tensor/IR"
  "${CMAKE_BINARY_DIR}/bishengir/lib/Dialect/Tensor/IR")
add_subdirectory(
  "${BISHENGIR_SRC_DIR}/lib/Dialect/Symbol/IR"
  "${CMAKE_BINARY_DIR}/bishengir/lib/Dialect/Symbol/IR")
add_subdirectory(
  "${BISHENGIR_SRC_DIR}/lib/Dialect/Math"
  "${CMAKE_BINARY_DIR}/bishengir/lib/Dialect/Math")
add_subdirectory(
  "${BISHENGIR_SRC_DIR}/lib/Dialect/HFusion/IR"
  "${CMAKE_BINARY_DIR}/bishengir/lib/Dialect/HFusion/IR")
add_subdirectory(
  "${BISHENGIR_SRC_DIR}/lib/Dialect/Utils"
  "${CMAKE_BINARY_DIR}/bishengir/lib/Dialect/Utils")
add_subdirectory(
  "${BISHENGIR_SRC_DIR}/lib/Dialect/HIVM/IR"
  "${CMAKE_BINARY_DIR}/bishengir/lib/Dialect/HIVM/IR")

# ---------------------------------------------------------------------------
# Vendored Triton IR (no nvidia backend / no GPU Transforms avalanche)
# ---------------------------------------------------------------------------
add_subdirectory(
  "${TRITON_VENDOR_DIR}/include/triton/Dialect/Triton"
  "${CMAKE_BINARY_DIR}/triton_inc/triton/Dialect/Triton")
add_subdirectory(
  "${TRITON_VENDOR_DIR}/include/triton/Dialect/TritonGPU"
  "${CMAKE_BINARY_DIR}/triton_inc/triton/Dialect/TritonGPU")
add_subdirectory(
  "${TRITON_VENDOR_DIR}/include/triton/Dialect/TritonNvidiaGPU"
  "${CMAKE_BINARY_DIR}/triton_inc/triton/Dialect/TritonNvidiaGPU")
add_subdirectory(
  "${TRITON_VENDOR_DIR}/include/triton/Dialect/Gluon"
  "${CMAKE_BINARY_DIR}/triton_inc/triton/Dialect/Gluon")

add_subdirectory(
  "${TRITON_VENDOR_DIR}/third_party/f2reduce"
  "${CMAKE_BINARY_DIR}/triton_lib/f2reduce")
# Skip PluginUtils.cpp — it needs community python/src/ir.h (extend_with ABI).
add_triton_library(TritonTools
  "${TRITON_VENDOR_DIR}/lib/Tools/GenericSwizzling.cpp"
  "${TRITON_VENDOR_DIR}/lib/Tools/LayoutUtils.cpp"
  "${TRITON_VENDOR_DIR}/lib/Tools/LinearLayout.cpp"
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  f2reduce
)
add_subdirectory(
  "${TRITON_VENDOR_DIR}/lib/Dialect/Triton"
  "${CMAKE_BINARY_DIR}/triton_lib/Dialect/Triton")
# Traits.cpp includes TritonGPU Types.h / generated .inc. Community TritonIR
# only DEPENDS TritonTableGen; with Unix Makefiles -jN that races on a clean
# build (Types.h.inc missing). TritonGPUIR already lists these tablegen deps.
if(TARGET TritonIR)
  add_dependencies(TritonIR
    TritonGPUTableGen
    TritonGPUAttrDefsIncGen
    TritonGPUOpsEnumsIncGen
    TritonGPUCGAAttrIncGen
    TritonGPUTypeInterfacesIncGen
    TritonGPUOpInterfacesIncGen)
endif()
add_subdirectory(
  "${TRITON_VENDOR_DIR}/lib/Dialect/TritonGPU/IR"
  "${CMAKE_BINARY_DIR}/triton_lib/Dialect/TritonGPU/IR")
add_subdirectory(
  "${TRITON_VENDOR_DIR}/lib/Dialect/TritonNvidiaGPU/IR"
  "${CMAKE_BINARY_DIR}/triton_lib/Dialect/TritonNvidiaGPU/IR")
add_subdirectory(
  "${TRITON_VENDOR_DIR}/lib/Dialect/Gluon/IR"
  "${CMAKE_BINARY_DIR}/triton_lib/Dialect/Gluon/IR")
add_subdirectory(
  "${TRITON_VENDOR_DIR}/lib/Analysis"
  "${CMAKE_BINARY_DIR}/triton_lib/Analysis")

# Narrow GPU helper (full TritonGPUTransforms pulls NVWS).
add_triton_object(TritonGPUUtility
  "${TRITON_VENDOR_DIR}/lib/Dialect/TritonGPU/Transforms/Utility.cpp"
  DEPENDS
    TritonGPUTableGen TritonGPUAttrDefsIncGen TritonGPUTransformsIncGen
    TritonGPUCGAAttrIncGen TritonGPUOpInterfacesIncGen
  LINK_LIBS TritonIR TritonGPUIR TritonNvidiaGPUIR TritonAnalysis
)
target_include_directories(TritonGPUUtility PRIVATE
  "${TRITON_VENDOR_DIR}/include"
  "${CMAKE_BINARY_DIR}/triton_inc"
)

if(EXISTS "${TRITON_VENDOR_DIR}/lib/Dialect/TritonGPU/Transforms/DescriptorMemoryLayouts.cpp")
  add_triton_object(TritonGPUDescriptorLayouts
    "${TRITON_VENDOR_DIR}/lib/Dialect/TritonGPU/Transforms/DescriptorMemoryLayouts.cpp"
    DEPENDS
      TritonGPUTableGen TritonGPUAttrDefsIncGen TritonGPUCGAAttrIncGen
      TritonGPUOpInterfacesIncGen TritonNvidiaGPUTableGen
    LINK_LIBS TritonIR TritonGPUIR TritonNvidiaGPUIR TritonTools
  )
  target_include_directories(TritonGPUDescriptorLayouts PRIVATE
    "${TRITON_VENDOR_DIR}/include"
    "${CMAKE_BINARY_DIR}/triton_inc"
  )
endif()

if(EXISTS "${TRITON_VENDOR_DIR}/lib/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.cpp")
  add_triton_object(TritonNvidiaTMAUtilities
    "${TRITON_VENDOR_DIR}/lib/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.cpp"
    DEPENDS
      TritonGPUTableGen TritonGPUAttrDefsIncGen TritonNvidiaGPUTableGen
      TritonNvidiaGPUAttrDefsIncGen
    LINK_LIBS TritonIR TritonGPUIR TritonNvidiaGPUIR TritonTools
  )
  target_include_directories(TritonNvidiaTMAUtilities PRIVATE
    "${TRITON_VENDOR_DIR}/include"
    "${CMAKE_BINARY_DIR}/triton_inc"
  )
endif()

# ---------------------------------------------------------------------------
# Ascend tablegen + pass/dialect OBJECT libraries
# ---------------------------------------------------------------------------
set(TRITON_ASCEND_PASSES_IN_PLUGIN ON)
set(TRITON_ASCEND_STANDALONE_PLUGIN ON)
set(TRITON_ASCEND_OWNED_FROM_SOURCE ON)
set_property(GLOBAL PROPERTY ASCEND_PASS_PLUGIN_LIBS "")
set_property(GLOBAL PROPERTY ASCEND_DIALECT_PLUGIN_LIBS "")
add_compile_definitions(TRITON_ASCEND_PASSES_IN_PLUGIN=1)
include("${ASCEND_ROOT}/cmake/AscendPasses.cmake")

add_subdirectory("${ASCEND_ROOT}/include" "${CMAKE_BINARY_DIR}/include")
add_subdirectory("${ASCEND_ROOT}/lib" "${CMAKE_BINARY_DIR}/lib")

# ir.cpp: MLIR TUs must match LLVM_ENABLE_RTTI=OFF. Keep exceptions so
# this API can throw into nanobind. bindings.cpp stays on the MODULE
# target with RTTI enabled.
add_library(TritonAscendIr OBJECT
  "${CMAKE_CURRENT_SOURCE_DIR}/ir.cpp"
)
target_compile_options(TritonAscendIr PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>)
target_include_directories(TritonAscendIr PRIVATE
  "${ASCEND_ROOT}"
  "${ASCEND_ROOT}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/include"
  "${CMAKE_BINARY_DIR}/include_compat"
  "${CMAKE_BINARY_DIR}/gen_compat"
  "${ASCEND_ROOT}/AscendNPU-IR/bishengir/triton"
  "${ASCEND_ROOT}/AscendNPU-IR/bishengir/triton/include"
  "${CMAKE_BINARY_DIR}/triton_inc"
  "${ASCEND_ROOT}/AscendNPU-IR/bishengir/include"
  "${CMAKE_BINARY_DIR}/bishengir/include"
)
get_property(_owned_api_deps GLOBAL PROPERTY ASCEND_PASS_PLUGIN_LIBS)
get_property(_owned_api_dials GLOBAL PROPERTY ASCEND_DIALECT_PLUGIN_LIBS)
if(_owned_api_deps OR _owned_api_dials)
  add_dependencies(TritonAscendIr ${_owned_api_deps} ${_owned_api_dials})
endif()
foreach(_lib TritonIR TritonGPUIR TritonNvidiaGPUIR GluonIR TritonTools TritonAnalysis)
  if(TARGET ${_lib})
    add_dependencies(TritonAscendIr ${_lib})
  endif()
endforeach()

function(triton_ascend_link_object_libs dest)
  get_property(_pass_libs GLOBAL PROPERTY ASCEND_PASS_PLUGIN_LIBS)
  get_property(_dial_libs GLOBAL PROPERTY ASCEND_DIALECT_PLUGIN_LIBS)
  foreach(_lib ${_pass_libs} ${_dial_libs})
    if(TARGET ${_lib})
      target_sources(${dest} PRIVATE $<TARGET_OBJECTS:${_lib}>)
      add_dependencies(${dest} ${_lib})
    endif()
  endforeach()

  set(_triton_objs
    f2reduce TritonTools TritonIR TritonTransforms
    TritonGPUIR TritonNvidiaGPUIR GluonIR TritonAnalysis TritonGPUUtility
    TritonGPUDescriptorLayouts TritonNvidiaTMAUtilities TritonAscendIr
  )
  foreach(_lib ${_triton_objs})
    if(TARGET ${_lib})
      target_sources(${dest} PRIVATE $<TARGET_OBJECTS:${_lib}>)
      add_dependencies(${dest} ${_lib})
    endif()
  endforeach()

  set(_bisheng_libs
    BiShengIRAnnotationDialect
    BiShengIRScopeDialect
    BiShengIRHACCDialect
    BiShengIRHACCTransforms
    BiShengIRHACCUtils
    BiShengIRMemRefDialect
    BiShengIRMemRefExtDialect
    BiShengIRTensorDialect
    BiShengIRSymbolDialect
    BiShengIRMathExtDialect
    BiShengIRHFusionDialect
    BiShengIRDialectUtils
    BiShengIRHIVMDialect
  )
  foreach(_lib ${_bisheng_libs})
    if(TARGET obj.${_lib})
      target_sources(${dest} PRIVATE $<TARGET_OBJECTS:obj.${_lib}>)
      add_dependencies(${dest} obj.${_lib})
    elseif(TARGET ${_lib})
      get_target_property(_type ${_lib} TYPE)
      if(_type STREQUAL "OBJECT_LIBRARY")
        target_sources(${dest} PRIVATE $<TARGET_OBJECTS:${_lib}>)
        add_dependencies(${dest} ${_lib})
      endif()
    else()
      message(WARNING "compile_sources.cmake: missing ${_lib}")
    endif()
  endforeach()
endfunction()
