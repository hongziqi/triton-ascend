# Ascend + BiShengIR objects for the scheme-1 plugin.
# Triton dialect sources are not compiled here. Their TypeIDs stay in libtriton.

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

# Sources still #include "ascend/include/..." from the monorepo layout.
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
  "${TRITON_WHEEL_INCLUDE_DIR}"
  "${BISHENGIR_MAIN_INCLUDE_DIR}"
  "${BISHENGIR_INCLUDE_DIR}"
)

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

set(_BISHENG_INC_DIALECTS
  Annotation HACC HFusion HIVM MemRef MemRefExt Scope Symbol Tensor MathExt)
foreach(_d ${_BISHENG_INC_DIALECTS})
  add_subdirectory(
    "${BISHENGIR_SRC_DIR}/include/bishengir/Dialect/${_d}"
    "${BISHENGIR_INCLUDE_DIR}/bishengir/Dialect/${_d}")
endforeach()

foreach(_pair
    "lib/Dialect/Annotation/IR"
    "lib/Dialect/Scope/IR"
    "lib/Dialect/HACC/IR"
    "lib/Dialect/HACC/Transforms"
    "lib/Dialect/HACC/Utils"
    "lib/Dialect/MemRef/IR"
    "lib/Dialect/MemRefExt/IR"
    "lib/Dialect/Tensor/IR"
    "lib/Dialect/Symbol/IR"
    "lib/Dialect/Math"
    "lib/Dialect/HFusion/IR"
    "lib/Dialect/Utils"
    "lib/Dialect/HIVM/IR")
  add_subdirectory(
    "${BISHENGIR_SRC_DIR}/${_pair}"
    "${CMAKE_BINARY_DIR}/bishengir/${_pair}")
endforeach()

set(TRITON_ASCEND_PASSES_IN_PLUGIN ON)
set(TRITON_ASCEND_PLUGIN_ONLY ON)
set(TRITON_ASCEND_STANDALONE_PLUGIN ON)
# Full Ascend dialects + lowering passes into libtriton_ascend.so.
set_property(GLOBAL PROPERTY ASCEND_PASS_PLUGIN_LIBS "")
set_property(GLOBAL PROPERTY ASCEND_DIALECT_PLUGIN_LIBS "")
add_compile_definitions(TRITON_ASCEND_PASSES_IN_PLUGIN=1)
include("${ASCEND_ROOT}/cmake/AscendPasses.cmake")

add_subdirectory("${ASCEND_ROOT}/include" "${CMAKE_BINARY_DIR}/include")
add_subdirectory("${ASCEND_ROOT}/lib" "${CMAKE_BINARY_DIR}/lib")
message(STATUS "scheme 1 plugin: full Ascend passes in libtriton_ascend.so")

function(triton_ascend_attach_plugin_objects dest)
  get_property(_pass_libs GLOBAL PROPERTY ASCEND_PASS_PLUGIN_LIBS)
  get_property(_dial_libs GLOBAL PROPERTY ASCEND_DIALECT_PLUGIN_LIBS)
  foreach(_lib ${_pass_libs} ${_dial_libs})
    if(TARGET ${_lib})
      target_sources(${dest} PRIVATE $<TARGET_OBJECTS:${_lib}>)
      add_dependencies(${dest} ${_lib})
    endif()
  endforeach()

  # Ascend-only dialects. Their TypeIDs live in this DSO and are registered
  # into the host context. Do not attach a second copy of TritonIR.
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
      message(WARNING "scheme-1 plugin: missing ${_lib}")
    endif()
  endforeach()
endfunction()
