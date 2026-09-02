# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Project entry for libtriton_ascend.so (cmake -S standalone -B <build>).
# Does not build community / fork libtriton.so.
#
# Layout:
#   CMakeLists.txt          this file — LLVM, nanobind, MODULE target
#   compile_sources.cmake   include() — TritonIR + BiShengIR + Ascend OBJECTs
#   bindings.cpp            nanobind (RTTI on)
#   ir.h / ir.cpp           private MLIRContext (RTTI off)

cmake_minimum_required(VERSION 3.20)
project(triton_ascend_standalone LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(NOT LLVM_SYSPATH)
  message(FATAL_ERROR "LLVM_SYSPATH must point at the LLVM/MLIR install used by host triton")
endif()

set(ASCEND_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "triton-ascend package root")
set(STANDALONE_OUTPUT_DIR "${ASCEND_ROOT}/_C" CACHE PATH "Where to place libtriton_ascend.so")
option(TRITON_ASCEND_OWNED_FROM_SOURCE
       "Compile Ascend/BiShengIR/TritonIR from this tree (default)" ON)
option(TRITON_ASCEND_USE_SUBMODULE_TRITON
       "Compile TritonIR from third_party/triton" ON)

list(APPEND CMAKE_PREFIX_PATH "${LLVM_SYSPATH}")
set(LLVM_DIR "${LLVM_SYSPATH}/lib/cmake/llvm")
set(MLIR_DIR "${LLVM_SYSPATH}/lib/cmake/mlir")
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
if(NOT nanobind_DIR AND DEFINED Python_SITELIB)
  set(nanobind_DIR "${Python_SITELIB}/nanobind/cmake")
elseif(NOT nanobind_DIR AND DEFINED Python3_SITELIB)
  set(nanobind_DIR "${Python3_SITELIB}/nanobind/cmake")
endif()
find_package(nanobind CONFIG REQUIRED)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

if(NOT MSVC)
  set(TRITON_DISABLE_EH_RTTI_FLAGS "$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions;-fno-rtti>")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error -Wno-covered-switch-default")
endif()

# BiShengIR / Ascend headers expect the same LLVM-compat macros as the monorepo.
if(LLVM_VERSION_MAJOR GREATER_EQUAL 21)
  add_definitions(-D__LLVM_MAJOR_VERSION_21_COMPATIBLE__)
endif()
if(LLVM_VERSION_MAJOR GREATER_EQUAL 22)
  add_definitions(-D__LLVM_MAJOR_VERSION_22_COMPATIBLE__)
endif()
if(LLVM_VERSION_MAJOR GREATER_EQUAL 23)
  add_definitions(-D__LLVM_MAJOR_VERSION_23_COMPATIBLE__)
endif()
if(LLVM_VERSION_MAJOR GREATER_EQUAL 24)
  add_definitions(-D__LLVM_MAJOR_VERSION_24_COMPATIBLE__)
endif()
if(LLVM_VERSION_MAJOR GREATER_EQUAL 22 AND NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif()
message(STATUS "triton_ascend standalone: LLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR}")

# Community cmake shims — OBJECT libraries, no libtriton plugin embed.
function(add_triton_object name)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
  add_library(${name} OBJECT)
  target_sources(${name} PRIVATE ${ARG_UNPARSED_ARGUMENTS})
  target_compile_options(${name} PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
  if(ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()
  if(ARG_LINK_LIBS)
    target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
  endif()
endfunction()

set_property(GLOBAL PROPERTY TRITON_LIBS "")
function(add_triton_library name)
  set_property(GLOBAL APPEND PROPERTY TRITON_LIBS ${name})
  add_triton_object(${name} ${ARGN})
  target_compile_options(${name} PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
endfunction()

set_property(GLOBAL PROPERTY TRITON_PLUGINS "")
function(add_triton_plugin name)
  set_property(GLOBAL APPEND PROPERTY TRITON_PLUGINS ${name})
  add_triton_object(${name} ${ARGN})
endfunction()

set(_ASCEND_NATIVE_SOURCES bindings.cpp)
set(_ASCEND_HAS_OWNED_PASSES_VALUE 0)

if(TRITON_ASCEND_OWNED_FROM_SOURCE)
  include("${CMAKE_CURRENT_SOURCE_DIR}/compile_sources.cmake")
  set(_ASCEND_HAS_OWNED_PASSES_VALUE 1)
endif()

add_library(libtriton_ascend MODULE ${_ASCEND_NATIVE_SOURCES})
if(TRITON_ASCEND_OWNED_FROM_SOURCE)
  triton_ascend_link_object_libs(libtriton_ascend)
  target_include_directories(libtriton_ascend PRIVATE
    ${ASCEND_ROOT}
    ${ASCEND_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_BINARY_DIR}/include_compat
    ${CMAKE_BINARY_DIR}/gen_compat
    ${ASCEND_ROOT}/AscendNPU-IR/bishengir/triton
    ${ASCEND_ROOT}/AscendNPU-IR/bishengir/triton/include
    ${CMAKE_BINARY_DIR}/triton_inc
    ${ASCEND_ROOT}/AscendNPU-IR/bishengir/include
    ${CMAKE_BINARY_DIR}/bishengir/include
  )
endif()

nanobind_build_library(nanobind-static)
target_link_libraries(libtriton_ascend PRIVATE nanobind-static Python::Module)

set(_ASCEND_MLIR_WHOLE
  MLIRLinalgDialect MLIRBufferizationDialect MLIRTensorDialect
  MLIRMemRefDialect MLIRAffineDialect MLIRSCFDialect
  MLIRArithDialect MLIRMathDialect MLIRFuncDialect
  MLIRLLVMDialect MLIRVectorDialect MLIRComplexDialect
  MLIRControlFlowDialect MLIRUBDialect
)
set(_ASCEND_MLIR_NORMAL
  MLIRIR MLIRParser MLIRPass MLIRSupport MLIRDialect MLIRTransforms
  MLIRAnalysis MLIRRewrite MLIRTransformUtils MLIRFuncTransforms
  MLIRLinalgTransforms MLIRSCFTransforms MLIRAffineTransforms
  MLIRTensorTransforms MLIRMemRefTransforms MLIRBufferizationTransforms
  MLIRArithTransforms MLIRMathTransforms
  MLIRSCFTransformOps MLIRLinalgTransformOps MLIRTransformDialect
  MLIRDLTIDialect
)

target_link_libraries(libtriton_ascend PRIVATE ${_ASCEND_MLIR_NORMAL})
if(UNIX AND NOT APPLE)
  target_link_libraries(libtriton_ascend PRIVATE
    "-Wl,--whole-archive" ${_ASCEND_MLIR_WHOLE} "-Wl,--no-whole-archive")
else()
  target_link_libraries(libtriton_ascend PRIVATE ${_ASCEND_MLIR_WHOLE})
endif()

target_compile_definitions(libtriton_ascend PRIVATE
  TRITON_ASCEND_SELF_CONTAINED=1
  TRITON_ASCEND_HAS_OWNED_PASSES=${_ASCEND_HAS_OWNED_PASSES_VALUE}
)

set_target_properties(libtriton_ascend PROPERTIES
  PREFIX ""
  OUTPUT_NAME "libtriton_ascend"
  LIBRARY_OUTPUT_DIRECTORY "${STANDALONE_OUTPUT_DIR}"
)

message(STATUS "triton_ascend standalone: LLVM=${LLVM_SYSPATH}")
message(STATUS "triton_ascend standalone: OWNED_FROM_SOURCE=${TRITON_ASCEND_OWNED_FROM_SOURCE}")
message(STATUS "triton_ascend standalone: HAS_OWNED_PASSES=${_ASCEND_HAS_OWNED_PASSES_VALUE}")
message(STATUS "triton_ascend standalone: output=${STANDALONE_OUTPUT_DIR}")
