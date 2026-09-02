# Minimal LoopSplit plugin to validate triton-ext toolchain before Ascend passes.

cmake_minimum_required(VERSION 3.20)
project(loop_split_probe LANGUAGES C CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/triton-ext.cmake")
configure_triton_extension()

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Extension)
add_public_tablegen_target(MLIRLoopSplitProbeIncGen)
add_dependencies(mlir-generic-headers MLIRLoopSplitProbeIncGen)

add_library(${TRITON_EXT_NAME} SHARED
    src/LoopSplit.cpp
    "${TRITON_EXT_SUPPORT_DIR}/Export.cpp")
add_dependencies(${TRITON_EXT_NAME} MLIRLoopSplitProbeIncGen)

target_compile_definitions(${TRITON_EXT_NAME} PRIVATE
    TRITON_EXT_VERSION="${TRITON_EXT_VERSION}"
    TRITON_EXT_NAME=${TRITON_EXT_NAME}
    TRITON_EXT_CLASS=LoopBisectPass)

target_link_libraries(${TRITON_EXT_NAME} PRIVATE triton_ext::plugin)

set_target_properties(${TRITON_EXT_NAME} PROPERTIES
    PREFIX "lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

install(TARGETS ${TRITON_EXT_NAME} LIBRARY DESTINATION triton_loop_split_probe)
