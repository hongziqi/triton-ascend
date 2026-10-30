# Scheme 1: out-of-tree Triton plugin.
# Public MLIR/LLVM symbols resolve to the host libtriton.so (one copy).
# Linalg/Tensor/Complex/Bufferization are not in upstream libtriton.so.
# Those four dialect archives are partial-linked into this .so; symbols the
# host already exports are rewritten to undefined so they still bind to
# libtriton.so. Do not link the MLIR CMake dependency closure.
#
# Community OpInfo path: NEVER compile ascend_ir.cc / AscendExtendIrModule /
# nanobind. create_* live in AscendOpInfo.cpp only.

cmake_minimum_required(VERSION 3.20)
project(triton_ascend_plugin LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT LLVM_SYSPATH)
  message(FATAL_ERROR "LLVM_SYSPATH must point at the LLVM/MLIR tree used to build host triton")
endif()

set(ASCEND_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "triton-ascend package root")
set(STANDALONE_OUTPUT_DIR "${ASCEND_ROOT}/_C" CACHE PATH "Where to place libtriton_ascend.so")

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

# AscendNPU-IR submodule stays clean at HEAD; adapter patch is applied at configure.
set(_APPLY_NPUIR_PATCH "${ASCEND_ROOT}/scripts/apply_npuir_patch.py")
if(NOT EXISTS "${_APPLY_NPUIR_PATCH}")
  message(FATAL_ERROR "Missing ${_APPLY_NPUIR_PATCH}")
endif()
execute_process(
  COMMAND "${Python_EXECUTABLE}" "${_APPLY_NPUIR_PATCH}"
  WORKING_DIRECTORY "${ASCEND_ROOT}"
  RESULT_VARIABLE _npuir_patch_rc
  ERROR_VARIABLE _npuir_patch_err
  OUTPUT_VARIABLE _npuir_patch_out)
if(NOT _npuir_patch_rc EQUAL 0)
  message(FATAL_ERROR
    "Failed to apply AscendNPU-IR adapter patch:\n${_npuir_patch_out}${_npuir_patch_err}")
endif()
if(_npuir_patch_out)
  message(STATUS "${_npuir_patch_out}")
endif()

# Clear pip build-isolation PYTHONPATH so host site-packages (triton) is visible.
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env --unset=PYTHONPATH --unset=PYTHONHOME
          "${Python_EXECUTABLE}" -c
          "import triton, pathlib; print(pathlib.Path(triton.__file__).resolve().parent)"
  OUTPUT_VARIABLE TRITON_WHEEL_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _triton_import_rc
  ERROR_VARIABLE _triton_import_err)
if(NOT _triton_import_rc EQUAL 0)
  message(FATAL_ERROR
    "Could not import triton with ${Python_EXECUTABLE}: ${_triton_import_err}\n"
    "Install host triton first, and prefer:\n"
    "  pip install -e . --no-build-isolation\n"
    "(default pip isolation sets PYTHONPATH and hides site-packages).")
endif()
set(TRITON_WHEEL_INCLUDE_DIR "${TRITON_WHEEL_DIR}/include")
find_library(TRITON_LIB NAMES triton PATHS "${TRITON_WHEEL_DIR}/_C" NO_DEFAULT_PATH REQUIRED)
message(STATUS "Host libtriton: ${TRITON_LIB}")
if(NOT EXISTS "${TRITON_WHEEL_INCLUDE_DIR}/triton/Tools/PluginUtils.h")
  message(FATAL_ERROR
    "Triton wheel is missing C++ headers at ${TRITON_WHEEL_INCLUDE_DIR}. "
    "Build the host with TRITON_EXT_ENABLED=1 so wheel_headers are installed.")
endif()

list(APPEND CMAKE_PREFIX_PATH "${LLVM_SYSPATH}")
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(SYSTEM ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS} "${TRITON_WHEEL_INCLUDE_DIR}")
add_definitions(${LLVM_DEFINITIONS})

if(NOT MSVC)
  set(TRITON_DISABLE_EH_RTTI_FLAGS "$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions;-fno-rtti>")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error -Wno-covered-switch-default -Wno-deprecated-declarations")
endif()

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

# Object-library shims. LINK_LIBS stay off the final .so: only TARGET_OBJECTS
# are absorbed, then the .so links libtriton.
function(add_triton_object name)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
  add_library(${name} OBJECT)
  target_sources(${name} PRIVATE ${ARG_UNPARSED_ARGUMENTS})
  if(ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()
  if(ARG_LINK_LIBS)
    target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
  endif()
endfunction()

function(add_triton_library name)
  add_triton_object(${name} ${ARGN})
  target_compile_options(${name} PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
endfunction()

function(add_triton_plugin name)
  add_triton_object(${name} ${ARGN})
endfunction()

# Host dialects are not compiled into this .so. Pass cmake still names them
# in LINK_LIBS; an INTERFACE target satisfies that without embedding TypeIDs.
foreach(_host_dep TritonIR TritonTransforms TritonAnalysis TritonGPUIR TritonNvidiaGPUIR GluonIR TritonTools)
  if(NOT TARGET ${_host_dep})
    add_library(${_host_dep} INTERFACE)
  endif()
endforeach()

include("${CMAKE_CURRENT_SOURCE_DIR}/PassObjects.cmake")

# Detect host API version for PluginInfo layout (extend field on v3).
set(TRITON_PLUGIN_API_VERSION 2)
file(READ "${TRITON_WHEEL_INCLUDE_DIR}/triton/Tools/PluginUtils.h" _plugin_utils_h)
if(_plugin_utils_h MATCHES "#define TRITON_PLUGIN_API_VERSION ([0-9]+)")
  set(TRITON_PLUGIN_API_VERSION "${CMAKE_MATCH_1}")
endif()
message(STATUS "Host TRITON_PLUGIN_API_VERSION=${TRITON_PLUGIN_API_VERSION}")
message(STATUS "Ascend plugin: OpInfo path (no HostExtend / ascend_ir.cc / nanobind)")

# Force community-only sources (NO ascend_ir.cc, NO AscendExtendIrModule, NO nanobind).
set(_ascend_plugin_sources AscendPassPlugin.cpp AscendOpInfo.cpp)
set(NANOBIND_INCLUDE_DIR "")
set(TRITON_NANOBIND_LIB "")


add_library(libtriton_ascend MODULE ${_ascend_plugin_sources})
triton_ascend_attach_plugin_objects(libtriton_ascend)
# AscendPassPlugin.cpp / AscendOpInfo.cpp need generated .inc headers.
set(_ascend_pass_incgen_deps
  DiscreteMaskAccessConversionPassIncGen
  AddDynamicCVPipelinePassIncGen
  TritonControlFlowOptPassIncGen
  TritonToAnnotationConversionPassIncGen
  TritonToGraphPassIncGen
  TritonToHFusionConversionPassIncGen
  TritonToHIVMConversionPassIncGen
  TritonToLinalgConversionPassIncGen
  TritonToLLVMConversionPassIncGen
  TritonToStructuredConversionPassIncGen
  TritonToUnstructureConversionPassIncGen
  TritonAscendTableGen
  TritonAscendAttrDefsIncGen
)
foreach(_dep ${_ascend_pass_incgen_deps})
  if(TARGET ${_dep})
    add_dependencies(libtriton_ascend ${_dep})
  endif()
endforeach()
target_include_directories(libtriton_ascend PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${ASCEND_ROOT}
  ${ASCEND_ROOT}/include
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_BINARY_DIR}/include_compat
  ${CMAKE_BINARY_DIR}/gen_compat
  ${ASCEND_ROOT}/AscendNPU-IR/bishengir/include
  ${CMAKE_BINARY_DIR}/bishengir/include
  ${TRITON_WHEEL_INCLUDE_DIR}/python/src
  ${Python_INCLUDE_DIRS}
)
# Pass / OpInfo TUs: no-EH/RTTI (community OpInfo, no nanobind).
target_compile_options(libtriton_ascend PRIVATE -fvisibility=hidden)
set_source_files_properties(
  AscendPassPlugin.cpp AscendOpInfo.cpp
  PROPERTIES COMPILE_OPTIONS "${TRITON_DISABLE_EH_RTTI_FLAGS}")
# Missing host dialects: partial-link + rewrite via link_ext_dialects.py
# (required at build time, not a test helper). See standalone/README.md.
# Do NOT whole-archive MLIRAffineDialect here: merging it with Linalg into one
# relocatable bloated FillOp ODS and broke named-ops FillOp::regionBuilder.
# LLVM24 MaybeAlign create_* overloads are handled by a thin AffineOps-only
# rewritten object below.
set(_ext_dialect_names
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRComplexDialect
  MLIRBufferizationDialect
)
set(_ext_dialect_archives)
set(_ext_dialect_archive_args)
foreach(_name ${_ext_dialect_names})
  set(_archive "${LLVM_SYSPATH}/lib/lib${_name}.a")
  if(NOT EXISTS "${_archive}")
    message(FATAL_ERROR "Missing dialect archive ${_archive}")
  endif()
  list(APPEND _ext_dialect_archives "${_archive}")
  list(APPEND _ext_dialect_archive_args --archive "${_archive}")
endforeach()
set(_ext_dialects_obj "${CMAKE_BINARY_DIR}/ext_dialects_unique.o")
add_custom_command(
  OUTPUT "${_ext_dialects_obj}"
  COMMAND "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/link_ext_dialects.py"
    --libtriton "${TRITON_LIB}"
    --output "${_ext_dialects_obj}"
    ${_ext_dialect_archive_args}
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/link_ext_dialects.py"
    ${_ext_dialect_archives}
    "${TRITON_LIB}"
  COMMENT "Link four MLIR dialects, binding shared symbols to libtriton"
  VERBATIM
)
set_source_files_properties("${_ext_dialects_obj}" PROPERTIES
  EXTERNAL_OBJECT TRUE
  GENERATED TRUE
)
target_sources(libtriton_ascend PRIVATE "${_ext_dialects_obj}")

# Optional thin AffineOps extract (MaybeAlign create_*).
# Default OFF: PyPI community libtriton already embeds Affine TypeIDs; linking a
# second AffineOps.cpp.o from patched LLVM leaves linalg.fill regions empty
# ('expects a non-empty block'). Enable only for EXT hosts that need MaybeAlign
# overloads and do NOT already export AffineDialect.
option(TRITON_ASCEND_LINK_AFFINE_OPS_THIN
  "Rewrite/link AffineOps.cpp.o for MaybeAlign (OFF for community Triton)" OFF)
if(TRITON_ASCEND_LINK_AFFINE_OPS_THIN)
  set(_affine_ops_dir "${CMAKE_BINARY_DIR}/affine_ops_extract")
  set(_affine_ops_a "${_affine_ops_dir}/libAffineOpsThin.a")
  set(_affine_ops_obj "${CMAKE_BINARY_DIR}/affine_ops_unique.o")
  add_custom_command(
    OUTPUT "${_affine_ops_obj}"
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${_affine_ops_dir}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_affine_ops_dir}"
    COMMAND ${CMAKE_COMMAND} -E chdir "${_affine_ops_dir}"
      ${CMAKE_AR} x "${LLVM_SYSPATH}/lib/libMLIRAffineDialect.a" AffineOps.cpp.o
    COMMAND ${CMAKE_COMMAND} -E chdir "${_affine_ops_dir}"
      ${CMAKE_AR} crs "${_affine_ops_a}" AffineOps.cpp.o
    COMMAND "${Python_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/link_ext_dialects.py"
      --libtriton "${TRITON_LIB}"
      --output "${_affine_ops_obj}"
      --archive "${_affine_ops_a}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/link_ext_dialects.py"
      "${LLVM_SYSPATH}/lib/libMLIRAffineDialect.a"
      "${TRITON_LIB}"
    COMMENT "Rewrite AffineOps.cpp.o (MaybeAlign create_*) against libtriton"
    VERBATIM
  )
  set_source_files_properties("${_affine_ops_obj}" PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  target_sources(libtriton_ascend PRIVATE "${_affine_ops_obj}")
endif()

# Non-whole-archive helpers: satisfy Ascend undefineds without embedding a
# second copy of host TypeIDs (host wins for any remaining overlap via DT_NEEDED).
# Keep this close to the 1-2 working set. Prefer plain .a (below) over MLIR
# CMake TARGETS so dialect TypeIDs stay solely in ext_dialects_unique.o.
# Extra archives below cover dynsyms host does not export (Index TypeID,
# TrackingListener vtable, bubbleUpPadSlice) without re-embedding the four
# missing dialects.
set(_ext_mlir_helpers
  MLIRTensorUtils
  MLIRTilingInterface
  MLIRTensorTilingInterfaceImpl
  MLIRSparseTensorDialect
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRLinalgTransformOps
  MLIRSCFTransformOps
  MLIRTransformDialect
  MLIRTransformDialectInterfaces
  MLIRTransformDialectUtils
  MLIRIndexDialect
  MLIRBufferizationTransforms
  MLIRAffineUtils
  MLIRAffineTransforms
  MLIRSCFTransforms
  MLIRSCFUtils
  MLIRMemRefTransforms
  MLIRTensorTransforms
  MLIRVectorTransforms
  MLIRVectorUtils
  MLIRVectorToSCF
  MLIRValueBoundsOpInterface
  MLIRDialectUtils
  MLIRTransformUtils
  MLIRAnalysis
  MLIRSupport
)
# Always use plain .a paths — never MLIR CMake TARGETS. Linking a TARGET
# expands INTERFACE_LINK_LIBRARIES and re-embeds Linalg/Bufferization/etc.,
# duplicating TypeIDs already provided by ext_dialects_unique.o (→ ~280MB .so
# and PropertyRef TypeID asserts). Ordinary .a links only pull needed members.
set(_ext_mlir_helper_libs)
foreach(_name ${_ext_mlir_helpers})
  set(_archive "${LLVM_SYSPATH}/lib/lib${_name}.a")
  if(EXISTS "${_archive}")
    list(APPEND _ext_mlir_helper_libs "${_archive}")
  else()
    message(WARNING "scheme-1 plugin: missing helper archive ${_archive}")
  endif()
endforeach()

# --start-group: plain .a archives are single-pass; LinalgTransformOps may
# need vtables from LinalgTransforms (and vice versa). Do not switch back to
# MLIR CMake TARGETS — that re-expands the dialect TypeID closure.
target_link_libraries(libtriton_ascend PRIVATE
  "-Wl,--no-as-needed"
  ${TRITON_LIB}
  "-Wl,--start-group"
  ${_ext_mlir_helper_libs}
  "-Wl,--end-group"
  "-Wl,--as-needed"
)
set_target_properties(libtriton_ascend PROPERTIES
  BUILD_RPATH "${TRITON_WHEEL_DIR}/_C"
  INSTALL_RPATH "$ORIGIN:$ORIGIN/../../triton/_C"
)
set_target_properties(libtriton_ascend PROPERTIES
  PREFIX ""
  OUTPUT_NAME "libtriton_ascend"
  LIBRARY_OUTPUT_DIRECTORY "${STANDALONE_OUTPUT_DIR}"
)

message(STATUS "scheme 1 plugin: link only ${TRITON_LIB}")
message(STATUS "scheme 1 plugin: output ${STANDALONE_OUTPUT_DIR}/libtriton_ascend.so")
