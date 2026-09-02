# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Helper included by standalone/compile_sources.cmake (not a CMakeLists.txt).
# Defines add_ascend_{dialect,pass}_library as OBJECT targets linked into
# libtriton_ascend.so. Pass *order* is Python (backend/compiler.py).

function(add_ascend_dialect_library name)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})

  if(TRITON_ASCEND_PASSES_IN_PLUGIN)
    add_triton_object(${name} ${ARG_UNPARSED_ARGUMENTS})
    target_compile_options(${name} PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
    if(ARG_DEPENDS)
      add_dependencies(${name} ${ARG_DEPENDS})
    endif()
    if(ARG_LINK_LIBS)
      list(REMOVE_ITEM ARG_LINK_LIBS "PUBLIC" "PRIVATE" "INTERFACE")
      target_link_libraries(${name} PRIVATE ${ARG_LINK_LIBS})
    endif()
    set_property(GLOBAL APPEND PROPERTY ASCEND_DIALECT_PLUGIN_LIBS ${name})
  else()
    add_triton_library(${name} ${ARGN})
  endif()
endfunction()

function(add_ascend_pass_library name)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})

  if(TRITON_ASCEND_PASSES_IN_PLUGIN)
    add_triton_object(${name} ${ARG_UNPARSED_ARGUMENTS})
    target_compile_options(${name} PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
    if(ARG_DEPENDS)
      add_dependencies(${name} ${ARG_DEPENDS})
    endif()
    if(ARG_LINK_LIBS)
      list(REMOVE_ITEM ARG_LINK_LIBS "PUBLIC" "PRIVATE" "INTERFACE")
      # PRIVATE: headers/compile flags only — do not embed MLIR static archives in plugin.
      target_link_libraries(${name} PRIVATE ${ARG_LINK_LIBS})
    endif()
    set_property(GLOBAL APPEND PROPERTY ASCEND_PASS_PLUGIN_LIBS ${name})
  else()
    add_triton_library(${name} ${ARGN})
  endif()
endfunction()

# Link Ascend pass object files into bin tools (triton-opt, etc.) when passes
# are not part of libtriton.
function(attach_ascend_pass_objects target)
  if(NOT TRITON_ASCEND_PASSES_IN_PLUGIN)
    return()
  endif()
  get_property(_ascend_pass_plugin_libs GLOBAL PROPERTY ASCEND_PASS_PLUGIN_LIBS)
  foreach(_lib ${_ascend_pass_plugin_libs})
    if(TARGET ${_lib})
      target_sources(${target} PRIVATE $<TARGET_OBJECTS:${_lib}>)
    endif()
  endforeach()
  get_property(_ascend_dialect_plugin_libs GLOBAL PROPERTY ASCEND_DIALECT_PLUGIN_LIBS)
  foreach(_lib ${_ascend_dialect_plugin_libs})
    if(TARGET ${_lib})
      target_sources(${target} PRIVATE $<TARGET_OBJECTS:${_lib}>)
    endif()
  endforeach()
  # DynamicCVPipelineCommon is already in ASCEND_PASS_PLUGIN_LIBS when built
  # via add_ascend_pass_library; only attach separately for legacy layouts.
  get_property(_ascend_pass_plugin_libs_check GLOBAL PROPERTY ASCEND_PASS_PLUGIN_LIBS)
  if(TARGET DynamicCVPipelineCommon)
    list(FIND _ascend_pass_plugin_libs_check DynamicCVPipelineCommon _cv_idx)
    if(_cv_idx EQUAL -1)
      target_sources(${target} PRIVATE $<TARGET_OBJECTS:DynamicCVPipelineCommon>)
    endif()
  endif()
endfunction()

# Keep MLIR transform symbols in libtriton.so for dlopen'd pass plugin objects.
function(link_libtriton_for_ascend_pass_plugin triton_target)
  if(NOT TRITON_ASCEND_PASSES_IN_PLUGIN)
    return()
  endif()
  if(NOT TARGET ${triton_target})
    return()
  endif()
  set(_mlir_whole_archive
    MLIRLinalgTransforms
    MLIRLinalgTransformOps
    MLIRSCFTransforms
    MLIRSCFTransformOps
    MLIRTransformDialectTransforms
    MLIRBufferizationTransforms
    MLIRAffineTransforms
    MLIRTensorTransforms
    MLIRMemRefTransforms
    MLIRArithTransforms
    MLIRFuncTransforms
    MLIRGPUPasses
  )
  set(_present_whole_archive)
  foreach(_lib ${_mlir_whole_archive})
    if(TARGET ${_lib})
      list(APPEND _present_whole_archive ${_lib})
    endif()
  endforeach()
  if(_present_whole_archive)
    target_link_libraries(${triton_target} PRIVATE
      "-Wl,--whole-archive"
      ${_present_whole_archive}
      "-Wl,--no-whole-archive"
    )
  endif()
  if(UNIX AND NOT APPLE)
    target_link_options(${triton_target} PRIVATE LINKER:--export-dynamic)
  endif()
endfunction()
