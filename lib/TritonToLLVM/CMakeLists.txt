add_ascend_pass_library(TritonToLLVM
  NormalizeDebugLineLocations.cpp
  TritonToLLVM.cpp

  DEPENDS
  TritonToLLVMConversionPassIncGen

  LINK_LIBS
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
)
