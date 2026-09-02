add_ascend_pass_library(TritonToHIVM
  TritonToHIVM.cpp

  DEPENDS
  TritonToHIVMConversionPassIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
)
