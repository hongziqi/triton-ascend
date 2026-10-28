add_ascend_pass_library(DiscreteMaskAccessConversion
  DiscreteMaskAccessConversionPass.cpp

  DEPENDS
  DiscreteMaskAccessConversionPassIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonToLinalg
  TritonMemoryAccessAnalysis
  TritonToLinalg
)
