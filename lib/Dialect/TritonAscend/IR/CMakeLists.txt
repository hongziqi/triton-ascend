add_ascend_dialect_library(TritonAscendIR
  TritonAscendAttrs.cpp
  TritonAscendDialect.cpp
  TritonAscendOps.cpp

  DEPENDS
  TritonAscendTableGen
  TritonAscendAttrDefsIncGen

  LINK_LIBS
  MLIRIR
  MLIRLLVMDialect
  TritonIR
)
