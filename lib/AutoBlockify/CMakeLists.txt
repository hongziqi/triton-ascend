add_ascend_pass_library(AutoBlockify
  AutoBlockify.cpp
  RewriteOperation.cpp
  Utils.cpp

  DEPENDS
  AutoBlockifyPassIncGen

  LINK_LIBS
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  TritonIR
  TritonTransforms
  TritonAnalysis
  MLIRTransforms
  MLIRSupport
  MLIRSCFTransforms
  TritonAscendIR
)
