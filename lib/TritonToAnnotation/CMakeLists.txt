add_ascend_pass_library(TritonToAnnotation
  TritonToAnnotation.cpp

  DEPENDS
  TritonToAnnotationConversionPassIncGen

  LINK_LIBS
  BiShengIRAnnotationDialect
  BiShengIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
)
