add_ascend_dialect_library(TritonStructuredIR
  TritonStructuredOps.cpp
  TritonStructuredDialect.cpp

  DEPENDS
  TritonStructuredTableGen

  LINK_LIBS
  TritonIR
  MLIRIR
)
