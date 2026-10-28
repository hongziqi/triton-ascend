if(TRITON_ASCEND_EXTENSION OR TRITON_ASCEND_STANDALONE_PLUGIN)
  add_ascend_dialect_library(TritonStructuredIR
    TritonStructuredOps.cpp
    TritonStructuredDialect.cpp

    DEPENDS
    TritonStructuredTableGen

    LINK_LIBS
    TritonIR
    MLIRIR
  )
else()
  add_triton_library(TritonStructuredIR
    TritonStructuredOps.cpp
    TritonStructuredDialect.cpp

    DEPENDS
    TritonStructuredTableGen

    LINK_LIBS PUBLIC
    TritonIR
    MLIRIR
  )
endif()
