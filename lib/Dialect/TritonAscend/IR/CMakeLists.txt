if(TRITON_ASCEND_EXTENSION OR TRITON_ASCEND_STANDALONE_PLUGIN)
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
else()
  add_triton_library(TritonAscendIR
    TritonAscendAttrs.cpp
    TritonAscendDialect.cpp
    TritonAscendOps.cpp

    DEPENDS
    TritonAscendTableGen
    TritonAscendAttrDefsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRLLVMDialect
    TritonIR
  )
endif()
