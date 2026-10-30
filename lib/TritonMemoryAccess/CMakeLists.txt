if(TRITON_ASCEND_PASSES_IN_PLUGIN)
  add_ascend_pass_library(TritonMemoryAccessCore
    OpFoldResultUtils.cpp

    LINK_LIBS
    MLIRArithDialect
    MLIRDialectUtils
    MLIRIR
    MLIRSupport
  )

  add_ascend_pass_library(TritonMemoryAccessAnalysis
    MaskAnalysis.cpp

    LINK_LIBS
    MLIRArithDialect
    MLIRDialectUtils
    MLIRIR
    MLIRMemRefDialect
    MLIRSupport
    MLIRTensorDialect
    MLIRTransforms
    TritonIR
    TritonMemoryAccessCore
  )
else()
  add_triton_library(TritonMemoryAccessCore
    OpFoldResultUtils.cpp

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRDialectUtils
    MLIRIR
    MLIRSupport
  )

  add_triton_library(TritonMemoryAccessAnalysis
    MaskAnalysis.cpp

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRDialectUtils
    MLIRIR
    MLIRMemRefDialect
    MLIRSupport
    MLIRTensorDialect
    MLIRTransforms
    TritonIR
    TritonMemoryAccessCore
  )
endif()
