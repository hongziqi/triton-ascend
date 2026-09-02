if(TRITON_ASCEND_PASSES_IN_PLUGIN)
  add_ascend_pass_library(MLIRTritonNPUUtils
    Utils.cpp
    InterleaveOptimization.cpp
    DebugUtils.cpp

    LINK_LIBS
    MLIRIR
    MLIRLLVMDialect
    TritonIR
    TritonMemoryAccessAnalysis
    TritonMemoryAccessCore
  )
else()
  add_triton_library(MLIRTritonNPUUtils
    Utils.cpp
    InterleaveOptimization.cpp
    DebugUtils.cpp

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRLLVMDialect
    TritonIR
    TritonMemoryAccessAnalysis
    TritonMemoryAccessCore
  )
endif()
