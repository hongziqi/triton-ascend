#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

namespace mlir {
namespace CVPipeline {

static bool g_enableCubeBlockMerge = false;

void setEnableCubeBlockMerge(bool enable) { g_enableCubeBlockMerge = enable; }

bool isCubeBlockMergeEnabled() { return g_enableCubeBlockMerge; }

} // namespace CVPipeline
} // namespace mlir
