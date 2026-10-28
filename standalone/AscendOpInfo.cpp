// Ascend fills PluginInfo.ops with an empty table. Full builder surface is
// registered via HostExtendCallback (ascend_ir.cc → ir._ascend), not via
// community ir.builder.extend_with / OpInfo.

#include "AscendOpInfo.h"

namespace mlir::triton::ascend {

mlir::triton::plugin::OpInfo *getAscendOpInfoTable(size_t &numOps) {
  numOps = 0;
  return nullptr;
}

} // namespace mlir::triton::ascend
