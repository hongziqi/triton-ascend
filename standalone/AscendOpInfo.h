#ifndef TRITON_ASCEND_PLUGIN_ASCEND_OP_INFO_H
#define TRITON_ASCEND_PLUGIN_ASCEND_OP_INFO_H

#include "triton/Tools/PluginUtils.h"
#include <cstddef>

namespace mlir::triton::ascend {

/// Stock PluginInfo.ops entry (always empty for Ascend; builders use HostExtend).
mlir::triton::plugin::OpInfo *getAscendOpInfoTable(size_t &numOps);

} // namespace mlir::triton::ascend

#endif // TRITON_ASCEND_PLUGIN_ASCEND_OP_INFO_H
