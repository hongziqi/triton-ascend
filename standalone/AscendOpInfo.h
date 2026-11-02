#ifndef TRITON_ASCEND_PLUGIN_ASCEND_OP_INFO_H
#define TRITON_ASCEND_PLUGIN_ASCEND_OP_INFO_H

#include "triton/Tools/PluginUtils.h"
#include <cstddef>

namespace mlir::triton::ascend {

/// Community PluginInfo.ops table for Triton 3.7 ``TRITON_PLUGIN_PATHS``.
/// Each OpInfo.name must include the ``create_`` prefix (3.7 binds the
/// name as-is; 3.9+ auto-prefixes). operands[0] is the result slot.
mlir::triton::plugin::OpInfo *getAscendOpInfoTable(size_t &numOps);

} // namespace mlir::triton::ascend

#endif // TRITON_ASCEND_PLUGIN_ASCEND_OP_INFO_H
