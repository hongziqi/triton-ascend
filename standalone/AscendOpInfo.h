#ifndef TRITON_ASCEND_PLUGIN_ASCEND_OP_INFO_H
#define TRITON_ASCEND_PLUGIN_ASCEND_OP_INFO_H

#include "triton/Tools/PluginUtils.h"
#include <cstddef>

namespace mlir::triton::ascend {

/// Community PluginInfo.ops table for ``ir.builder.extend_with``.
/// Each OpInfo.name is without the ``create_`` prefix (host binds
/// ``create_<name>(list[Value])``). operands[0] is the result slot.
mlir::triton::plugin::OpInfo *getAscendOpInfoTable(size_t &numOps);

} // namespace mlir::triton::ascend

#endif // TRITON_ASCEND_PLUGIN_ASCEND_OP_INFO_H
