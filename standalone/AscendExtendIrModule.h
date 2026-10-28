#ifndef TRITON_ASCEND_EXTEND_IR_MODULE_H
#define TRITON_ASCEND_EXTEND_IR_MODULE_H

#include "triton/Tools/PluginUtils.h"

/// API v3 HostExtendCallback: register every binding from ascend_ir.cc onto
/// the host nanobind \c ir module (via \c HostExtensionArgs::irModule).
void ascendExtendHost(mlir::triton::plugin::HostExtensionArgs *args);

#endif
