#include "AscendExtendIrModule.h"

#include <nanobind/nanobind.h>

namespace py = nanobind;

// Defined in ascend_ir.cc — registers the full Ascend IR Python surface.
void init_ascend_ir(py::module_ &m);

void ascendExtendHost(mlir::triton::plugin::HostExtensionArgs *args) {
  if (!args || !args->irModule)
    return;
  auto *host = static_cast<py::module_ *>(args->irModule);

  // Per-DSO nanobind `internals` must join the host domain before any NB API.
  if (!py::register_module(*host)) {
    PyErr_Print();
    return;
  }

  if (py::hasattr(*host, "_ascend"))
    return;

  py::module_ sub =
      host->def_submodule("_ascend", "Ascend IR bindings (plugin)");
  init_ascend_ir(sub);
}
