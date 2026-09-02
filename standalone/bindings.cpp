// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
//
// nanobind entry for libtriton_ascend.so (compiled WITH RTTI).
//   triton_ascend.ir            — private MLIRContext / parse / pass_manager
//   triton_ascend.passes.ttir   — add_* (Python owns the pass order)
//
// This is not a pipeline and not the CANN op builder. Host TTIR still
// crosses as text (str(mod)). MLIR types live in ir.cpp (-fno-rtti).

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
#include <utility>

#if defined(TRITON_ASCEND_HAS_OWNED_PASSES) && TRITON_ASCEND_HAS_OWNED_PASSES
#include "ir.h"
#endif

namespace nb = nanobind;

namespace {

#if defined(TRITON_ASCEND_HAS_OWNED_PASSES) && TRITON_ASCEND_HAS_OWNED_PASSES
using triton_ascend::ir::Attribute;
using triton_ascend::ir::Builder;
using triton_ascend::ir::Context;
using triton_ascend::ir::Module;
using triton_ascend::ir::PassManager;

void bindOwnedIr(nb::module_ &ir) {
  nb::class_<Context>(ir, "context").def(nb::init<>());

  nb::class_<Attribute>(ir, "attribute");

  nb::class_<Module>(ir, "module")
      .def("__str__", [](Module &m) { return triton_ascend::ir::module_str(m); })
      .def(
          "set_attr",
          [](Module &m, const std::string &name, Attribute &attr) {
            triton_ascend::ir::module_set_attr(m, name, attr);
          },
          nb::arg("name"), nb::arg("attr"));

  ir.def(
      "parse",
      [](const std::string &mlir_text, Context &ctx) {
        return triton_ascend::ir::parse(mlir_text, ctx);
      },
      nb::arg("mlir_text"), nb::arg("context"), nb::keep_alive<0, 2>());

  ir.def(
      "get_int_attr",
      [](Module &mod, const std::string &name) -> nb::object {
        auto v = triton_ascend::ir::get_int_attr(mod, name);
        if (!v)
          return nb::none();
        return nb::cast(*v);
      },
      nb::arg("module"), nb::arg("name"));

  ir.def(
      "remove_attr",
      [](Module &mod, const std::string &name) {
        triton_ascend::ir::remove_attr(mod, name);
      },
      nb::arg("module"), nb::arg("name"));

  nb::class_<Builder>(ir, "ascendnpu_ir_builder")
      .def(nb::init<Context &, const std::string &, const std::string &>(),
           nb::arg("context"), nb::arg("target") = "",
           nb::arg("compile_mode") = "simd", nb::keep_alive<1, 2>())
      .def(
          "parse_attr",
          [](Builder &self, const std::string &value) {
            return triton_ascend::ir::parse_attr(self, value);
          },
          nb::arg("value"));

  nb::class_<PassManager>(ir, "pass_manager")
      .def(nb::init<Context &>(), nb::arg("context"), nb::keep_alive<1, 2>())
      .def("enable_debug", [](PassManager &self) { (void)self; })
      .def("get_pipeline_str", [](PassManager &) { return std::string(); })
      .def(
          "run",
          [](PassManager &self, Module &mod, const std::string & /*name*/) {
            nb::gil_scoped_release release;
            triton_ascend::ir::pass_manager_run(self, mod);
          },
          nb::arg("module"), nb::arg("name") = "");
}

void bindOwnedPassesTtir(nb::module_ &m) {
  m.def("add_triton_control_flow_opt",
        [](PassManager &pm) {
          triton_ascend::ir::add_triton_control_flow_opt(pm);
        });
  m.def(
      "add_triton_to_structure",
      [](PassManager &pm, bool enableMaskFallbackConversion,
         bool optimizeDynamicOffset) {
        triton_ascend::ir::add_triton_to_structure(
            pm, enableMaskFallbackConversion, optimizeDynamicOffset);
      },
      nb::arg("pm"), nb::arg("enable_mask_fallback_conversion") = false,
      nb::arg("optimize_dynamic_offset") = false);
  m.def(
      "add_discrete_mask_access_conversion",
      [](PassManager &pm, bool compileOn91095, const std::string &compileMode) {
        triton_ascend::ir::add_discrete_mask_access_conversion(
            pm, compileOn91095, compileMode);
      },
      nb::arg("pm"), nb::arg("compile_on_910_95") = false,
      nb::arg("compile_mode") = "simd_simt_template");
  m.def("add_triton_to_annotation", [](PassManager &pm) {
    triton_ascend::ir::add_triton_to_annotation(pm);
  });
  m.def(
      "add_triton_to_unstructure",
      [](PassManager &pm, bool compileOn91095, const std::string &compileMode) {
        triton_ascend::ir::add_triton_to_unstructure(pm, compileOn91095,
                                                        compileMode);
      },
      nb::arg("pm"), nb::arg("compile_on_910_95") = false,
      nb::arg("compile_mode") = "simd_simt_template");
  m.def("add_triton_to_hivm", [](PassManager &pm) {
    triton_ascend::ir::add_triton_to_hivm(pm);
  });
  m.def(
      "add_triton_to_hfusion",
      [](PassManager &pm, bool compileOn91095) {
        triton_ascend::ir::add_triton_to_hfusion(pm, compileOn91095);
      },
      nb::arg("pm"), nb::arg("compile_on_910_95") = false);
  m.def("add_triton_to_llvm", [](PassManager &pm) {
    triton_ascend::ir::add_triton_to_llvm(pm);
  });
  m.def("add_bubble_up_operation", [](PassManager &pm) {
    triton_ascend::ir::add_bubble_up_operation(pm);
  });
  m.def(
      "add_triton_to_linalg",
      [](PassManager &pm, bool globalKernel, bool namedOps,
         bool enableNd2nzOnVector, bool enableSelectAnalysis,
         bool compileOn91095, const std::string &compileMode) {
        triton_ascend::ir::add_triton_to_linalg(
            pm, globalKernel, namedOps, enableNd2nzOnVector,
            enableSelectAnalysis, compileOn91095, compileMode);
      },
      nb::arg("pm"), nb::arg("global_kernel") = false,
      nb::arg("named_ops") = false, nb::arg("enable_nd2nz_on_vector") = false,
      nb::arg("enable_select_analysis") = true,
      nb::arg("compile_on_910_95") = false,
      nb::arg("compile_mode") = "simd_simt_template");
  m.def("add_merge_concat_load_buffer", [](PassManager &pm) {
    triton_ascend::ir::add_merge_concat_load_buffer(pm);
  });
  m.def(
      "add_dynamic_cv_pipeline",
      [](PassManager &pm, bool compileOn91095) {
        triton_ascend::ir::add_dynamic_cv_pipeline(pm, compileOn91095);
      },
      nb::arg("pm"), nb::arg("compile_on_910_95") = false);
  m.def("add_normalize_debug_line_locations", [](PassManager &pm) {
    triton_ascend::ir::add_normalize_debug_line_locations(pm);
  });
  m.def(
      "add_graph_optimize",
      [](PassManager &pm, std::uint64_t ruleMask,
         std::uint64_t maxRewritesPerFunction, std::uint64_t ubCapacityBytes,
         const std::string &compileMode) {
        triton_ascend::ir::add_graph_optimize(
            pm, ruleMask, maxRewritesPerFunction, ubCapacityBytes, compileMode);
      },
      nb::arg("pm"), nb::arg("rule_mask") = 511,
      nb::arg("max_rewrites_per_function") = 64,
      nb::arg("ub_capacity_bytes") = 0,
      nb::arg("compile_mode") = "simd_simt_template");
  m.def(
      "set_enable_cube_block_merge",
      [](bool enable) {
        triton_ascend::ir::set_enable_cube_block_merge(enable);
      },
      nb::arg("enable"));
  m.def(
      "set_enable_buffer_insert_optimization",
      [](Module &mod) {
        triton_ascend::ir::set_enable_buffer_insert_optimization(mod);
      },
      nb::arg("module"));
  m.def(
      "set_buffer_count",
      [](Module &mod, const std::string &type, int count) {
        triton_ascend::ir::set_buffer_count(mod, type, count);
      },
      nb::arg("module"), nb::arg("type"), nb::arg("count"));
}
#endif

} // namespace

NB_MODULE(libtriton_ascend, m) {
  m.doc() = "Triton Ascend native: ir + passes.ttir";
  m.attr("__triton_ascend_self_contained__") = true;
  m.attr("__triton_ascend_has_owned_passes__") =
#if defined(TRITON_ASCEND_HAS_OWNED_PASSES) && TRITON_ASCEND_HAS_OWNED_PASSES
      true;
#else
      false;
#endif

  auto ir = m.def_submodule("ir", "Private MLIRContext (not host libtriton.ir)");
  auto passes = m.def_submodule("passes");
  auto ttir = passes.def_submodule("ttir");

#if defined(TRITON_ASCEND_HAS_OWNED_PASSES) && TRITON_ASCEND_HAS_OWNED_PASSES
  bindOwnedIr(ir);
  bindOwnedPassesTtir(ttir);
#else
  ir.attr("__self_contained_scaffold__") = true;
#endif
}
