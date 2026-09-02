# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Text→text helper for tests. Production pass order lives in ``backend/compiler.py``
(``ascend.passes.ttir.add_*`` on ``ascend.ir.pass_manager``).
"""

from __future__ import annotations

from typing import Any, Sequence


def _native():
    from triton_ascend import _load_native

    return _load_native()


def _as_pass_args(options: Sequence[str] | dict[str, Any] | None) -> dict[str, Any]:
    if options is None:
        return {}
    if isinstance(options, dict):
        return dict(options)
    out: dict[str, Any] = {}
    for item in options:
        if "=" not in item:
            continue
        key, _, value = item.partition("=")
        if value in ("true", "True", "1"):
            out[key] = True
        elif value in ("false", "False", "0"):
            out[key] = False
        else:
            out[key] = value
    return out


def run_ttir_to_linalg(
    mlir_text: str,
    *,
    named_ops: bool = False,
    compile_on_910_95: bool = False,
    compile_mode: str = "simd_simt_template",
    enable_select_analysis: bool = True,
    enable_dynamic_cv_pipeline: bool = False,
    enable_msdebug: bool = False,
    enable_cube_block_merge: bool = False,
    enable_graph_optimize: bool = False,
    ub_capacity_bytes: int = 0,
) -> str:
    """Parse TTIR text in the owned context and run ``passes.ttir`` in order."""
    native = _native()
    ir = native.ir
    ttir = native.passes.ttir
    ctx = ir.context()
    mod = ir.parse(mlir_text, ctx)
    pm = ir.pass_manager(ctx)

    if enable_graph_optimize:
        ttir.add_graph_optimize(
            pm,
            ub_capacity_bytes=int(ub_capacity_bytes),
            compile_mode=compile_mode,
        )
    ttir.add_triton_control_flow_opt(pm)
    ttir.add_triton_to_structure(
        pm,
        enable_mask_fallback_conversion=False,
        optimize_dynamic_offset=False,
    )
    ttir.add_discrete_mask_access_conversion(
        pm, compile_on_910_95=compile_on_910_95, compile_mode=compile_mode
    )
    ttir.add_triton_to_annotation(pm)
    ttir.add_triton_to_unstructure(
        pm, compile_on_910_95=compile_on_910_95, compile_mode=compile_mode
    )
    ttir.add_triton_to_hivm(pm)
    ttir.add_triton_to_hfusion(pm, compile_on_910_95=compile_on_910_95)
    ttir.add_triton_to_llvm(pm)
    ttir.add_bubble_up_operation(pm)
    ttir.add_triton_to_structure(
        pm,
        enable_mask_fallback_conversion=False,
        optimize_dynamic_offset=False,
    )
    ttir.add_triton_to_linalg(
        pm,
        global_kernel=False,
        named_ops=named_ops,
        enable_nd2nz_on_vector=False,
        enable_select_analysis=enable_select_analysis,
        compile_on_910_95=compile_on_910_95,
        compile_mode=compile_mode,
    )
    if compile_on_910_95:
        ttir.add_merge_concat_load_buffer(pm)
    if enable_dynamic_cv_pipeline:
        ttir.set_enable_cube_block_merge(enable_cube_block_merge)
        ttir.set_enable_buffer_insert_optimization(mod)
        ttir.add_dynamic_cv_pipeline(pm, compile_on_910_95=compile_on_910_95)
    if enable_msdebug:
        ttir.add_normalize_debug_line_locations(pm)

    pm.run(mod)
    return str(mod)


def ttir_to_linalg(
    mlir_text: str,
    *,
    options: Sequence[str] | dict[str, Any] | None = None,
) -> str:
    args = _as_pass_args(options)
    return run_ttir_to_linalg(
        mlir_text,
        named_ops=bool(args.get("named_ops", False)),
        compile_on_910_95=bool(args.get("compile_on_910_95", False)),
        compile_mode=str(args.get("compile_mode", "simd_simt_template")),
        enable_select_analysis=bool(args.get("enable_select_analysis", True)),
        enable_dynamic_cv_pipeline=bool(args.get("enable_dynamic_cv_pipeline", False)),
        enable_msdebug=bool(args.get("enable_msdebug", False)),
        enable_cube_block_merge=bool(args.get("enable_cube_block_merge", False)),
        enable_graph_optimize=bool(args.get("enable_graph_optimize", False)),
        ub_capacity_bytes=int(args.get("ub_capacity_bytes", 0) or 0),
    )


def lower_ttir_to_linalg(mlir_text: str, *, options: dict[str, Any] | None = None) -> str:
    return ttir_to_linalg(mlir_text, options=options)


__all__ = ["ttir_to_linalg", "run_ttir_to_linalg", "lower_ttir_to_linalg"]
