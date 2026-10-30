# Triton-Ascend Pluginization 1-3 (Community OpInfo)

## Source layout (plugin package)

- Host `triton` from pip — **no** vendored `third_party/triton`.
- `AscendNPU-IR/` is a **git submodule** (pin `aea934a6`). Keep HEAD clean; build runs `scripts/apply_npuir_patch.py` to apply `patch/npuir_adapter_to_llvm_24.patch`.
- One-shot LLVM Ascend patch: `patch/llvm_patch_b010a18.patch` (not applied by plugin build).

## Goal

Standardize Ascend IR construction on the community plugin ABI:

- `passes.plugin.extend_with(so)` — PassInfo
- `ir.extend_dialects_with(so)` — DialectInfo (queued for `ir.load_dialects`)
- `ir.builder.extend_with(so)` — OpInfo → `create_<name>(list[Value])`

**Not used:** HostExtend / `extend_module_with` / `AscendNPUIROpBuilder` / `ascend_ir.cc` / nanobind in the plugin `.so`.

## Symbol injection (triton-shared / host dialects)

Upstream `libtriton.so` (built with `--as-needed`) does **not** export TypeIDs for Linalg, Tensor, Complex, or Bufferization. Ascend lowering needs those dialects.

**Scheme used here (`standalone/link_ext_dialects.py`):**

1. `ld -r --whole-archive` these dialect archives from `LLVM_SYSPATH` (prefer `ld.lld`; GNU `ld` can produce an empty relocatable object on this tree):
   - **Always:** Linalg, Tensor, Complex, Bufferization (missing from stock `libtriton`).
   - **Also:** Affine — host has Affine TypeIDs, but LLVM24 `create_*(…, MaybeAlign)` overloads are not exported; rewrite keeps TypeIDs bound to host while leaving the new overloads defined in the plugin.
2. Rewrite any global that `libtriton.so` already exports to `SHN_UNDEF` so the plugin binds those symbols to the host copy.
3. Keep dialect-local symbols defined inside `libtriton_ascend.so`.

**Helper `.a` link (critical):** always pass plain `${LLVM_SYSPATH}/lib/lib*.a` paths inside `-Wl,--start-group … --end-group`. **Never** link MLIR CMake TARGETS — that expands `INTERFACE_LINK_LIBRARIES` and re-embeds `LinalgDialect` / `BufferizationDialect` (duplicate TypeIDs → ~280MB `.so` and `PropertyRef` asserts). Do **not** put the four missing dialect `.a` files on the helper link line either (they already live in `ext_dialects_unique.o`).

Result: one TypeID per shared MLIR symbol (host wins); Ascend-only dialect TypeIDs live in the plugin DSO and register via DialectInfo.

## Load path (`triton_ascend/__init__.py`)

Idempotent:

```python
libtriton.passes.plugin.extend_with(so_s)
if hasattr(host_ir, "extend_dialects_with"):
    host_ir.extend_dialects_with(so_s)
host_ir.builder.extend_with(so_s)
# NO extend_module_with
```

`triton_ascend.ir` aliases stock host `ir` (no `ir._ascend` submodule).

## Backend (`backend/compiler.py`)

- `register_ascend_plugin_dialects()` → `register_ascend_native()`.
- `load_dialects(ctx)` → register plugin, then `ir.load_dialects(ctx)` so queued DialectInfo installs into the context.
- Grid tile attrs use `ir.builder(mod.context).parse_attr(...)` when available; soft-fail otherwise.
- `get_int_attr` soft-fails to `-1` when Ascend IR helpers are absent.

## OpInfo packing convention

Host prepends an empty `Value` as `operands[0]` (result slot), then calls `AddOpCallback`. Python wrappers pack non-Value args as `arith.constant` Values:

| Kind | Packing |
|------|---------|
| int32 / int64 / bool | `arith.constant` of that width / i1 |
| `vector[int]` | `i32` length, then that many constants |
| `vector[Value]` | `i32` length, then Values |
| optional Value | `i32` has_flag (0/1), then Value if present |
| Type | type-carrier Value (scalar/tensor whose type is the carrier) |
| string | `i32` length, then `i32` char codes |
| HIVM enum | `i32` ordinal from `HIVMAttrs.td` |

Registered names (host binds `create_<name>`):

`extract_scalar`, `extract_slice`, `insert_slice`, `custom_op_for_inter_core_sync`, `index_select_simd`, `index_put`, `gather_out_to_ub`, `scatter_ub_to_out`, `conv1d`, `dot`, `sort`, `flip`, `tanh`, `annotation`, `fixpipe`, `annotation_mark`, `bind_buffer`, `debug_barrier`, `custom_op`, `custom_macro_op`, `scope_op`, `scope_return`, `sync_block_set`, `sync_block_wait`, `sync_block_all`, `get_sub_vec_id`, `copy_buffer`, `copy_tensor`, `convert_layout`.

No `conv2d`: this tree’s `TritonAscendOps.td` has no `Conv2dOp`.

Legacy call sites that used `builder.sync_block_*` (no `create_` prefix) need thin Python aliases that pack and call `create_sync_block_*`.

## Files / dirs touched

| Path | Role |
|------|------|
| `standalone/AscendOpInfo.{h,cpp}` | Full OpInfo table (ported from `ascend_ir.cc` create_*) |
| `standalone/AscendPassPlugin.cpp` | Passes + dialects + ops; `extend=nullptr` |
| `standalone/CMakeLists.txt` | Always OpInfo-only; no ascend_ir/nanobind |
| `standalone/link_ext_dialects.py` | Host-symbol rewrite for Linalg/Tensor/Complex/Bufferization (+ Affine overloads) |
| `__init__.py` | Community extend_with path |
| `backend/compiler.py` | Dialect load + stock parse_attr |
| `language/cann/extension/builder.py` | `setup_unified_builder` no-op / packing wrappers |
| `PLUGINIZATION_1_3.md` | This note |

`ascend_ir.cc` / `AscendExtendIrModule.*` remain in-tree for reference but are **not** compiled into the puncture `.so`.

## Build & use

```bash
source ~/miniconda/etc/profile.d/conda.sh && conda activate plugin13
export LLVM_SYSPATH=/home/coder/bin/llvm-project-3.9-b010a18-patch
# optional: LD=$LLVM_SYSPATH/bin/ld.lld  — only if link_ext_dialects empty-.o with GNU ld
cd /home/coder/workspace/main/triton-ascend-plugin-1-3
cmake -S standalone -B build/plugin \
  -DLLVM_SYSPATH=$LLVM_SYSPATH \
  -DASCEND_ROOT=$PWD \
  -DSTANDALONE_OUTPUT_DIR=$PWD/_C \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/plugin -j8 --target libtriton_ascend
# Release builds may still carry dialect debug info from LLVM .a members:
strip -g _C/libtriton_ascend.so
# Prefer pip install (writes RECORD). If uninstall fails with empty RECORD, rm site-packages/triton_ascend* first.
pip install . --no-build-isolation
```

Verify:

```python
import triton, triton_ascend
from triton._C.libtriton import ir, passes
triton_ascend.register_ascend_native()
assert hasattr(ir.builder, "create_extract_scalar")
assert sum(1 for n in dir(ir.builder) if n.startswith("create_")) >= 160
assert hasattr(passes.plugin, "add_triton_to_linalg")
assert not hasattr(ir, "_ascend")
```

```bash
cd unittest/pytest_ut && TRITON_DEBUG=0 pytest test_add.py -q
# → 6 passed
```

## Known gaps

1. **Attribute-heavy helpers** (`get_pipe_attr`, Affine maps, rich attrs on mark/annotation) were HostExtend-only; some language paths soft-fail or need textual `parseAttribute` packing.
2. **`scope_op` OpState** — OpInfo returns a single Value; entering a scope region via OpState is incomplete vs HostExtend.
3. **`custom_op` multi-result** — only the first result is returned via `operands[0]`.
4. **`create_dot` overload** — Ascend DotOp is `create_dot(list[Value])`; stock `create_dot(a,b,acc,...)` remains a separate nanobind overload.
5. **`.so` size** — stripped ≈118MB (1-2 was ≈52MB) mainly from whole-archive dialect objects + helpers; further GC/`-ffunction-sections` is optional.
6. **Build tip** — set `LD=$LLVM_SYSPATH/bin/ld.lld` if `link_ext_dialects` yields an empty relocatable object with GNU `ld`.
7. **Community host** — leave `TRITON_ASCEND_LINK_AFFINE_OPS_THIN=OFF` (default). PyPI `libtriton` already has Affine; linking patched `AffineOps.cpp.o` causes empty `linalg.fill` regions.

## test_add status

**Green on plugin13** (community OpInfo path, no HostExtend):

```
create_extract_scalar True
create_* 167
add_triton_to_linalg True
has _ascend False
pytest test_add.py → 6 passed
```
