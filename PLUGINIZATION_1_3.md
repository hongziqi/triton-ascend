# Triton-Ascend 插件化 1-3（社区 OpInfo）

## 源码布局（插件包）

- Host `triton` 来自 pip — **不**内嵌 `third_party/triton`。
- `AscendNPU-IR/` 是 **git submodule**（固定 `aea934a6`）。保持 HEAD 干净；构建时由 `scripts/apply_npuir_patch.py` 打上 `patch/npuir_adapter_to_llvm_24.patch`。
- 一次性 LLVM Ascend 补丁：`patch/llvm_patch_b010a18.patch`（插件构建不会自动应用）。

## 目标

将 Ascend IR 构造统一到社区插件 ABI：

- `passes.plugin.extend_with(so)` — PassInfo
- `ir.extend_dialects_with(so)` — DialectInfo（排队，供 `ir.load_dialects` 安装）
- `ir.builder.extend_with(so)` — OpInfo → `create_<name>(list[Value])`

**不使用：** HostExtend / `extend_module_with` / `AscendNPUIROpBuilder` / `ascend_ir.cc` / 插件 `.so` 内的 nanobind。

## 符号注入（triton-shared / host dialects）

上游 `libtriton.so`（按 `--as-needed` 构建）**不会**导出 Linalg、Tensor、Complex、Bufferization 的 TypeID。Ascend lowering 需要这些 dialect。

**本方案（`standalone/link_ext_dialects.py`）：**

1. 从 `LLVM_SYSPATH` 用 `ld -r --whole-archive` 拉入这些 dialect 静态库（优先 `ld.lld`；本树上 GNU `ld` 可能产出空的 relocatable object）：
   - **始终包含：** Linalg、Tensor、Complex、Bufferization（库存 `libtriton` 缺失）。
   - **另外包含：** Affine — host 已有 Affine TypeID，但 LLVM24 的 `create_*(…, MaybeAlign)` 重载未导出；改写后 TypeID 仍绑定 host，新重载定义在插件内。
2. 将 `libtriton.so` 已导出的全局符号改写为 `SHN_UNDEF`，使插件绑定到 host 副本。
3. dialect 本地符号仍定义在 `libtriton_ascend.so` 内。

**Helper `.a` 链接（关键）：** 始终在 `-Wl,--start-group … --end-group` 中传入普通路径 `${LLVM_SYSPATH}/lib/lib*.a`。**切勿**链接 MLIR CMake TARGET — 会展开 `INTERFACE_LINK_LIBRARIES` 并再次嵌入 `LinalgDialect` / `BufferizationDialect`（重复 TypeID → 约 280MB `.so` 以及 `PropertyRef` assert）。那四个缺失 dialect 的 `.a` **也不要**再放进 helper 链接行（它们已在 `ext_dialects_unique.o` 中）。

结果：每个共享 MLIR 符号只有一个 TypeID（host 胜出）；仅 Ascend 的 dialect TypeID 留在插件 DSO 中，经 DialectInfo 注册。

## 加载路径（`triton_ascend/__init__.py`）

幂等：

```python
libtriton.passes.plugin.extend_with(so_s)
if hasattr(host_ir, "extend_dialects_with"):
    host_ir.extend_dialects_with(so_s)
host_ir.builder.extend_with(so_s)
# NO extend_module_with
```

`triton_ascend.ir` 别名到库存 host `ir`（无 `ir._ascend` 子模块）。

## 后端（`backend/compiler.py`）

- `register_ascend_plugin_dialects()` → `register_ascend_native()`。
- `load_dialects(ctx)` → 先注册插件，再 `ir.load_dialects(ctx)`，使排队的 DialectInfo 安装进 context。
- Grid tile attrs 在可用时用 `ir.builder(mod.context).parse_attr(...)`；否则 soft-fail。
- 缺少 Ascend IR helpers 时，`get_int_attr` soft-fail 为 `-1`。

## OpInfo 打包约定

Host 在 `operands[0]` 前插入空 `Value`（结果槽），再调用 `AddOpCallback`。Python 包装把非 Value 参数打成 `arith.constant` Values：

| 类型 | 打包方式 |
|------|---------|
| int32 / int64 / bool | 对应位宽 / i1 的 `arith.constant` |
| `vector[int]` | `i32` 长度，后跟等量 constants |
| `vector[Value]` | `i32` 长度，后跟 Values |
| optional Value | `i32` has_flag（0/1），有则再跟 Value |
| Type | type-carrier Value（标量/张量，类型即载体） |
| string | `i32` 长度，后跟各字符的 `i32` 码 |
| HIVM enum | 来自 `HIVMAttrs.td` 的 `i32` 序数 |

已注册名称（host 绑定 `create_<name>`）：

`extract_scalar`, `extract_slice`, `insert_slice`, `custom_op_for_inter_core_sync`, `index_select_simd`, `index_put`, `gather_out_to_ub`, `scatter_ub_to_out`, `conv1d`, `dot`, `sort`, `flip`, `tanh`, `annotation`, `fixpipe`, `annotation_mark`, `bind_buffer`, `debug_barrier`, `custom_op`, `custom_macro_op`, `scope_op`, `scope_return`, `sync_block_set`, `sync_block_wait`, `sync_block_all`, `get_sub_vec_id`, `copy_buffer`, `copy_tensor`, `convert_layout`。

无 `conv2d`：本树 `TritonAscendOps.td` 没有 `Conv2dOp`。

旧调用点若使用 `builder.sync_block_*`（无 `create_` 前缀），需要薄 Python 别名打包后调用 `create_sync_block_*`。

## 涉及文件 / 目录

| 路径 | 作用 |
|------|------|
| `standalone/AscendOpInfo.{h,cpp}` | 完整 OpInfo 表（从 `ascend_ir.cc` 的 create_* 移植） |
| `standalone/AscendPassPlugin.cpp` | Passes + dialects + ops；`extend=nullptr` |
| `standalone/CMakeLists.txt` | 始终仅 OpInfo；不含 ascend_ir/nanobind |
| `standalone/link_ext_dialects.py` | 对 Linalg/Tensor/Complex/Bufferization（+ Affine 重载）做 host 符号改写 |
| `__init__.py` | 社区 extend_with 路径 |
| `backend/compiler.py` | Dialect 加载 + 库存 parse_attr |
| `language/cann/extension/builder.py` | `setup_unified_builder` no-op / 打包包装 |
| `PLUGINIZATION_1_3.md` | 本文 |

`ascend_ir.cc` / `AscendExtendIrModule.*` 仍留在树内作参考，但**不会**编进 puncture `.so`。

## 构建与使用

```bash
source ~/miniconda/etc/profile.d/conda.sh && conda activate plugin13
export LLVM_SYSPATH=/home/coder/bin/llvm-project-3.9-b010a18-patch
# 可选：LD=$LLVM_SYSPATH/bin/ld.lld  — 仅当 GNU ld 导致 link_ext_dialects 产出空 .o 时使用
cd /home/coder/workspace/main/triton-ascend-plugin-1-3
cmake -S standalone -B build/plugin \
  -DLLVM_SYSPATH=$LLVM_SYSPATH \
  -DASCEND_ROOT=$PWD \
  -DSTANDALONE_OUTPUT_DIR=$PWD/_C \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/plugin -j8 --target libtriton_ascend
# Release 构建仍可能携带来自 LLVM .a 成员的 dialect 调试信息：
strip -g _C/libtriton_ascend.so
# 优先 pip install（会写 RECORD）。若卸载因空 RECORD 失败，先 rm site-packages/triton_ascend*
pip install . --no-build-isolation
```

验证：

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

## 已知缺口

1. **属性较重的 helpers**（`get_pipe_attr`、Affine maps、mark/annotation 上的富属性）原先仅 HostExtend；部分 language 路径 soft-fail，或需文本形式的 `parseAttribute` 打包。
2. **`scope_op` OpState** — OpInfo 只返回单个 Value；经 OpState 进入 scope region 相对 HostExtend 尚不完整。
3. **`custom_op` 多结果** — 仅通过 `operands[0]` 返回第一个结果。
4. **`create_dot` 重载** — Ascend DotOp 为 `create_dot(list[Value])`；库存 `create_dot(a,b,acc,...)` 仍是独立的 nanobind 重载。
5. **`.so` 体积** — strip 后约 118MB（1-2 约 52MB），主要来自 whole-archive dialect 对象 + helpers；进一步 GC/`-ffunction-sections` 可选。
6. **构建提示** — 若 `link_ext_dialects` 在 GNU `ld` 下得到空 relocatable object，设置 `LD=$LLVM_SYSPATH/bin/ld.lld`。
7. **社区 host** — 保持 `TRITON_ASCEND_LINK_AFFINE_OPS_THIN=OFF`（默认）。PyPI `libtriton` 已有 Affine；再链接打过补丁的 `AffineOps.cpp.o` 会导致空的 `linalg.fill` region。

## test_add 状态

**在 plugin13 上通过**（社区 OpInfo 路径，无 HostExtend）：

```
create_extract_scalar True
create_* 167
add_triton_to_linalg True
has _ascend False
pytest test_add.py → 6 passed
```
