# triton-ascend（共享 MLIRContext 插件）

树外 Ascend 后端，对接社区 Triton plugin ABI。原生库 **只有一个插件** `libtriton_ascend.so`（setuptools 安装时可能再复制一份带平台标签的同名文件，内容相同）。它导出 `tritonGetPluginInfo`，只链接宿主 `libtriton.so` + `libnanobind.so`。

```python
import triton
import triton_ascend   # __init__ 自动 dlopen 插件并完成注册
# Pass:   passes.plugin.add_*
# Builder: triton_ascend.ir.ascendnpu_ir_builder  (== ir._ascend)
```

---

## 插件接入逻辑（必读）

社区 plugin 有 **三条互不替代** 的入口；Ascend 用前两条，**不用**第三条挂全量 builder。

| 社区 API | PluginInfo 字段 | Ascend 是否用 | 效果 |
|----------|-----------------|---------------|------|
| `passes.plugin.extend_with(so)` | `passes` / PassInfo | 是（Pass 编入时） | 注册 `passes.plugin.add_*` |
| `ir.extend_module_with(so)` | `extend` / HostExtendCallback | **是（builder 主路径）** | 调 `ascendExtendHost` → `init_ascend_ir` → `ir._ascend` |
| `ir.extend_dialects_with(so)` | `dialects` / DialectInfo | 可由 `extend_module_with` 一并加载 | Dialect 进宿主 registry |
| `ir.builder.extend_with(so)` | `ops` / OpInfo（社区既有，未改） | **否**（表为空） | 简单表驱动；Ascend 不用 |

### 为什么不是 `ir.builder.extend_with`？

`builder.extend_with` 只能按 OpInfo 表往 **社区** `ir.builder`（`TritonOpBuilder`）上挂固定签名的 `create_*`。

Ascend 需要：

- 子类 `AscendNPUIROpBuilder(TritonOpBuilder)`（额外 `target` / `compile_mode`）
- kwargs、Affine、Attribute 工厂、`sync_block_*` 等非表驱动形态

因此走 API v3 的 **`ir.extend_module_with` → HostExtendCallback**：在宿主 `ir` 上建子模块 `_ascend`，把 `ascend_ir.cc` 整面挂上去。`AscendNPUIROpBuilder` 是子类构造，不是往社区 builder 上 `extend_with`。

### `__init__.py` 里实际做了什么

```text
import triton_ascend
  └─ register_ascend_native()          # 幂等，只跑一次
       ├─ passes.plugin.extend_with(so)     # Pass
       └─ ir.extend_module_with(so)         # Builder + Dialect（HostExtend）
            └─ PluginInfo.extend = ascendExtendHost
                 └─ nanobind::register_module(host_ir)
                 └─ host_ir.def_submodule("_ascend")
                 └─ init_ascend_ir(_ascend)   # ascend_ir.cc
  └─ triton_ascend.ir  =  ir._ascend
```

### 调用侧怎么用

| 能力 | 调用 |
|------|------|
| Pass | `passes.plugin.add_triton_to_linalg(pm, ...)` 等（先已 `extend_with`） |
| Builder | `triton_ascend.ir.ascendnpu_ir_builder(ctx, arch)` 或 DSL 经 `setup_unified_builder` 挂到 codegen builder |
| Dialect 进具体 Context | `triton_ascend.ir.load_dialects(ctx)`（`NPUBackend.load_dialects`） |

`backend/compiler.py` **不再**重复 `extend_with` / `builder.extend_with`；只做幂等 `register_ascend_native()`，以及编译管线里对 `passes.plugin.add_*` 的**调用**。

---

## 目录：插件化相关文件

### 必备

| 路径 | 作用 |
|------|------|
| `ascend_ir.cc` | 全量 Ascend IR nanobind（含 `AscendNPUIROpBuilder`）；编进 `.so` |
| `standalone/AscendPassPlugin.cpp` | `tritonGetPluginInfo` |
| `standalone/AscendExtendIrModule.*` | `HostExtendCallback` |
| `standalone/AscendOpInfo.*` | PluginInfo.ops 空表（builder 走 HostExtend） |
| `standalone/PassObjects.cmake` + `CMakeLists.txt` | 插件 MODULE；宿主 so + 缺失方言 `.o` |
| `standalone/link_ext_dialects.py` | **构建必跑**：四方言 partial-link + 与宿主去重 |
| `__init__.py` | import 时自动 `extend_with` + `extend_module_with` |
| `setup.py` | 调 cmake，打包 `_C/libtriton_ascend*.so` |
| `lib/` + `include/` | Dialect / Pass 实现 |
| `backend/` + `language/` | Python 后端与 DSL |

### 原生库命名

- **逻辑上只有一个插件**：`libtriton_ascend.so`
- 安装目录里可能同时有 `libtriton_ascend.cpython-311-*.so`（setuptools 标签副本），`plugin_library()` 会解析到其中一个
- **没有**单独的 `triton_ascend.so` / `libTritonAscendPassPlugin.so` 第二条产物线

`patch/` 是历史 fork 补丁；Track A 社区补丁在 `~/workspace/docs/设计文档/patches/`。

---

## 环境

```bash
conda activate plugin
export LLVM_SYSPATH=$HOME/.triton/llvm/llvm-b010a18d-ubuntu-arm64-1
```

| 项 | 路径 |
|----|------|
| 社区 Triton | `~/workspace/main/triton`（基线 `872c90a6`） |
| 本仓库 | `~/workspace/main/triton-ascend-plugin-test-1` |
| Track A patches | `~/workspace/docs/设计文档/patches/` |
| ↳ 0001 | TypeID（`plugin_ext_dialects.whitelist` → 链库 + 生成活引用） |
| ↳ 0002 | HostExtend + shared `libnanobind`（**无** OpBuildArgs） |
| ↳ 0000 | 0001+0002 合并包 |

---

## 一、安装带 Track A 的社区 Triton

### 方式 A：预构建 wheel

```bash
pip install \
  ~/workspace/docs/设计文档/artifacts/triton-3.9.0+git872c90a6-cp311-cp311-linux_aarch64.whl \
  --force-reinstall --no-deps
```

若预构建 wheel 不含 HostExtend / `libnanobind.so`，需用方式 B。

### 方式 B：打补丁编 wheel

```bash
cd ~/workspace/main/triton
git apply ~/workspace/docs/设计文档/patches/0001-link-linalg-tensor-force-typeid-into-libtriton.patch
git apply ~/workspace/docs/设计文档/patches/0002-plugin-abi-v3-host-extend-shared-nanobind.patch
# 或：git apply …/0000-combined-track-a-host-extend-typeid-nanobind.patch

TRITON_EXT_ENABLED=1 MAX_JOBS=64 \
  pip wheel . -w /tmp/triton_wheel_out --no-deps --no-build-isolation
pip install /tmp/triton_wheel_out/triton-*.whl --force-reinstall --no-deps
```

### 宿主自检

```bash
python - <<'PY'
import pathlib, triton
c = pathlib.Path(triton.__file__).parent / "_C"
assert (c / "libtriton.so").is_file()
assert (c / "libnanobind.so").is_file()
from triton._C.libtriton import ir
assert hasattr(ir, "extend_module_with")
print("PASS", triton.__version__)
PY
```

---

## 二、安装 triton-ascend

```bash
export LLVM_SYSPATH=$HOME/.triton/llvm/llvm-b010a18d-ubuntu-arm64-1
export LD_LIBRARY_PATH="$(python -c 'import triton,pathlib; print(pathlib.Path(triton.__file__).parent/"_C")'):${LD_LIBRARY_PATH:-}"

cd ~/workspace/main/triton-ascend-plugin-test-1
pip install . --no-build-isolation
```

默认编入完整 Ascend Dialect + lowering Pass + 全量 `ascend_ir.cc`（HostExtend）。

```bash
python - <<'PY'
import triton_ascend
assert triton_ascend.ir is not None
print("so:", triton_ascend.plugin_library())
print("builder:", hasattr(triton_ascend.ir, "ascendnpu_ir_builder"))
PY
```

---

## 三、冒烟验证

```bash
# 建议关闭 TRITON_DEBUG；插件加载与编译路径依赖双包已安装
unset TRITON_DEBUG
export LD_LIBRARY_PATH="$(python -c 'import triton,pathlib; print(pathlib.Path(triton.__file__).parent/"_C")'):$LD_LIBRARY_PATH"
cd unittest/pytest_ut && pytest test_add.py -q
```

---

## 链接配方

- 编译：宿主 `TRITON_EXT` 头文件（API v3 时还有 `libnanobind.so`）
- 链接：`libtriton.so` +（v3）`libnanobind.so` + `ext_dialects_unique.o`（缺失四方言）+ 少量 helper `.a`
- 不要链私有 `nanobind-static`
- **公共** MLIR 符号仍解析到宿主；**缺失方言** TypeID 定义在插件 so 内

细节见 `standalone/README.md`（含「为何先做成 `.o`」）。
