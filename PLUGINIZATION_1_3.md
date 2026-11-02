# Triton-Ascend 插件化（Triton 3.7.1 / `TRITON_PLUGIN_PATHS`）

## 与 3.9 穿刺的差异

| | Triton 3.9（plugin-1-3） | Triton 3.7.1（本树） |
|--|-------------------------|---------------------|
| 加载方式 | `extend_with` / `extend_dialects_with` | **`TRITON_PLUGIN_PATHS`**（import 前） |
| 头文件 | wheel 内 `triton/include/` | **`third_party/triton` submodule**（v3.7.1） |
| Pass/Op 名 | 表内无前缀，宿主自动加 `add_`/`create_` | **表内自带** `add_*` / `create_*` |
| LLVM | `b010a18` + `llvm_patch_b010a18` | **`ac5dc54` + `llvm_patch_ac5dc54`** |
| NPUIR adapter | `npuir_adapter_to_llvm_24.patch` | **`npuir_adapter_to_llvm_23.patch`** |
| AscendNPU-IR pin | `aea934a6` | **`bb13ef612`** |

## 源码布局

- Host `triton==3.7.1` 来自 pip（需 `TRITON_EXT_ENABLED=1`）。
- `third_party/triton`：社区 v3.7.1 源码，仅作编译头文件。
- `AscendNPU-IR/`：git submodule（`bb13ef612`）；构建时打 `npuir_adapter_to_llvm_23.patch`。
- 一次性 LLVM Ascend 补丁：`patch/llvm_patch_ac5dc54.patch`。

## 加载路径

宿主在初始化 `libtriton` 时读取 `TRITON_PLUGIN_PATHS`（冒号分隔），`dlopen` 插件并：

- PassInfo → `passes.plugin.<name>`（本树用 `add_*`）
- OpInfo → `ir.builder.<name>`（本树用 `create_*`）
- DialectInfo → 在 `ir.load_dialects` 时注册

安装后由 `triton_ascend_plugin.pth` → `_autoload_plugin` 在解释器启动时写入 env。也可手动：

```bash
export TRITON_PLUGIN_PATHS=/path/to/libtriton_ascend.so
python -c "import triton, triton_ascend; ..."
```

## 构建

```bash
source ~/miniconda/etc/profile.d/conda.sh && conda activate plugin3.7
export LLVM_SYSPATH=${LLVM_INSTALL_PREFIX_3_7_PATCH:-/home/coder/bin/llvm-install-3-7_ac5dc54_patch}
cd ~/workspace/main/triton-ascend-plugin-3.7-1-3
git submodule update --init AscendNPU-IR third_party/triton

cmake -S standalone -B build/plugin \
  -DLLVM_SYSPATH=$LLVM_SYSPATH \
  -DASCEND_ROOT=$PWD \
  -DSTANDALONE_OUTPUT_DIR=$PWD/_C \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/plugin -j8 --target libtriton_ascend
strip -g _C/libtriton_ascend.so
pip install . --no-build-isolation
```

验证：

```python
import triton, triton_ascend
from triton._C.libtriton import ir, passes
triton_ascend.register_ascend_native()
assert hasattr(ir.builder, "create_extract_scalar")
assert hasattr(passes.plugin, "add_triton_to_linalg")
```

```bash
cd unittest/pytest_ut && TRITON_DEBUG=0 pytest test_add.py -q
```
