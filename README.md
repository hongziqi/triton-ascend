# triton-ascend-plugin-3.7-1-3（Triton 3.7.1 / `TRITON_PLUGIN_PATHS`）

树外 Ascend 后端插件：产出 **`libtriton_ascend.so`**，对接社区 Triton **3.7.1**（Plugin ABI **v2**，环境变量加载）。

与 3.9 / `extend_with` 穿刺的差异见 [`PLUGINIZATION_1_3.md`](PLUGINIZATION_1_3.md)。

---

## 仓库布局

```text
.
├── AscendNPU-IR/          # git submodule（bb13ef612；构建时打 llvm23 adapter）
├── third_party/triton/    # git submodule（v3.7.1；提供 C++ 头，wheel 无 include）
├── backend/               # Python backend + triton_hotpatch
├── language/              # cann DSL / kernels
├── include/ lib/          # Ascend C++ passes / dialects
├── standalone/            # 插件 CMake：编出 libtriton_ascend.so
├── patch/
│   ├── triton-llvm_patch_1f126a6.patch  # Ascend MLIR 补丁（对齐宿主 3.7.1 LLVM）
│   └── npuir_adapter_to_llvm_23.patch
├── scripts/apply_npuir_patch.py
├── _autoload_plugin.py    # 由 .pth 在解释器启动时设置 TRITON_PLUGIN_PATHS
└── _C/                    # 构建产物
```

## 穿刺依赖（LLVM 必须一致）

| 项 | 值 |
|------|------|
| 宿主 | `triton==3.7.1`（PyPI / `1f126a6`） |
| LLVM 源码 | `~/origin/triton-llvm-project-3.7.1-1f126a6-patch` @ `1f126a6` + `triton-llvm_patch_1f126a6.patch` |
| 本机 install | `LLVM_INSTALL_PREFIX_3_7_1_PATCH=/home/coder/bin/triton-llvm-project-3.7.1-1f126a6-patch` |
| 加载 | `TRITON_PLUGIN_PATHS`（无 `extend_with`） |
| 头文件 | `third_party/triton` submodule（v3.7.1） |

**重要：** 插件必须用与宿主 **同一 LLVM commit**（`1f126a6`）构建；仅打 Ascend 的 MLIR 文本/折叠补丁。跨 commit（例如插件 `ac5dc54`、宿主 `1f126a6`）会在 `TritonToLinalg` 因 TypeID/promised-interface 混用而 SIGSEGV。

### 编译 LLVM（一次性）

```bash
export LLVM_INSTALL_PREFIX_3_7_1_PATCH=/home/coder/bin/triton-llvm-project-3.7.1-1f126a6-patch
cd ~/origin/triton-llvm-project-3.7.1-1f126a6-patch
git apply --whitespace=nowarn \
  ~/workspace/main/triton-ascend-plugin-3.7-1-3/patch/triton-llvm_patch_1f126a6.patch
mkdir -p build && cd build
cmake ../llvm -G Ninja \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX_3_7_1_PATCH} \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
ninja install
```

## 构建与安装

```bash
source ~/miniconda/etc/profile.d/conda.sh && conda activate plugin3.7
export LLVM_SYSPATH=${LLVM_INSTALL_PREFIX_3_7_1_PATCH:-/home/coder/bin/triton-llvm-project-3.7.1-1f126a6-patch}
export LD=$LLVM_SYSPATH/bin/ld.lld
cd ~/workspace/main/triton-ascend-plugin-3.7-1-3
git submodule update --init AscendNPU-IR third_party/triton

cmake -S standalone -B build/plugin \
  -DLLVM_SYSPATH=$LLVM_SYSPATH \
  -DASCEND_ROOT=$PWD \
  -DSTANDALONE_OUTPUT_DIR=$PWD/_C \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/plugin -j --target libtriton_ascend
strip -g _C/libtriton_ascend.so
pip install . --no-build-isolation
# 若未自动写入 .pth：
echo 'import triton_ascend._autoload_plugin' > $CONDA_PREFIX/lib/python3.11/site-packages/triton_ascend_plugin.pth
```

## 验证（插件 ABI）

```python
import triton, triton_ascend
from triton._C.libtriton import ir, passes
triton_ascend.register_ascend_native()
assert hasattr(ir.builder, "create_extract_scalar")
assert hasattr(passes.plugin, "add_triton_to_linalg")
```

```bash
# 手动指定时必须在 import triton 之前：
export TRITON_PLUGIN_PATHS=$PWD/_C/libtriton_ascend.so
```
