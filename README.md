# triton-ascend main adapter（Triton 3.7.1 / `TRITON_PLUGIN_PATHS`）

树外 Ascend 后端插件：产出 **`libtriton_ascend.so`**，对接社区 Triton **3.7.1**。

PASS 源码来自 triton-ascend **main**（仍保留 `tt.make_tensor_ptr` 等 3.7 op）。插件接入只在 `standalone/`：重定义 `add_triton_library` 为 object library，不改写各 pass 的实现。

社区 Triton 源码不再 fork 进本仓库。头文件来自 submodule `third_party/triton`（v3.7.1）。对社区 Python 的修改走 `backend/triton_hotpatch.py`。AscendNPU-IR 保持干净 commit，构建时热打 `patch/npuir_adapter_to_llvm_23.patch`。

## 仓库布局

```text
.
├── AscendNPU-IR/          # git submodule（4eb12142；构建时打 llvm23 adapter）
├── third_party/triton/    # git submodule（v3.7.1；仅 C++ 头）
├── backend/ language/ include/ lib/ unittest/
├── standalone/            # 插件 CMake、PassInfo、OpInfo
├── patch/
│   ├── triton-llvm_patch_1f126a6.patch
│   └── npuir_adapter_to_llvm_23.patch
├── scripts/apply_npuir_patch.py
└── _autoload_plugin.py
```

### 编译 LLVM（一次性）

```bash
export LLVM_INSTALL_PREFIX_3_7_1_PATCH=/home/coder/bin/triton-llvm-project-3.7.1-1f126a6-patch
git clone git@github.com:triton-lang/llvm-project.git
mv llvm-project triton-llvm-project-3.7.1-1f126a6-patch
cd triton-llvm-project-3.7.1-1f126a6-patch
# 需要 apply triton-ascend仓库的patch文件
git apply --whitespace=nowarn \
  triton-ascend/patch/triton-llvm_patch_1f126a6.patch

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
export LLVM_SYSPATH=${LLVM_INSTALL_PREFIX_3_7_1_PATCH}
git clone -b main-plugin-adapter-3.7 https://github.com/hongziqi/triton-ascend.git
cd triton-ascend
git submodule update --init --depth 1 --progress AscendNPU-IR third_party/triton
pip install triton==3.7.1
pip install -e . --no-build-isolation
```

## 验证（插件 ABI）

```python
import triton, triton_ascend
from triton._C.libtriton import ir, passes
triton_ascend.register_ascend_native()
assert hasattr(ir.builder, "create_extract_scalar")
assert hasattr(passes.plugin, "add_triton_to_linalg")
```
验证 test_add:
```bash
pytest unittest/pytest_ut/test_add.py
```
