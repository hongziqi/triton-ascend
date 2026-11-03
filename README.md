# triton-ascend main adapter（Triton 3.7.1 / `TRITON_PLUGIN_PATHS`）

树外 Ascend 后端插件：产出 **`libtriton_ascend.so`**，对接社区 Triton **3.7.1**。

PASS 源码来自 triton-ascend **main**（仍保留 `tt.make_tensor_ptr` 等 3.7 op）。插件接入只在 `standalone/`：重定义 `add_triton_library` 为 object library，不改写各 pass 的实现。

社区 Triton 源码不再 fork 进本仓库。头文件来自 submodule `third_party/triton`（v3.7.1）。对社区 Python 的修改走 `backend/triton_hotpatch.py`。AscendNPU-IR 保持干净 commit，构建时热打 `patch/npuir_adapter_to_llvm_23.patch`。

## 仓库布局

```text
.
├── third_party/
│   ├── AscendNPU-IR/      # git submodule（4eb12142；构建时打 llvm23 adapter）
│   └── triton/            # git submodule（v3.7.1；仅 C++ 头）
├── backend/ language/ include/ lib/ unittest/
├── standalone/            # 插件 CMake、PassInfo、OpInfo
└── patch/
    ├── triton-llvm_patch_1f126a6.patch
    └── npuir_adapter_to_llvm_23.patch
```

### 插件如何加载（无 site `.pth` / 无 conda hook）

Triton 3.7 在**首次** `import triton._C.libtriton` 时读取 `TRITON_PLUGIN_PATHS` 并加载插件。社区 3.7.1 会在 `runtime.autotuner` 里早于后端发现导入 `libtriton`，因此需要在 import triton **之前**设置好路径。

标准流程：先装 `triton==3.7.1`，再装本包。然后任选其一：

```bash
# 推荐：安装后把插件路径写进环境（venv / shell profile / CI 均可）
export TRITON_PLUGIN_PATHS="$(python -c 'import triton_ascend; print(triton_ascend.plugin_library())')"
```

或在代码里先导入本包再导入 triton：

```python
import triton_ascend  # sets TRITON_PLUGIN_PATHS
import triton
```

后端发现阶段（`backend/plugin_bootstrap.py`）会再次检查路径；若宿主 Triton 版本不是 **3.7.1**，会 warning 提示 `pip install triton==3.7.1`。

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
git submodule update --init --depth 1 --progress \
  third_party/AscendNPU-IR third_party/triton

pip install triton==3.7.1

# 可编辑安装
pip install -e . --no-build-isolation

# 或编 wheel 再安装
python setup.py bdist_wheel
pip install dist/triton_ascend-*.whl

# 安装后设置插件路径（见上文）
export TRITON_PLUGIN_PATHS="$(python -c 'import triton_ascend; print(triton_ascend.plugin_library())')"
```

## 验证（插件 ABI）

```python
import triton
from triton._C.libtriton import ir, passes
import triton_ascend
triton_ascend.register_ascend_native()
assert hasattr(ir.builder, "create_extract_scalar")
assert hasattr(passes.plugin, "add_triton_to_linalg")
```

```bash
pytest tutorials/01-vector-add.py
```
