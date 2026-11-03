# triton-ascend 独立插件化 Triton 3.7.1 / `TRITON_PLUGIN_PATHS`）
> 树外独立插件方案，基于社区原生 Triton 3.7.1，通过 Triton 官方插件机制接入昇腾后端，不再 fork 维护 Triton 主仓。

本仓库产出独立插件库 `libtriton_ascend.so`，对接上游社区 Triton **3.7.1**。

- 插件接入逻辑收敛在 `standalone/`，**不改动原有 Pass 业务实现**；
- 不再维护一份 fork 的 Triton 源码，Triton C++ 头文件通过 `third_party/triton` submodule（v3.7.1）引入
> PS：社区Triton 3.9.0版本支持导出 C++ 头文件，triton的第三方引入后续升级后可移除
- Python 侧对上游 Triton 的适配补丁统一放在 `backend/triton_hotpatch.py`；
- AscendNPU-IR 编译阶段自动应用 `patch/npuir_adapter_to_llvm_23.patch` 做 LLVM23 语法适配。

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
# 拉取 LLVM patch
wget https://raw.githubusercontent.com/hongziqi/triton-ascend/main-plugin-adapter-3.7/patch/triton-llvm_patch_1f126a6.patch
# 定义 LLVM 安装路径
export LLVM_INSTALL_PREFIX_3_7_1_PATCH=/home/coder/bin/triton-llvm-project-3.7.1-1f126a6-patch
# 编译 LLVM patch 版本
git clone git@github.com:triton-lang/llvm-project.git
cd llvm-project
git checkout 1f126a6dea50d185c0781743a667390037ae88bd
# 需要 apply triton-ascend仓库的patch文件
git apply --whitespace=nowarn ../triton-llvm_patch_1f126a6.patch

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
git submodule update --init --depth 1 --progress

pip install triton==3.7.1 -i https://mirrors.huaweicloud.com/repository/pypi/simple --trusted-host mirrors.huaweicloud.com

# 可编辑安装
pip install -e . --no-build-isolation

# 或编 wheel 再安装
python setup.py bdist_wheel
pip install dist/triton_ascend-*.whl

# 安装后设置插件路径（见上文）
export TRITON_PLUGIN_PATHS="$(python -c 'import triton_ascend; print(triton_ascend.plugin_library())')"
```

## 验证（插件 ABI）
1. 直接验证向量加法：
```bash
export TRITON_PLUGIN_PATHS="$(python -c 'import triton_ascend; print(triton_ascend.plugin_library())')"
python tutorials/01-vector-add.py
```

2. 或执行一下python脚本，验证plugin的pass和op是否导入成功：
```python
import triton
from triton._C.libtriton import ir, passes
import triton_ascend
triton_ascend.register_ascend_native()
assert hasattr(ir.builder, "create_extract_scalar")
assert hasattr(passes.plugin, "add_triton_to_linalg")
```
