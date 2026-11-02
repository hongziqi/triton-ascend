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

## LLVM

| 项 | 值 |
|------|------|
| 宿主 | `triton==3.7.1`（`1f126a6`） |
| 本机 install | `LLVM_INSTALL_PREFIX_3_7_1_PATCH=/home/coder/bin/triton-llvm-project-3.7.1-1f126a6-patch` |
| 加载 | `TRITON_PLUGIN_PATHS` |
| 虚拟环境 | `plugin3.7` |

LLVM 补丁打在上述 install 对应的源码树上，不进本仓库的编译。

## 构建

```bash
source ~/miniconda/etc/profile.d/conda.sh && conda activate plugin3.7
export LLVM_SYSPATH=${LLVM_INSTALL_PREFIX_3_7_1_PATCH:-/home/coder/bin/triton-llvm-project-3.7.1-1f126a6-patch}
cd ~/workspace/main/triton-ascend-main-adapter-plugin-3.7
git submodule update --init AscendNPU-IR third_party/triton
pip install -e . --no-build-isolation
```

`pip` 默认会吞掉构建子进程 stdout，终端上常只剩一行 `Building editable ...`。本仓库的 `setup.py` 会：

- 把 cmake 进度打到 **`/dev/tty`**（即使非 `-v` 也能看到 `[ xx%]`）
- 完整日志写到 `build/plugin/logs/cmake-configure.log` / `cmake-build.log`（可用 `tail -f`）
- 若 `CMakeCache` 里的 `LLVM_DIR` 与当前 `LLVM_SYSPATH` 不一致，自动清掉旧 build 目录

需要 pip 原样回显全部输出时：`pip install -v -e . --no-build-isolation`。