# triton-ascend-plugin-1-3（社区 Plugin ABI / OpInfo）

树外 Ascend 后端插件：产出 **`libtriton_ascend.so`**，对接社区 Triton（Plugin ABI **v2**）。

- `passes.plugin.extend_with(so)` — PassInfo
- `ir.extend_dialects_with(so)` — DialectInfo
- `ir.builder.extend_with(so)` — OpInfo → `create_*`

细节：[`PLUGINIZATION_1_3.md`](PLUGINIZATION_1_3.md)、[`standalone/README.md`](standalone/README.md)。

---

## 仓库布局

```text
.
├── AscendNPU-IR/          # git submodule（干净 HEAD；构建时打 adapter patch）
├── backend/               # Python backend + triton_hotpatch
├── language/              # cann DSL / kernels
├── include/ lib/          # Ascend C++ passes / dialects
├── standalone/            # 插件 CMake：编出 libtriton_ascend.so
├── patch/
│   ├── llvm_patch_b010a18.patch      # 一次性打进 llvm-project
│   └── npuir_adapter_to_llvm_24.patch # 每次构建打进 AscendNPU-IR
├── scripts/apply_npuir_patch.py
├── _C/                    # 构建产物（gitignore）
└── unittest/
```

**不是** monorepo：宿主 `triton` 来自 PyPI/conda，**不要**再挂 `third_party/triton`。

### Submodule

```bash
git submodule update --init AscendNPU-IR
# 当前穿刺 pin：aea934a6（与 monorepo main + npuir_adapter_to_llvm_24 对齐）
```

`AscendNPU-IR` 工作树在构建前后可能短暂脏（patch 已打入）；下次构建会先 `git checkout` 相关文件再重新 `git apply`。

---

## 穿刺依赖（LLVM 必须一致）

```text
        LLVM_SYSPATH（同一棵 install）
                 │
     ┌───────────┴───────────┐
     ▼                       ▼
 宿主 triton wheel      triton-ascend 插件
 libtriton.so  ◄DT_NEEDED─  libtriton_ascend.so
```

| 项 | 当前穿刺结论 |
|------|------|
| LLVM commit | 社区 Triton **3.9.0** → **`b010a18d`**（LLVM 24） |
| Ascend LLVM patch | `patch/llvm_patch_b010a18.patch`（一次性打进 llvm-project） |
| NPUIR adapter | `patch/npuir_adapter_to_llvm_24.patch`（**每次构建**打进 submodule） |
| 本机 install 前缀 | `/home/coder/bin/llvm-project-3.9-b010a18-patch` |
| 宿主 | PyPI **`triton==3.9.0`**（ABI v2） |
| AscendNPU-IR pin | `aea934a66646e837c54fea11e87db54d42eb3221` |

### `patch/` 保留什么

| 文件 | 何时打 | 打到哪 |
|------|--------|--------|
| `llvm_patch_b010a18.patch` | 编 LLVM 前，一次性 | `llvm-project` @ `b010a18d` |
| `npuir_adapter_to_llvm_24.patch` | **每次** `pip install` / cmake configure | `AscendNPU-IR/` submodule |

社区源码树（LLVM / AscendNPU-IR）保持干净，便于跟上游；Ascend 适配以独立 `.patch` 沉淀。

### 与 triton-shared 的差异（Linalg）

| | **triton-shared** | **本插件（树外 `.so`）** |
|--|--|--|
| 集成 | `TRITON_PLUGIN_DIRS` → OBJECT 打进宿主 | 独立 `libtriton_ascend.so`，运行时 `dlopen` |
| Linalg/Tensor | 普通链进宿主 | `link_ext_dialects.py`：四方言 `.a` → `.o`，宿主已有符号改 UND |
| TypeID | 天然单份 | 禁止与宿主重复（勿整库 Affine + Linalg 同一次 `ld -r`） |

社区 PyPI 宿主已带 Affine / 部分 Tensor TypeID。默认 **`TRITON_ASCEND_LINK_AFFINE_OPS_THIN=OFF`**。

---

## 构建 patched LLVM

```bash
git clone git@github.com:llvm/llvm-project.git
mv llvm-project llvm-project-3.9-b010a18-patch
cd llvm-project-3.9-b010a18-patch
git checkout b010a18d2b648cab83c83967ff26b8fde11acdc6
export LLVM_INSTALL_PREFIX_3_9_PATCH=/path/to/bin/llvm-project-3.9-b010a18-patch
git apply /path/to/triton-ascend/patch/llvm_patch_b010a18.patch

mkdir -p build && cd build
cmake ../llvm -G Ninja \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX_3_9_PATCH} \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
ninja install
```

---

## 宿主Triton + 插件Triton-Ascend安装

```bash
pip install 'triton==3.9.0'

git submodule update --init AscendNPU-IR

export LLVM_SYSPATH=/home/coder/bin/llvm-project-3.9-b010a18-patch
cd /path/to/triton-ascend-plugin-1-3

# tips: 使用  --no-build-isolation 加快构建
pip install -e .
```

验证：

```bash
rm -rf ~/.triton/cache ~/.triton/dump
pytest unittest/pytest_ut/test_add.py 
```

### 链接硬规则

- Helper：`${LLVM_SYSPATH}/lib/lib*.a` + `--start-group`，禁止 MLIR CMake TARGET。
- 四方言 → `ext_dialects_unique.o`（`link_ext_dialects.py`）。
- 默认不链 AffineOps 薄抽取；仅 EXT 宿主缺 MaybeAlign 时：`-DTRITON_ASCEND_LINK_AFFINE_OPS_THIN=ON`。
- `_C/` 只保留 `libtriton_ascend.so`。
