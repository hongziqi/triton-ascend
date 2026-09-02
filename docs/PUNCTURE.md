# Ascend 插件化穿刺计划

## 宿主要求

L3+ 穿刺需 **TRITON_EXT 构建的 triton wheel**（非 PyPI release）：

- `triton._C.libtriton.passes.plugin.extend_with` 存在
- `triton/include/` 存在（out-of-tree 编插件）
- `libtriton.so` 导出 MLIR/Triton 符号

**当前 L3 宿主**（已验证）：

```text
triton-3.8.0+gitf893845b  (triton-ext nightly, aarch64 cp311)
pin: f893845b9b91599ebd3b7a9c7f28164f39c7ed94
```

---

## 目标目录（Step 1 ✅ 已完成）

```text
triton-ascend-plugin-test/
├── cmake/ support/ ci/
├── extensions/ascend/          # canonical Ascend 源码
├── docs/PUNCTURE.md
├── Makefile
└── README.md
```

已删除：Triton fork 主体（`lib/`、`python/`、`setup.py` 等）、`third_party/nvidia|amd|proton|ascend`、社区 CI/docs。

---

## 穿刺步骤

| Step | 内容 | 状态 |
|------|------|------|
| 1 | 目录重构 → out-of-tree 布局 | ✅ |
| 2 | 验证 TRITON_EXT 宿主（extend_with + headers） | ✅ |
| 3 | 最小 plugin（LoopSplit 探针 → libtriton_ascend.so） | 🔄 |
| 4 | ascend_ir 独立 pybind | ⏸ |
| 5 | 全 pass 迁移 + 参数 string 化 | ⏸ |
| 6 | compiler/driver/runtime Python 双包 | 🔄 L1 已通过 |
| 7 | Patch 最小化 | ⏸ |
| 8 | CI / 发布 | ⏸ |

### Step 1 — 目录重构 ✅

- [x] 引入 triton-ext `cmake/`、`support/`、`ci/`
- [x] Ascend 代码 canonical 在 `extensions/ascend/`
- [x] 删除 fork 遗留（`lib/`、`python/`、`third_party/`、`setup*.py` 等）

### Step 2 — 验证宿主 ✅

`plugin` env + triton-ext nightly：

```bash
python -c "
import triton._C.libtriton as L
from pathlib import Path
print('extend_with:', hasattr(L.passes.plugin,'extend_with'))
print('include:', (Path(triton.__file__).parent/'include').exists())
"
# extend_with: True, include: True
```

### Step 3 — L3 native plugin 🔄

**3a LoopSplit 探针**（验证 triton-ext 工具链）：

```bash
pip wheel extensions/ascend/puncture/loop_split_probe --no-build-isolation -w dist
pip install dist/triton_loop_split_probe-*.whl --no-deps
python extensions/ascend/scripts/l3_puncture_test.py
```

**3b Ascend pass**（待做）：迁 `TritonControlFlowOpt` → `libtriton_ascend.so` + compiler.py 适配。

### Step 2 (L1) — Python backend ✅

```bash
bash extensions/ascend/scripts/build_python_wheel.sh
pip install extensions/ascend/dist/triton_ascend-*.whl --no-deps --force-reinstall
PYTHONPATH= python extensions/ascend/scripts/l1_puncture_test.py
```

---

## 已知要点

1. **`libtriton.ascend` 不在 bare host** — C++ 能力走 out-of-tree `.so` 或独立 pybind（Step 4）
2. **Pass 参数** — plugin ABI 用 `vector<string>`；compiler.py 需适配
3. **BiShengIR** — 链入 plugin.so，不能只 link libtriton
4. **TA 自编宿主**（备选）：`extensions/ascend/scripts/build_triton_ext_host.sh` 指向 `triton-ascend-main`
