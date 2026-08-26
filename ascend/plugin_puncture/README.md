# Ascend 插件化方案穿刺（第一步）

本目录只做**可运行穿刺**，完整设计文档（好处 / 为何做 / 编译与目录 / 算子与 pass / triton-opt / so 链接）**后续再写**。TA patch 本步不考虑。

## 穿刺目标

1. **Layer C 绑定**：`TRITON_PASS_PLUGIN_PATH` 的 `.so` 能被当前进程的 `libtriton` 加载成 `passes.plugin.*`。  
2. **Ascend 接线**：`AscendBackend.add_stages` 能调到 `knobs.runtime.add_stages_inspection_hook`，从而在 NPU 流水线里插入自定义逻辑。  
3. **Layer C 真正跑 pass**：社区 `tritongpu-plugin` 把函数改成 `@foo` —— **需要 `LLVM_BUILD_SHARED_LIBS=1`**；本仓默认静态 LLVM 下会 TypeID 冲突，记为 EXPECTED_FAIL。

源码侧已改：`ascend/backend/compiler.py`（`add_stages` 末尾调用 stages hook，与 nvidia/amd 对齐）。  
Ascend 已迁到仓根 `ascend/`，经 **`TRITON_PLUGIN_DIRS`（out-of-tree）** 接入，不再走 `third_party/ascend` in-tree。
若环境里是旧 wheel 副本，脚本 02 会**进程内 monkey-patch** `AscendBackend.add_stages`。

## 怎么跑

```bash
cd ~/workspace/main/triton-ascend-candy-dev
conda activate triton   # 需已装本仓 triton-ascend；plugin 环境需先 install

# 1) Layer C 绑定 + 可选 triton-opt CLI
bash ascend/plugin_puncture/01_triton_opt_plugin.sh

# 2) Ascend JIT：hook 接线（2a）+ .so pass（2b，静态 LLVM 预期失败）
python ascend/plugin_puncture/02_ascend_stages_hook.py
```

## 期望结果

| 步骤 | PASS 判据 |
| --- | --- |
| 01a | `passes.plugin.add_plugin` 为 True |
| 01b | `triton-opt -tritongpu-plugin` 改出 `@foo`；静态 LLVM 时 SKIP |
| 02a | hook 被调用，且能观察到 `ttir` stage / entry func 名 |
| 02b | 共享 LLVM 时 `@foo`；静态 LLVM 时 EXPECTED_FAIL（cast/PassManager） |

## A / B / C（先记一笔）

| 层 | 含义 | 本穿刺 |
| --- | --- | --- |
| A | Backend 包（`TRITON_PLUGIN_DIRS` / entry_points） | 未做 out-of-tree |
| B | `add_triton_plugin` 链进 `libtriton` | Ascend 已是此形态，不改 |
| C | 运行时 `.so` + stages hook | **本步重点**；真跑 pass 依赖共享 LLVM |

## 已知结论（穿刺产出）

- Ascend 原先缺 nvidia/amd 那一行 stages hook；源码已补，重装后无需 monkey-patch。  
- 默认 `LLVM_BUILD_SHARED_LIBS=0`：`.so` 能 `dlopen` 并暴露 Python binding，但 pass 内 `cast<ModuleOp>` 会断言失败 —— 与社区 `REQUIRES: shared-libs` 一致。  
- 若产品路径要「自动链接运行时 so 并进 triton-opt / JIT」，需评估共享 LLVM 构建成本，或继续走 Layer B 静态链入。
