# 插件适配层（`libtriton_ascend.so`）

本目录产出**唯一**原生插件 `libtriton_ascend.so`。由 `setup.py` → CMake 构建；`pip install .` 时会跑完整链路。

```python
# 由 triton_ascend.__init__ 自动调用（community，无 HostExtend）
passes.plugin.extend_with(so)    # PassInfo
ir.extend_dialects_with(so)      # DialectInfo → load_dialects
ir.builder.extend_with(so)       # OpInfo → create_<name>
# 不用 ir.extend_module_with
```

详见仓库根目录 `PLUGINIZATION_1_3.md`。

## 文件（均参与产品构建，无测试探针）

| 文件 | 作用 |
|------|------|
| `CMakeLists.txt` | 插件 MODULE：仅 `AscendPassPlugin.cpp` + `AscendOpInfo.cpp`（无 ascend_ir / nanobind） |
| `link_ext_dialects.py` | **setup/cmake 必跑**：四方言 `.a` → `.o`，宿主已有符号改 UND（优先 `ld.lld`） |
| `PassObjects.cmake` | Ascend / BiShengIR Dialect、Pass → OBJECT（不链 MLIR 闭包） |
| `AscendPassPlugin.cpp` | `tritonGetPluginInfo`（`extend=nullptr`） |
| `AscendOpInfo.*` | Community `PluginInfo.ops`（`builder.extend_with`） |
| `ArgParser.h` | Pass 参数解析 |
| `../ascend_ir.cc` | 参考实现 only（**不**编入插件） |

## `link_ext_dialects.py` 是不是测试脚本？

**不是。** 它是构建步骤，由 `CMakeLists.txt` 的 `add_custom_command` 在编 `libtriton_ascend` 时调用：

```text
pip install .
  → setup.py CMakeBuild.build_extension
    → cmake -S standalone -B build/plugin
    → cmake --build … --target libtriton_ascend
         → 生成 build/plugin/ext_dialects_unique.o   ← 跑 link_ext_dialects.py
         → 链入 libtriton_ascend.so
```

没有这一步，插件里就没有 Linalg/Tensor/Complex/Bufferization 的 TypeID，`dlopen` 或 Pass 会因 `undefined symbol` 失败（社区宿主默认不导出这些方言）。

## 为什么要先做成 `.o`，而不是「直接把缺失符号打进 ascend.so」？

直接 `target_link_libraries(… MLIRLinalgDialect …)`（甚至 `--whole-archive`）看起来更简单，但有两个硬问题：

1. **整库吸入会带进宿主已有的符号**  
   方言 `.a` 里不只含 `LinalgDialect` TypeID，还有对 Arith / Builder / MLIRContext 等的引用，以及少量与宿主重叠的强符号。若原样打进插件，进程里会出现**第二份**公共 MLIR TypeID → 与 `libtriton.so` 冲突。

2. **链接器不会「自动识别缺失集合再只链那一块」**  
   `--as-needed` / 普通链 `.a` 只会拉「当前未定义符号需要的成员」。宿主缺的是整套方言 TypeID 活引用；若没有强制 whole-archive，TypeID 对象常被丢掉。反过来 whole-archive 又把重叠符号一起带进来。

因此采用两段式：

```text
① ld -r --whole-archive 四个方言 .a → combined.o
     （保证四方言 TypeID 全部留下）

② 读 libtriton.so 的动态导出表；把 combined.o 里
     「宿主已经定义」的全局符号改成 UND（未定义）
     「宿主没有」的符号保留为插件内定义

③ 最终链 so 时把该 .o 当作 EXTERNAL_OBJECT 打进去，
     并 DT_NEEDED → libtriton.so
```

结果：

| 符号 | 定义在哪 |
|------|----------|
| Linalg/Tensor/Complex/Bufferization TypeID 等 | `libtriton_ascend.so` |
| Arith、Builder、公共 MLIR/LLVM | 仍是 `libtriton.so`（一份） |

辅助库（`MLIRTensorUtils`、`MLIRTilingInterface` 等）用**普通**链 `.a`（非整库），只拉当前未定义需要的成员，进一步缩小闭包。

## setup / 编译构建流程

```text
pip install .  （或 cmake -S standalone …）
│
├─ 1. 发现宿主
│     import triton → …/triton/_C/libtriton.so
│     头文件 wheel_headers；API≥3 时还要 libnanobind.so
│
├─ 2. PassObjects.cmake
│     tablegen + Ascend/BiShengIR → OBJECT
│     TritonIR 等做成 INTERFACE，不在插件里再编一份
│
├─ 3. link_ext_dialects.py   ← 必跑
│     LLVM_SYSPATH 的 4× libMLIR*Dialect.a
│       → ext_dialects_unique.o
│
├─ 4. 编译插件源
│     AscendPassPlugin / OpInfo /（v3）ascend_ir + Extend
│
└─ 5. 链 MODULE → _C/libtriton_ascend.so
      输入: Ascend OBJECT + ext_dialects_unique.o + helper .a
      DT_NEEDED: libtriton.so [+ libnanobind.so]
```

## 构建命令

```bash
export LLVM_SYSPATH=/path/to/llvm-used-by-host-triton
# 包根目录
pip install . --no-build-isolation

# 或仅编插件
cmake -S standalone -B build/plugin -DLLVM_SYSPATH="$LLVM_SYSPATH"
cmake --build build/plugin -j --target libtriton_ascend
```

验证可用上层 `unittest/pytest_ut/test_add.py`（关闭 `TRITON_DEBUG`），不必依赖本目录下的探针脚本。
