# triton-ascend

Independent Ascend NPU backend for community **triton 3.7.1**.

Installable artifact: a **`triton-ascend` wheel** with only Ascend runtime
content (`backend`, `language`, `libtriton_ascend.so`). No community
`python/` / `third_party/{nvidia,amd}` and no forked `libtriton.so`.

## Architecture (scheme 2)

```
@triton.jit
    │  community triton  (TTIR frontend, host MLIRContext)
    ▼
make_ttir                     # host passes.ttir (inliner/cse/unroll)
    │  str(mod)  text boundary  (cannot share ModuleOp* across DSOs)
    ▼
backend/compiler.py          # Python owns pass order, same as candy-dev:
    ascend.ir.context / parse / pass_manager
    ascend.passes.ttir.add_triton_control_flow_opt(pm)
    ascend.passes.ttir.add_triton_to_structure(pm, ...)
    ...
    pm.run(owned_mod, "ttir_to_linalg")
    │  adapter MLIR text
    ▼
bishengir-compile             # process-external, produces npubin
    ▼
NPUDriver.load_binary         # CANN runtime
```

C++ in `libtriton_ascend.so` only:

- private `MLIRContext` + dialect registration (`standalone/ir.cpp`)
- nanobind `ir.*` and `passes.ttir.add_*` (`standalone/bindings.cpp`)
- Ascend / BiShengIR pass *implementations* from `lib/` + `include/`

There is no C++ “pipeline blob”. `pipeline.py` is a text→text test helper only.

`ascend.ir.ascendnpu_ir_builder` is a thin `parse_attr` / `set_attr` helper
for `hacc.target`. The fork-host CANN codegen builder (`TritonOpBuilder`
subclass) is not part of this tree.

Triton dialect sources come from git submodule `third_party/triton`
(pinned to **v3.7.1 / f797708**, matching `pip install triton==3.7.1`).
Do not pin main's `76e2689` (LLVM bump on main): that tree already uses a
newer `SideEffects::Resource` / `PropertyRef` API than
`LLVM_INSTALL_PREFIX_3_7_PATCH`. Compile the submodule as-is into
`libtriton_ascend.so` (private `MLIRContext`). This is not a community
source patch: candy-dev's `triton-ascend-3.7.0.patch` is fork-only
(cmake / `triton-opt` / Python frontend / HF32) and is not applied here.

## Layout

```
setup.py  pyproject.toml  __init__.py  pipeline.py
backend/                 compiler, driver, runtime
language/                cann + kernels (sys.modules inject)
_C/                      libtriton_ascend.so
standalone/              cmake + bindings + private MLIRContext
  CMakeLists.txt         cmake -S entry
  compile_sources.cmake  include(): TritonIR / BiShengIR / Ascend OBJECTs
  bindings.cpp           nanobind
  ir.h ir.cpp            private context
third_party/triton       community submodule (IR compiled into the .so as-is)
lib/  include/           Ascend pass/dialect sources
AscendNPU-IR/            BiShengIR sources
cmake/                   AscendPasses.cmake (OBJECT helpers)
```

## Build / install

Same LLVM as host triton (`LLVM_SYSPATH`). For 3.7.1 use
`LLVM_INSTALL_PREFIX_3_7_PATCH` (llvm-install-3-7_ac5dc54_patch).

CMake writes into `<repo>/build/` (CMakeCache, object files). The native
`.so` is copied to `_C/libtriton_ascend.so`. This tree is scheme 2 only:
no `TRITON_ASCEND_PLUGIN_MODE`.

```bash
git -C third_party/triton checkout f797708c06   # v3.7.1, not 76e2689
conda activate plugin2
cd ~/workspace/main/triton-ascend-plugin-test-2
export LLVM_SYSPATH="${LLVM_INSTALL_PREFIX_3_7_PATCH}"
pip install triton==3.7.1
```

Editable (dev) install — uses the current env, does not re-install triton:

```bash
pip install -e . --no-build-isolation --no-deps
```

Independent wheel / site-packages install (same cmake path as above;
`bdist_wheel` is a setuptools command, not a function in `setup.py`):

```bash
python3 setup.py bdist_wheel    # -> dist/triton_ascend-*.whl
python3 setup.py install        # compile + install into this Python
```

## Verify

```bash
python - <<'PY'
import triton, triton_ascend
from triton.backends import backends
n = triton_ascend._load_native()
print("backends:", sorted(backends))
print("owned_passes:", getattr(n, "__triton_ascend_has_owned_passes__", None))
print("passes.ttir:", hasattr(triton_ascend.passes.ttir, "add_triton_to_linalg"))
PY

cd unittest/pytest_ut
python -m pytest test_add.py --assert=plain -q --tb=short
```
