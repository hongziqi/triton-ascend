# standalone/ — scheme 2 native build

Produces `_C/libtriton_ascend.so`. Host Triton is an unmodified
`pip install triton==3.7.1` wheel.

```
CMakeLists.txt          cmake -S standalone  (LLVM, nanobind, MODULE)
compile_sources.cmake   include() of TritonIR + BiShengIR + Ascend OBJECTs
bindings.cpp            nanobind: triton_ascend.ir / passes.ttir  (RTTI on)
ir.h  ir.cpp            private MLIRContext, parse, add_*         (RTTI off)
```

`compile_sources.cmake` is not a `CMakeLists.txt` because it is `include()`d
into this project. Triton / BiShengIR / Ascend sources live outside this
folder; a nested `CMakeLists.txt` is for `add_subdirectory(dir)` when
sources are under `dir/`.

`bindings.cpp` is not a pipeline and not the CANN op builder.
`ir::Builder` (`ascendnpu_ir_builder` in Python) only does `parse_attr`.
Pass order is `backend/compiler.py`.
