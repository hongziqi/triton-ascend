# Triton Ascend Plugin (out-of-tree)

Out-of-tree **triton-ascend** extension aligned with [triton-ext](https://github.com/triton-lang/triton-ext).

## Layout

```text
triton-ascend-plugin-test/
├── cmake/ support/ ci/          # triton-ext build infrastructure
├── extensions/ascend/           # Ascend backend + C++ passes/dialects
│   ├── backend/                 # Python compiler / driver / runtime
│   ├── lib/ include/            # C++ passes & dialects
│   ├── AscendNPU-IR/            # BiShengIR
│   ├── python/triton_ascend/    # extend_with registration
│   ├── puncture/                # Step 3 probes
│   ├── pyproject.toml           # triton-ascend wheel
│   └── scripts/
├── docs/PUNCTURE.md             # Puncture plan & progress
└── Makefile
```

**Host triton is not built here.** Install a TRITON_EXT wheel separately:

```bash
# triton-ext nightly (L3 puncture host)
pip install triton-3.8.0+gitf893845b-*.whl --no-deps
```

## Quick start

### L1 — Python backend (entry_points)

```bash
pip install 'triton==3.7.0'   # or TRITON_EXT nightly for L3+
bash extensions/ascend/scripts/build_python_wheel.sh
pip install extensions/ascend/dist/triton_ascend-*.whl --force-reinstall --no-deps
PYTHONPATH= python extensions/ascend/scripts/l1_puncture_test.py
```

### L3 — Native plugin (extend_with)

Requires TRITON_EXT host (`extend_with` + `triton/include/`). See `docs/PUNCTURE.md`.

```bash
pip wheel extensions/ascend/puncture/loop_split_probe --no-build-isolation -w dist
pip install dist/triton_loop_split_probe-*.whl --no-deps
python extensions/ascend/scripts/l3_puncture_test.py
```
